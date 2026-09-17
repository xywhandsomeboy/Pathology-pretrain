"""Conditional image DDPM training and deterministic DDIM reconstruction.

No mask or clean RGB bypass enters the denoiser. Its inputs are x_t, timestep,
pooled segmentation semantics and predicted segmentation probabilities.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def cosine_alpha_bar(steps):
    if steps < 2:
        raise ValueError('diffusion steps must be at least two')
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=.999)
    return (1 - beta).cumprod(0).float()


class TimeResidualBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time)[:, :, None, None]
        return x + self.conv2(F.silu(self.norm2(h)))


class ConditionalImageDiffusion(nn.Module):
    def __init__(self, semantic_channels, num_classes, channels=32, steps=1000,
                 condition_grid=14):
        super().__init__()
        if channels < 8 or channels % 8 or condition_grid < 1:
            raise ValueError('channels must be a positive multiple of 8; grid positive')
        self.steps = steps
        self.condition_grid = condition_grid
        self.register_buffer('alpha_bar', cosine_alpha_bar(steps))
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406])[None,:,None,None])
        self.register_buffer('image_std', torch.tensor([.229, .224, .225])[None,:,None,None])
        time_dim = channels * 4
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        self.condition_projection = nn.Conv2d(semantic_channels + num_classes, channels * 4, 1)
        self.input = nn.Conv2d(3, channels, 3, padding=1)
        self.high = TimeResidualBlock(channels, time_dim)
        self.down1 = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.middle = TimeResidualBlock(channels * 2, time_dim)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1)
        self.low = TimeResidualBlock(channels * 4, time_dim)
        self.up1 = nn.Conv2d(channels * 6, channels * 2, 3, padding=1)
        self.refine1 = TimeResidualBlock(channels * 2, time_dim)
        self.up2 = nn.Conv2d(channels * 3, channels, 3, padding=1)
        self.refine2 = TimeResidualBlock(channels, time_dim)
        self.output = nn.Conv2d(channels, 3, 3, padding=1)

    def clean_image(self, normalized_image):
        # JointPatchSegmentationDataset uses these ImageNet statistics. The
        # target is the same geometrically/color-augmented patch seen by Stage1.
        return ((normalized_image.float() * self.image_std + self.image_mean).clamp(0, 1) * 2 - 1).detach()

    def condition(self, semantic, logits):
        size = tuple(min(self.condition_grid, s) for s in semantic.shape[-2:])
        semantic = F.adaptive_avg_pool2d(semantic, size)
        probability = F.adaptive_avg_pool2d(logits.float().softmax(1), size)
        return self.condition_projection(torch.cat((semantic, probability.to(semantic.dtype)), 1))

    def _time(self, timesteps):
        half = self.time_dim // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
        angles = timesteps.float()[:,None] * frequency[None]
        return self.time_mlp(torch.cat((angles.cos(), angles.sin()), 1))

    def predict_noise(self, noisy, timesteps, condition):
        time = self._time(timesteps)
        h = self.high(self.input(noisy), time)
        middle = self.middle(self.down1(h), time)
        low = self.down2(middle)
        low = self.low(low + F.interpolate(condition, size=low.shape[-2:], mode='bilinear', align_corners=False), time)
        up = self.up1(torch.cat((F.interpolate(low, size=middle.shape[-2:], mode='bilinear', align_corners=False), middle), 1))
        up = self.refine1(up, time)
        up = self.up2(torch.cat((F.interpolate(up, size=h.shape[-2:], mode='bilinear', align_corners=False), h), 1))
        return self.output(self.refine2(up, time))

    def diffuse(self, clean, timesteps, noise):
        alpha = self.alpha_bar[timesteps][:,None,None,None]
        return alpha.sqrt() * clean + (1-alpha).sqrt() * noise

    def training_output(self, image, semantic, logits, *, timesteps=None, noise=None):
        clean = self.clean_image(image)
        if timesteps is None:
            timesteps = torch.randint(self.steps, (image.shape[0],), device=image.device)
        if noise is None:
            noise = torch.randn_like(clean)
        if timesteps.shape != (image.shape[0],) or noise.shape != clean.shape:
            raise ValueError('timestep/noise shapes do not match the batch')
        condition = self.condition(semantic, logits)
        noisy = self.diffuse(clean, timesteps, noise)
        predicted_noise = self.predict_noise(noisy, timesteps, condition).float()
        alpha = self.alpha_bar[timesteps][:,None,None,None]
        # Unclipped x0 preserves reconstruction gradients. Losses multiply its
        # error by sqrt(alpha_bar), avoiding high-noise timestep amplification.
        reconstructed = (noisy - (1-alpha).sqrt() * predicted_noise) / alpha.sqrt()
        return {'predicted_noise': predicted_noise, 'noise': noise.detach(),
                'reconstruction': reconstructed, 'clean': clean,
                'sqrt_alpha': alpha.sqrt(), 'timesteps': timesteps}

    @torch.no_grad()
    def sample(self, semantic, logits, output_size, *, sampling_steps=50, seed=42):
        """Conditional reconstruction from Gaussian noise, DDIM eta=0.

        This is conditioned on an observed patch, not unconditional synthesis.
        Masks and the source RGB tensor are not sampler inputs.
        """
        if self.training:
            raise RuntimeError('Call eval() before diffusion sampling')
        if not 2 <= sampling_steps <= self.steps:
            raise ValueError('sampling_steps must be between 2 and diffusion steps')
        condition = self.condition(semantic, logits)
        generator = torch.Generator(device=semantic.device).manual_seed(seed)
        x = torch.randn((semantic.shape[0], 3, *output_size), device=semantic.device, generator=generator)
        schedule = torch.linspace(self.steps-1, 0, sampling_steps, device=x.device).round().long()
        for index, timestep in enumerate(schedule):
            t = timestep.expand(x.shape[0])
            alpha = self.alpha_bar[timestep]
            eps = self.predict_noise(x, t, condition).float()
            x0 = ((x - (1-alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1, 1)
            if index == sampling_steps-1:
                x = x0
            else:
                previous_alpha = self.alpha_bar[schedule[index+1]]
                eps = (x - alpha.sqrt() * x0) / (1-alpha).sqrt()
                x = previous_alpha.sqrt() * x0 + (1-previous_alpha).sqrt() * eps
        return ((x + 1) / 2).clamp(0, 1)
