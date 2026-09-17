"""Composition wrapper: unchanged segmentation model plus auxiliary diffusion."""
from torch import nn
import torch
from .boundary_diffusion import ConditionalImageDiffusion


class DiffusionMultitaskModel(nn.Module):
    def __init__(self, segmentation, semantic_key, semantic_channels, num_classes,
                 diffusion_channels=32, diffusion_steps=1000):
        super().__init__()
        self.segmentation = segmentation
        self.semantic_key = semantic_key
        self.diffusion = ConditionalImageDiffusion(semantic_channels, num_classes,
                                                   channels=diffusion_channels, steps=diffusion_steps)

    def forward(self, raw_patch, dense_tokens, global_context,
                return_features=False, *, return_auxiliary=False,
                timesteps=None, noise=None):
        if not return_auxiliary and not return_features:
            # Exact base segmentation path; no sampling and no extra RNG use.
            return self.segmentation(raw_patch, dense_tokens, global_context)
        features = self.segmentation(raw_patch, dense_tokens, global_context, return_features=True)
        output = dict(features) if return_features else {'logits': features['logits']}
        if return_auxiliary:
            output['diffusion'] = self.diffusion.training_output(
                raw_patch, features[self.semantic_key], features['logits'],
                timesteps=timesteps, noise=noise)
        return output

    @torch.no_grad()
    def generate_reconstruction(self, raw_patch, dense_tokens, global_context,
                                sampling_steps=50, seed=42):
        if self.training:
            raise RuntimeError('Call eval() before reconstruction')
        features = self.segmentation(raw_patch, dense_tokens, global_context, return_features=True)
        image = self.diffusion.sample(features[self.semantic_key], features['logits'],
                                      raw_patch.shape[-2:], sampling_steps=sampling_steps, seed=seed)
        return {'logits': features['logits'], 'reconstruction': image}
