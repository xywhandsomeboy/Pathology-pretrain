"""High-resolution raw-patch branch."""

from torch import nn

from .blocks import ConvNeXtV2Block, HRFormerBlock, LayerNorm2d


class HighResolutionDetailEncoder(nn.Module):
    """Keep a single H/4 stream and alternate CNN/Transformer blocks.

    Unlike a classification backbone, this branch never constructs a deep
    low-resolution semantic pyramid. Its only downsampling is the shallow H/4
    stem, after which every block preserves the grid used for fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int = 7,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("detail depth must be at least 2")
        stem_channels = max(channels // 2, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, channels, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        probabilities = [drop_path_rate * index / max(depth - 1, 1) for index in range(depth)]
        blocks = []
        for index, probability in enumerate(probabilities):
            if index % 2 == 0:
                blocks.append(ConvNeXtV2Block(channels, drop_path_probability=probability))
            else:
                blocks.append(
                    HRFormerBlock(
                        channels,
                        num_heads=num_heads,
                        window_size=window_size,
                        drop_path_probability=probability,
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.output_norm = LayerNorm2d(channels)
        self.out_channels = channels
        self.output_stride = 4

    def forward(self, image):
        return self.output_norm(self.blocks(self.stem(image)))
