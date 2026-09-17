"""V5 = unchanged V2 segmentation + the same auxiliary task used by V4."""
from .model_v2 import GlobalLocalSegmentationModelV2
from .diffusion_multitask import DiffusionMultitaskModel


class GlobalLocalSegmentationModelV5(DiffusionMultitaskModel):
    model_version = 'v5_v2_boundary_diffusion'

    def __init__(self, num_classes, token_dim=1024, context_dim=1024, channels=64,
                 diffusion_channels=32, diffusion_steps=1000, **segmentation_options):
        segmentation = GlobalLocalSegmentationModelV2(
            num_classes=num_classes, token_dim=token_dim, context_dim=context_dim,
            channels=channels, **segmentation_options)
        super().__init__(segmentation, 'global_initial', channels, num_classes,
                         diffusion_channels, diffusion_steps)
