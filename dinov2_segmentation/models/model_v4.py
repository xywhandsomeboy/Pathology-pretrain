"""V4 = unchanged V1 segmentation + boundary-aware diffusion reconstruction."""
from .model import GlobalLocalSegmentationModel
from .diffusion_multitask import DiffusionMultitaskModel


class GlobalLocalSegmentationModelV4(DiffusionMultitaskModel):
    model_version = 'v4_v1_boundary_diffusion'

    def __init__(self, num_classes, token_dim=1024, context_dim=1024, channels=128,
                 drop_path_rate=0.1, diffusion_channels=32, diffusion_steps=1000,
                 **segmentation_options):
        segmentation = GlobalLocalSegmentationModel(
            num_classes=num_classes, token_dim=token_dim, context_dim=context_dim,
            channels=channels, drop_path_rate=drop_path_rate, **segmentation_options)
        super().__init__(segmentation, 'semantic', channels, num_classes,
                         diffusion_channels, diffusion_steps)
