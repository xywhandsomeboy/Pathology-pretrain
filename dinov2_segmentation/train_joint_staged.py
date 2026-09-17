"""Joint training with separate GNN, Stage1 fusion, and DINO unfreezing.

Supports serial and torchrun via the existing parallel entry point. Existing
checkpoints with the legacy schedule are deliberately not silently re-labelled.
"""
import sys
from dinov2_segmentation.train_joint_parallel import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--unfreeze-schedule", "separate_gnn_fusion"])
