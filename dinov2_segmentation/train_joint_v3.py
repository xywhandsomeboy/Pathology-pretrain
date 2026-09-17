"""Dedicated V3 joint-training entry point; supports serial and torchrun/DDP.

Supply the usual train_joint_parallel options and a NEW output directory.
Decoder version is fixed to V3. No training is launched by importing this file.
"""
import sys
from dinov2_segmentation.train_joint_parallel import main as joint_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == '--decoder-version' or a.startswith('--decoder-version=') for a in argv):
        raise ValueError('train_joint_v3 fixes --decoder-version=v3; omit this option')
    return joint_main(['--decoder-version', 'v3', *argv])


if __name__ == '__main__':
    main()
