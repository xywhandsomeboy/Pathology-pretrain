"""Dedicated V5 (V2 + boundary diffusion) joint-training entry."""
import sys
from dinov2_segmentation.train_joint_parallel import main as joint_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == '--decoder-version' or a.startswith('--decoder-version=') for a in argv):
        raise ValueError('This entry fixes decoder-version=v5; omit the version option')
    return joint_main(['--decoder-version', 'v5', *argv])


if __name__ == '__main__':
    main()
