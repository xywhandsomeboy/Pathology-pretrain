#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err

# Maintained compatibility entry: Stage-1 training is Stage-1A spatial-fusion
# training. Machine-specific paths are supplied through environment variables.
set -euo pipefail
stage1_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${stage1_dir}/pretrain_stage1a.sh"
