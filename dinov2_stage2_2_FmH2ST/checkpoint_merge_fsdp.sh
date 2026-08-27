#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4           # 4 tasks * 4 nodes = 16
#SBATCH --ntasks-per-node=4   # 4 tasks on each node
#SBATCH --gres=gpu:4          # 4 GPUs per node
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err

torchrun --nproc_per_node=4 checkpoint_merge_fsdp.py