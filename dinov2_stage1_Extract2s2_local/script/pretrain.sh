#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4           # 4 tasks * 4 nodes = 16
#SBATCH --ntasks-per-node=4   # 4 tasks on each node
#SBATCH --gres=gpu:4          # 4 GPUs per node
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err
#SBATCH --chdir=/home/li_yu/Proj04_he/done_work/dinov2

python dinov2/run/train/train.py \
  --nnodes=1 \
  --ngpus=1 \
  --no-resume \
  --partition=gpu \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir dinov2/results/gcn_he0819_pretrain \
  train.dataset_path=ImageFolder:root=/home/li_yu/Proj04_he/HE_DNAMeth/multimodel_data/HE/ImageNet_like