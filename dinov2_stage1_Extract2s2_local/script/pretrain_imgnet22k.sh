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
  --ngpus=2 \
  --partition=gpu \
  --eval-only \
  --config-file dinov2/configs/train/vitl16_short_imgnet22k.yaml \
  --output-dir dinov2/results/pretrained_s1_embeddings-val-1007 \
  train.dataset_path=ImageNet22k:root=/home/li_yu/Proj04_he/HE_DNAMeth/multimodel_data/HE/ImageNet_like/imagenet22k:extra=/home/li_yu/Proj04_he/HE_DNAMeth/multimodel_data/HE/ImageNet_like/imagenet22k_extra