#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4           # 4 tasks * 4 nodes = 16
#SBATCH --ntasks-per-node=4   # 4 tasks on each node
#SBATCH --gres=gpu:4          # 4 GPUs per node
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err
#SBATCH --chdir=/home/li_yu/Proj04_he/done_work/dinov2

  python  dinov2/run/eval/test.py \
  --nnodes=1 \
  --ngpus=2 \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --no-resume \
  --eval-only \
  --partition=gpu \
  --output-dir dinov2/results/Grade/gcn5w_1112_pretrain_embeddings-1024-b4-mask75-finetune \
  train.dataset_path=ImageFolder:root=/home/li_yu/Proj04_he/HE_DNAMeth/multimodel_data/HE/ImageNet_like/Graph-1024-251111-thre0.8
  
 