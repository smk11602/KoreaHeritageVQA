#!/bin/bash
#SBATCH --job-name=vilt
#SBATCH --output=train_vilt.out
#SBATCH --gres=gpu:1

python ./models/VQAmodel/vilt.py