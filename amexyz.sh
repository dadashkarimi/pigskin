#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=280G
#SBATCH --partition=ai
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:30:00         # Set expected wall time
#SBATCH --job-name="50"
#SBATCH --output="50.out"

# Get k1 and k2 from command line arguments

# Activate the desired Conda environment
source ~/.bashrc  # Make sure Conda is initialized in your shell
conda activate tf-gpu-jk

module load cuda/11.8

export PYTHONUNBUFFERED=1

python amexyz.py
#python train_fov.py --model gmm --num_dims 128 --use_original -lr 1e-5
#python train_seg.py --model gmm --num_dims 128 -lr 1e-6

