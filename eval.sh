#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --time=4:00:00        # <--- ADJUST THIS WALL TIME! It will now run for each num_trainings.
#SBATCH --job-name="eval_sequential_trainings" # More descriptive job name
#SBATCH --output="eval.out" # Unique output file for the entire SBATCH job

# Get k1 and k2 from command line arguments (commented out as not used here)

# Activate the desired Conda environment
source ~/.bashrc  # Make sure Conda is initialized in your shell
conda activate tf-gpu-jk

module load cuda/11.8

# Update PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/pystrum
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/neurite
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/voxelmorph
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/neurite-sandbox
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/voxelmorph-sandbox

# Define the list of num_trainings values to iterate through

python eval.py

