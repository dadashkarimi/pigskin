#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=180G
#SBATCH --partition=ai
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-18:30:00         # Set expected wall time
#SBATCH --job-name="128"
#SBATCH --output="128.out"

# Get k1 and k2 from command line arguments

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

# Start Jupyter notebook with dynamic k1 and k2 values
#python train_fov.py --model gmm --num_dims 96 --olfactory -lr 1e-6 -k1 $k1 -k2 $k2
#python train_seg_atlas.py --new_labels --model gmm --num_dims 192 -lr 1e-6
#python train_seg_atlas.py --model gmm --num_dims 96 -lr 1e-6
#python train_fov.py --model gmm --num_dims 128 -lr 1e-6
#python train_fov.py --model hmrf --num_dims 128 -lr 1e-6
#python train_fov.py --model hmrf --num_dims 192 -lr 1e-6
##python train_fov.py --model hmrf --num_dims 192 -lr 1e-6python train_fov.py --model gmm --num_dims 192 -nt 1 -lr 1e-6
#python train_fov.py --model gmm --num_dims 192 -nt 1 -lr 1e-6python train_fov.py --model gmm --num_dims 192 -nt 1 -lr 1e-6
#python train_fov.py --model gmm --num_dims 192 -nt 7 -unique -t2 -lr 1e-6
#python train_fov.py --model gmm --num_dims 192 -nt 18 -t2 -injury -lr 1e-6
#python train_fov.py --model gmm --num_dims 96 -nt 7 -unique -lr 1e-6
python train_fov.py --model gmm --num_dims 128 -lr 1e-6
#python train_fov.py --model gmm --num_dims 192 -nt 18 -injury -unique -lr 1e-6
#python train_fov.py --model gmm --num_dims 192 -sc 0.5 -lr 1e-6
#python train_fov.py --model gmm --num_dims 192 --use_original -lr 1e-5
#python train_fov.py --model gmm --num_dims 192 -t2 -lr 1e-6

