#!/bin/bash
#SBATCH -c 4                # Number of CPU cores (-c)
#SBATCH -t 1-12:59           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=99G           # Memory pool for all CPU cores
#SBATCH --gres=gpu:1
#SBATCH -p gpu_requeue
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err 

# Load software
module load cuda/11.8.0-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
module load Mambaforge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate mamba env
mamba activate myenv
nvidia-smi

# Run the Python scripts with the provided parameters
echo "Setup done. Starting training..."

python3 train_segmentation.py --config configs/config_unet_ct_multi_att_dsv.json