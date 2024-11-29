#!/bin/bash
#SBATCH -c 4                # Number of CPU cores (-c)
#SBATCH -t 1-12:59          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=99G           # Memory pool for all CPU cores
#SBATCH --array=1-1         # number of jobs (job array)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -p gpu_requeue
#SBATCH -o validation_%j.out
#SBATCH -e validation_%j.err 

# Load software
module load cuda/11.8.0-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
module load python/3.10.12-fasrc01

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate mamba env
source ~/.bashrc
source activate NNBP-OSD
nvidia-smi

# Run the Python scripts with the provided parameters
echo "Setup done. Starting training..."

python3 validation.py --config configs/config_unet_ct_multi_att_dsv.json