#!/bin/bash
#SBATCH -c 4                # Number of CPU cores (-c)
#SBATCH -t 1-12:59          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=99G           # Memory pool for all CPU cores
#SBATCH --array=1-1         # number of jobs (job array)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -p gpu_requeue
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err 
#SBATCH --exclude=holygpu8a31305,holygpu8a31402,holygpu8a31202,holygpu8a29106,holygpu8a25104

# Load software
module load cuda/11.8.0-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
module load python/3.10.12-fasrc01

# Activate mamba env
source ~/.bashrc
source activate NNBP-OSD
nvidia-smi

# Run the Python scripts with the provided parameters
echo "Setup done. Starting training..."

python3 train_segmentation.py --config configs/config_unet_ct_multi_att_dsv.json