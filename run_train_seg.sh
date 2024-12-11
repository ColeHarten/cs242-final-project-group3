#!/bin/bash
#SBATCH -c 4
#SBATCH -t 1-12:59
#SBATCH --mem=99G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o baseline_train_%j.out
#SBATCH -e baseline_train_%j.err

module load Mambaforge
mamba activate myenv 
nvidia-smi

# Run the Python scripts with the provided parameters
echo "Setup done. Starting training..."
python train_segmentation.py --config configs/config_unet_ct_multi_att_dsv.json
