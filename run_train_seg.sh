#!/bin/bash
#SBATCH -c 8                # Number of CPU cores
#SBATCH -t 1-12:59          # Runtime
#SBATCH --mem=99G           # Memory
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH -p gpu
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err 

#SBATCH --mail-type=ALL               
#SBATCH --mail-user=charten@college.harvard.edu

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Activate mamba env
module load Mambaforge
mamba activate myenv

# Debug info
echo "CUDA version:"
nvcc --version
echo "Python version:"
python --version
echo "Python path:"
which python
nvidia-smi

# Run the Python scripts with the provided parameters
echo "Setup done. Starting training..."
python train_segmentation.py --config configs/config_unet_ct_multi_att_dsv.json
