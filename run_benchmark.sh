#!/bin/bash
#SBATCH -c 8                # Number of CPU cores
#SBATCH -t 1-12:59          # Runtime
#SBATCH --mem=99G           # Memory
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -p gpu
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err 


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
echo "Setup done. Starting benchmarking..."
python benchmarking.py --file checkpoints/experiment_unet_ct_multi_att_dsv/095_net_S.pth