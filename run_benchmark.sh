#!/bin/bash
#SBATCH -c 8                # Number of CPU cores
#SBATCH -t 1-12:59          # Runtime
#SBATCH --mem=99G           # Memory
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -p gpu_requeue
#SBATCH -o benchmarking_%j.out
#SBATCH -e benchmarking_%j.err 


# Activate mamba env
module load Mambaforge
mamba activate myenv

nvidia-smi

# Run the Python scripts with the provided parameters
echo "Setup done. Starting benchmarking..."
python benchmarking.py --file checkpoints/experiment_unet_ct_multi_att_dsv_early_exit_v2/095_net_S.pth