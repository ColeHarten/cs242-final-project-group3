#!/bin/bash 
#SBATCH -c 16                # Number of cores (-c) 
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb
#SBATCH -t 1-00:00           # Runtime in D-HH:MM, minimum of 10 minutes 
#SBATCH -p gpu               # Partition to submit to 
#SBATCH --mem=99G 
#SBATCH -o train_%j.out   # File to which STDOUT will be written, %j inserts $
#SBATCH -e train_%j.err   # File to which STDERR will be written, %j inserts $

module load python/3.10.12-fasrc01
source activate NNBP-OSD

python3 train_segmentation.py --config configs/config_unet_ct_multi_att_dsv.json