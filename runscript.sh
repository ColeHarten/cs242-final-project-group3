#!/bin/bash 

#SBATCH -c 16                # Number of cores (-c) 

#SBATCH --gres=gpu:4

#SBATCH -t 1-00:00           # Runtime in D-HH:MM, minimum of 10 minutes 

#SBATCH -p gpu_requeue            # Partition to submit to 

#SBATCH --mem=128G 

#SBATCH -o myoutput_%j.out   # File to which STDOUT will be written, %j inserts $

#SBATCH -e myerrors_%j.err   # File to which STDERR will be written, %j inserts $

module load  Mambaforge 
source activate cs242  
python -u /n/home01/charten/cs242-final-project-group3/train_segmentation.py