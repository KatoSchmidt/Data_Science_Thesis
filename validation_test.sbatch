#!/bin/bash

#SBATCH --job-name=validation_pretrained
#SBATCH --output=logs/validation_%A.out
#SBATCH --error=logs/validation_%A.err
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

# Laad modules
# module purge
# module load 2024
# module load Python/3.12.3-GCCcore-13.3.0

# # Activeer je venv
# source ~/myenv/bin/activate

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/the_well/the_well_venv/bin/activate

# Ga naar je projectmap
cd ~/the_well_sinenet

# Run validatie
# srun python -u validation_model.py  --data_path ~/Thesis_DS/data --well_split_name valid --model unet_classic 
srun python -m the_well.benchmark.validation_test  --data_path ~/the_well/datasets --well_split_name valid --model sinenet
# srun python -u validation_model.py  --data_path ~/Thesis_DS/data --well_split_name test --model unet_classic 
srun python -m the_well.benchmark.validation_test --data_path ~/the_well/datasets --well_split_name test --model sinenet


echo "Json files best.pt fininshed!"
