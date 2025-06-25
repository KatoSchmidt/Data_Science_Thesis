#!/bin/bash
#SBATCH --job-name=swinnet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/the_well/the_well_venv/bin/activate
cd ~/the_well_sinenet

srun python -m the_well.benchmark.train

