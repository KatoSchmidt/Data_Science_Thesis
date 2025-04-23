#!/bin/bash
#SBATCH --job-name=sinenet_full_run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00

# Modules laden
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Virtual environment activeren
source ~/the_well/the_well_venv/bin/activate

# Naar projectfolder
cd ~/the_well_sinenet

# Run de training met specifieke config
srun python -m the_well.benchmark.train 

