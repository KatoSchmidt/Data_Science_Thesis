#!/bin/bash
#SBATCH --job-name=model_validation
#SBATCH --output=logs/validation_%A.out
#SBATCH --error=logs/Validation_%A.err
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/the_well/the_well_venv/bin/activate
cd ~/the_well_sinenet

OUTPUT_DIR=validation_json_files
mkdir -p $OUTPUT_DIR

srun python -m the_well.benchmark.model_validation \
  --data_path ~/the_well/datasets \
  --well_split_name test \
  --model "" \
  --checkpoint_path "" \
  --output_dir $OUTPUT_DIR \ 
  --run "run1"
