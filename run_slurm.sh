#!/bin/bash

# --- SBATCH Directives (Defaults) ---
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --mem=0                         # Use all available memory on node
#SBATCH --time=24:00:00                 # Time limit
#SBATCH --exclusive                     # Request exclusive access to the node (optional, remove if unwanted)

# --- Script Body ---
set -e

# 1. Environment Setup
echo "Loading Conda environment..."
# Ensure this path is correct for your system
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate anomalib
echo "Environment loaded."

# 2. Run Training
# "$@" passes all arguments sent from submit_sweep.sh directly to python
# e.g., --model padim --dataset folder --root_dir ...
echo "Running Training Command:"
echo "python train.py $@"

srun python train.py "$@"

echo "Job finished successfully."