#!/bin/bash
#SBATCH -J ClaimEmbeddingClustering             # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --ntasks-per-node=1                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH -t0:10:00                               # Duration
#SBATCH -qembers                                # QOS Name (Jobs finishing in less than an hour -> ember)


set -ue

# Activate Conda environment
CONDA_ENV=checkthat-numerical                   # Conda Env for Job
module load anaconda3                           # Load module dependencies
conda activate $CONDA_ENV
echo "[$(date)] Activated Conda environment $CONDA_ENV"


# Change working directory
PACKAGE=~/p-dsgt_clef2025-0/checkthat-2025-numerical
cd $PACKAGE 
echo "[$(date)] Set SLURM_SUBMIT_DIR $PACKAGE"

# Run your Python script
conda run -n $CONDA_ENV python run_generate_configs.py