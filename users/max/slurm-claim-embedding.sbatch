#!/bin/bash
#SBATCH -J ClaimEmbeddingClustering             # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --gres=gpu:V100:1                       # GPU
#SBATCH --mem-per-gpu=16G                       # Memory per core/
#SBATCH -t0:30:00                               # Duration
#SBATCH -qembers                                # QOS Name (Jobs finishing in less than an hour -> ember)
#SBATCH --ooutput=Report-%j.log                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=mheil7@gatech.edu           # E-mail address for notifications
set -ue

# Change working directory
PACKAGE=~/p-dsgt_clef2025-0/checkthat-2025-numerical
cd $PACKAGE 
echo "[$(date)] Set SLURM_SUBMIT_DIR $PACKAGE"

# Activate Conda environment
CONDA_ENV=checkthat-numerical                   # Conda Env for Job
module load anaconda3                           # Load module dependencies
#conda env create -f environment.yml
conda activate $CONDA_ENV
echo "[$(date)] Activated Conda environment $CONDA_ENV"

# Run your Python script
conda run -n $CONDA_ENV python 00-run-eda-claims.py --config_path config/eda/run_config_modernbert-base_umap.yml
conda run -n $CONDA_ENV python 00-run-eda-claims.py --config_path config/eda/run_config_modernbert-base_pacmap.yml
conda run -n $CONDA_ENV python 00-run-eda-claims.py --config_path config/eda/run_config_modernbert-large_umap.yml
conda run -n $CONDA_ENV python 00-run-eda-claims.py --config_path config/eda/run_config_modernbert-large_pacmap.yml
conda run -n $CONDA_ENV python 00-run-eda-claims.py --config_path config/eda/run_config_all-MiniLM-L6-v2_umap.yml
conda run -n $CONDA_ENV python 00-run-eda-claims.py --config_path config/eda/run_config_all-MiniLM-L6-v2_pacmap.yml
