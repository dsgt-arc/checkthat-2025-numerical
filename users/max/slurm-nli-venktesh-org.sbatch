#!/bin/bash
#SBATCH -J NLI-FULL-VENK-ORG                                  # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --gres=gpu:RTX_6000:2            # Number of GPU
#SBATCH --cpus-per-task=12                       # Number of CPU
#SBATCH --mem-per-gpu=24G                       # Memory of GPU
#SBATCH -t0:240:00                              # Duration
#SBATCH -qinferno                                # QOS Name (Jobs finishing in less than an hour -> ember)
#SBATCH --output=Report-%j.log                         # Combined output and error messages file
#SBATCH --mail-type=NONE              # Mail preferences
#SBATCH --mail-user=mheil7@gatech.edu           # E-mail address for notifications
set -ue

# Change working directory
PACKAGE=~/p-dsgt_clef2025-0/checkthat-2025-numerical
cd $PACKAGE 
echo "[$(date)] Set SLURM_SUBMIT_DIR $PACKAGE"

echo "[$(date)] Load cuda"
module load cuda/12.6.1

# Activate venv
echo "[$(date)] Activate checknum venv"
module load python/3.12.5
source ~/scratch/checknum/bin/activate

# Run your Python script 
python run_nli.py -f --config_path config/encoder_classifier/pace_venktesh_org_data.yml