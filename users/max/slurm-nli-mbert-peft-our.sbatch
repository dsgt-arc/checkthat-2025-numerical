#!/bin/bash
#SBATCH -J NLI-PEFT                                  # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --gres=gpu:V100:2 -C V100-32GB           # Number of GPU
#SBATCH --cpus-per-task=16                       # Number of CPU
#SBATCH --mem-per-gpu=32G                       # Memory of GPU
#SBATCH -t0:480:00                              # Duration
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
python run_nli.py -f --config_path config/encoder_classifier/pace_our_data_long_context_peft.yml