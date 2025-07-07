#!/bin/bash
#SBATCH -J NLI-FULL                                  # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --gres=gpu:V100:1 -C V100-32GB            # Number of GPU
#SBATCH --cpus-per-task=12                       # Number of CPU
#SBATCH --mem-per-gpu=32G                       # Memory of GPU
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

HF_HOME=/storage/scratch1/1/mheil7/HF_HOME
echo "[$(date)] Setting huggingface model cache to $HF_HOME"
export HF_HOME=$HF_HOME

# Activate venv
echo "[$(date)] Activate checknum venv"
module load python/3.12.5
source /storage/scratch1/1/mheil7/checknum/bin/activate

# Run your Python script 
python run_nli.py -f --config_path config/encoder_classifier/pace_our_data_long_context.yml