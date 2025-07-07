#!/bin/bash
#SBATCH -J InformationRetrieval             # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --gres=gpu:H100:1         # Number of GPU
#SBATCH --mem-per-gpu=40GB                       # Memory per core/
#SBATCH -t3:00:00                               # Duration
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

JAVA_HOME=/usr/lib/jvm/jre
echo "[$(date)] Setting JAVA_HOME to $JAVA_HOME"
export JAVA_HOME=$JAVA_HOME

# Activate venv
echo "[$(date)] Activate checknum venv"
module load python/3.12.5
source /storage/scratch1/1/mheil7/checknum/bin/activate

# Run your Python script
python run_information_retrieval.py --config_path config/reranker/run_config_reranker_pace.yml