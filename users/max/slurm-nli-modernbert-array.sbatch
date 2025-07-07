#!/bin/bash
#SBATCH -J NLI_HyperparameterOptimization      # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --ntasks-per-node=6                 # Number of nodes and cores per node required
#SBATCH --gres=gpu:1 -C RTX6000
#SBATCH --mem-per-gpu=16G                       # Memory per core/
#SBATCH -t7:00:00                               # Duration
#SBATCH -q inferno                               # QOS Name (Jobs finishing in less than an hour -> ember)
#SBATCH -a 1-10%10                              # array job with 6 parallel jobs similtanously
#SBATCH --output=Report-%A_%a.log               # log std and error into output
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=mheil7@gatech.edu           # E-mail address for notifications
# Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users.

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
# Get inputs for array
file=$(ls config/generated_configs/*.yaml | sed -n ${SLURM_ARRAY_TASK_ID}p)
num_files=$(ls config/generated_configs/*.yaml | wc -l)
echo "[$num_files] Number of files forwarded to array Job"

# Run your Python script
conda run -n $CONDA_ENV python run_nli.py --config_path $file