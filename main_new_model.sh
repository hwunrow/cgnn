#!/bin/bash
#SBATCH --account=apam
#SBATCH -n 1                  # The number of cpu cores to use.
#SBATCH --time=0-05:00        # The time the job will take to run in D-HH:MM:SS
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
# Note: Job name and log files should be set by the submitting script
# This script accepts Hydra parameters as command-line arguments

# Accept command-line arguments for experiment parameters
# Usage: sbatch --job-name=cgnn_hospital_advan_cutoff500 \
#              --output=/burg/home/nhw2114/cgnn_hospital_advan_cutoff500.log \
#              --error=/burg/home/nhw2114/cgnn_hospital_advan_cutoff500.err \
#              main.sh data=hospital_advan model=gcn_hospital

# load cuda
module load cuda11.8/toolkit/11.8.0

# load virtual environment
source /burg-archive/home/nhw2114/.virtualenvs/tgt/bin/activate

REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
cd "${REPO_DIR}" || exit 1

sleep 10s
python "${REPO_DIR}/main_new_model.py" "$@"