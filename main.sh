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
#              main.sh data.data_source=hospital data.mobility_source=advan data.mobility_cutoff=500 data_dates=hospital_advan

# load cuda
module load cuda11.8/toolkit/11.8.0

# load conda env
conda init
conda activate cgnn
sleep 10s
# run main.py with all parameters passed as arguments
python /burg/apam/users/nhw2114/repos/cgnn/main.py "$@"