#!/bin/bash
#SBATCH --account=apam
#SBATCH -n 1
#SBATCH --time=0-01:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

# Load CUDA and environment (mirror main_new_model.sh)
module load cuda11.8/toolkit/11.8.0
source /burg-archive/home/nhw2114/.virtualenvs/tgt/bin/activate

REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
cd "${REPO_DIR}" || exit 1

sleep 10s

python experiments/cdcrnn_sweep_job.py \
  --transformation "${TRANSFORMATION}" \
  --target "${TARGET}" \
  --horizon "${HORIZON}" \
  --objective "${OBJECTIVE}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --out-dir "${OUT_DIR}" \
  --regions "1,2"

