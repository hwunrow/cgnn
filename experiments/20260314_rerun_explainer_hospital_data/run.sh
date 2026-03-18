#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main_new_model.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

MODEL="cdcrnn_hospital_hs16"
CHECKPOINT_VERSION="hospital_advan_plus_cdcrnn_h1_lr1e-05_hs16"
TRAIN_LR="1e-5"
TRAIN_EPOCHS="5000"

EXPLAIN_LR="0.01"
EXPLAIN_EP="2000"

JOB_NAME="${CHECKPOINT_VERSION}_rerun_explainer_lr${EXPLAIN_LR}_epochs${EXPLAIN_EP}"
CHECKPOINT_DIR="${REPO_DIR}/models/gcn_checkpoints/${CHECKPOINT_VERSION}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.err"

HYDRA_PARAMS="data=cdcrnn/hospital_advan_plus_full_cdcrnn model=${MODEL} \
    +skip_training=True \
    +checkpoint_version=${CHECKPOINT_VERSION} \
    training.learning_rate=${TRAIN_LR} \
    training.num_epochs=${TRAIN_EPOCHS} \
    explain.lr=${EXPLAIN_LR} \
    explain.epochs=${EXPLAIN_EP} \
    data.end_date='12/31/2022' \
    version=${JOB_NAME} \
    hydra.run.dir=${CHECKPOINT_DIR}"

echo "Submitting: ${JOB_NAME}"
echo "  Parameters: ${HYDRA_PARAMS}"
echo "  Log: ${LOG_FILE}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"

JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --output="${LOG_FILE}" \
    --error="${ERR_FILE}" \
    --time=0-12:00 \
    --mem=150GB \
    "${SLURM_SCRIPT}" \
    ${HYDRA_PARAMS} \
    2>&1 | grep -oP '\d+')

if [ -n "${JOB_ID}" ]; then
    echo "  Job ID: ${JOB_ID}"
    echo "${JOB_ID} | ${JOB_NAME}" >> "${LOG_DIR}/jobs_${TIMESTAMP}.txt"
else
    echo "  ERROR: Failed to submit job"
fi
