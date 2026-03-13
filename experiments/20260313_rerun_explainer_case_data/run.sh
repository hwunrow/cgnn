#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main_new_model.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

MODEL="cdcrnn_hospital_hs16"
DATA="cdcrnn/case_advan_plus_full_cdcrnn"
TRAIN_LR="1e-5"
TRAIN_EPOCHS="10000"
EXPLAIN_LR="0.01"
EXPLAIN_EP="2000"

HYDRA_PARAMS="data=${DATA} model=${MODEL} \
        training.learning_rate=${TRAIN_LR} \
        training.num_epochs=${TRAIN_EPOCHS} \
        explain.lr=${EXPLAIN_LR} \
        explain.epochs=${EXPLAIN_EP} \
        version=${VERSION}"

JOB_NAME="case_advan_plus_full_cdcrnn_lr${TRAIN_LR}_epochs${TRAIN_EPOCHS}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.err"

echo "Submitting: ${JOB_NAME}"
echo "  Parameters: ${HYDRA_PARAMS}"
echo "  Log: ${LOG_FILE}"

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