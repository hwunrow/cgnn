#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main_new_model.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

MODEL="cdcrnn_hospital_hs16"
DATA="cdcrnn/flu_advan_plus_full_cdcrnn"
TRAIN_LRS=("5e-6" "1e-6" "5e-7" "1e-7")
TRAIN_EPOCHS="100000"
EXPLAIN_LR="0.01"
EXPLAIN_EP="2000"
SEASON_CSV="${SCRIPT_DIR}/flu_season_dates.csv"

for TRAIN_LR in "${TRAIN_LRS[@]}"; do
LR_TAG="${TRAIN_LR//-/neg}"
LR_TAG="${LR_TAG//./p}"

tail -n +2 "${SEASON_CSV}" | while IFS=',' read -r SEASON START_DATE END_DATE || [ -n "$SEASON" ]; do
    START_DATE="${START_DATE//$'\r'/}"
    END_DATE="${END_DATE//$'\r'/}"
    SEASON_TAG="${SEASON//-/_}"
    JOB_NAME="flu_advan_plus_full_cdcrnn_${SEASON_TAG}_lr${LR_TAG}"
    LOG_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log"
    ERR_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.err"

    HYDRA_PARAMS="data=${DATA} model=${MODEL} \
        training.learning_rate=${TRAIN_LR} \
        training.num_epochs=${TRAIN_EPOCHS} \
        explain.lr=${EXPLAIN_LR} \
        explain.epochs=${EXPLAIN_EP} \
        data.start_date=${START_DATE} \
        data.end_date=${END_DATE} \
        version=${JOB_NAME}"

    echo "Submitting: ${JOB_NAME} (season ${SEASON}, ${START_DATE} – ${END_DATE}, lr=${TRAIN_LR})"
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
    echo ""
done

done  # end TRAIN_LRS loop
