#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main_new_model.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

MODEL="cdcrnn_hospital_hs16"
CHECKPOINT_VERSION="hospital_advan_plus_full_cdcrnn_x_log1p_y_log1p_h1_lr1e-05_hs16_epochs20000"
TRAIN_LR="1e-5"
TRAIN_EPOCHS="5000"

EXPLAIN_LRS=("0.01" "0.001" "0.0001")
EXPLAIN_EPOCHS=("200" "500" "1000" "2000" "5000" "10000")

COUNT=0
for EXPLAIN_LR in "${EXPLAIN_LRS[@]}"; do
    for EXPLAIN_EP in "${EXPLAIN_EPOCHS[@]}"; do
        JOB_NAME="explainer_lr${EXPLAIN_LR}_epochs${EXPLAIN_EP}"
        CHECKPOINT_DIR="${REPO_DIR}/models/gcn_checkpoints/${JOB_NAME}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log"
        ERR_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.err"

        HYDRA_PARAMS="data=cdcrnn/hospital_advan_plus_full_cdcrnn model=${MODEL} \
            +skip_training=True \
            +checkpoint_version=${CHECKPOINT_VERSION} \
            training.learning_rate=${TRAIN_LR} \
            training.num_epochs=${TRAIN_EPOCHS} \
            explain.lr=${EXPLAIN_LR} \
            explain.epochs=${EXPLAIN_EP} \
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

        COUNT=$((COUNT + 1))
        echo ""
    done
done

echo "Submitted ${COUNT} jobs."
