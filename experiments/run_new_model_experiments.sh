#!/bin/bash
# Submit CDCRNN experiments: 2 data sources x 3 mobility sources
# Sweeps over horizon, learning rate, and hidden size with log-log transforms

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main_new_model.sh"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

declare -a DATA_SOURCES=("hospital" "case")
declare -a MOBILITY_SOURCES=("advan" "advan_plus" "safegraph")
HORIZONS=(1 2)
LEARNING_RATES=("1e-5" "1e-4")
HIDDEN_SIZES=(16 32 64)

JOB_IDS=()
EXPERIMENT_NAMES=()

TOTAL=$(( ${#DATA_SOURCES[@]} * ${#MOBILITY_SOURCES[@]} * ${#HORIZONS[@]} * ${#LEARNING_RATES[@]} * ${#HIDDEN_SIZES[@]} ))

echo "=========================================="
echo "Submitting CDCRNN Experiments"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Total experiments: ${TOTAL}"
echo ""

for data_source in "${DATA_SOURCES[@]}"; do
  for mobility_source in "${MOBILITY_SOURCES[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for hs in "${HIDDEN_SIZES[@]}"; do
            VERSION="${data_source}_${mobility_source}_cdcrnn_h${horizon}_lr${lr}_hs${hs}"
            DATA_CONFIG="${data_source}_${mobility_source}_cdcrnn"
            MODEL_CONFIG="cdcrnn_${data_source}"

            LOG_FILE="${LOG_DIR}/cdcrnn_${VERSION}_${TIMESTAMP}.log"
            ERR_FILE="${LOG_DIR}/cdcrnn_${VERSION}_${TIMESTAMP}.err"

            echo "Submitting: ${VERSION}"
            echo "  Log: ${LOG_FILE}"

            JOB_ID=$(sbatch \
                --job-name="cdcrnn_${VERSION}" \
                --output="${LOG_FILE}" \
                --error="${ERR_FILE}" \
                --export=ALL,HORIZON=${horizon},LEARNING_RATE=${lr},HIDDEN_SIZE=${hs} \
                "${SLURM_SCRIPT}" \
                "data=${DATA_CONFIG}" \
                "model=${MODEL_CONFIG}" \
                "data.mobility_cutoff=1000" \
                2>&1 | grep -oP '\d+')

            if [ -n "${JOB_ID}" ]; then
                JOB_IDS+=("${JOB_ID}")
                EXPERIMENT_NAMES+=("${VERSION}")
                echo "  Job ID: ${JOB_ID}"
            else
                echo "  ERROR: Failed to submit job"
            fi
            echo ""
        done
      done
    done
  done
done

TRACKING_FILE="${LOG_DIR}/jobs_cdcrnn_${TIMESTAMP}.txt"
{
    echo "CDCRNN Experiment Submission Summary"
    echo "Timestamp: ${TIMESTAMP}"
    echo "Total jobs submitted: ${#JOB_IDS[@]}"
    echo ""
    echo "Job ID | Experiment Name"
    echo "------ | ----------------"
    for i in "${!JOB_IDS[@]}"; do
        echo "${JOB_IDS[$i]} | ${EXPERIMENT_NAMES[$i]}"
    done
} > "${TRACKING_FILE}"

echo "=========================================="
echo "Submission Complete"
echo "=========================================="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Tracking file: ${TRACKING_FILE}"
echo ""
echo "To check job status:"
echo "  squeue -u nhw2114"
echo ""
echo "To cancel all jobs:"
echo "  scancel ${JOB_IDS[*]}"
