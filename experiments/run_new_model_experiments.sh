#!/bin/bash
# Submit 6 CDCRNN experiments: 2 data sources x 3 mobility sources
# Uses new _cdcrnn configs only (same dates: 07/13/2020 - 12/31/2022)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main_new_model.sh"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

declare -a DATA_SOURCES=("hospital" "case")
declare -a MOBILITY_SOURCES=("advan" "advan_plus" "safegraph")

JOB_IDS=()
EXPERIMENT_NAMES=()

echo "=========================================="
echo "Submitting CDCRNN Experiments (6 jobs)"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Total experiments: 6"
echo ""

for data_source in "${DATA_SOURCES[@]}"; do
    for mobility_source in "${MOBILITY_SOURCES[@]}"; do
        VERSION="${data_source}_${mobility_source}_cdcrnn"
        DATA_CONFIG="${data_source}_${mobility_source}_cdcrnn"
        MODEL_CONFIG="gcn_${data_source}"

        LOG_FILE="${LOG_DIR}/cdcrnn_${VERSION}_${TIMESTAMP}.log"
        ERR_FILE="${LOG_DIR}/cdcrnn_${VERSION}_${TIMESTAMP}.err"

        HYDRA_PARAMS="data=${DATA_CONFIG} model=${MODEL_CONFIG} data.mobility_cutoff=1000"

        echo "Submitting: ${VERSION}"
        echo "  Parameters: ${HYDRA_PARAMS}"
        echo "  Log: ${LOG_FILE}"

        JOB_ID=$(sbatch \
            --job-name="cdcrnn_${VERSION}" \
            --output="${LOG_FILE}" \
            --error="${ERR_FILE}" \
            "${SLURM_SCRIPT}" \
            ${HYDRA_PARAMS} \
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
