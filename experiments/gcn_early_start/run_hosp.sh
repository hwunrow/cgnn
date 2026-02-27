#!/bin/bash
# GCN hospital early start date: advan+safegraph × cutoff500+1000, start_date=03/22/2020

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

declare -a DATA_SOURCES=("hospital")
declare -a MOBILITY_SOURCES=("advan" "safegraph")
declare -a MOBILITY_CUTOFFS=(500 1000)

JOB_IDS=()
EXPERIMENT_NAMES=()

echo "=========================================="
echo "Submitting GCN Hospital Early Start Experiments"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Total experiments: 4"
echo ""

for data_source in "${DATA_SOURCES[@]}"; do
    for mobility_source in "${MOBILITY_SOURCES[@]}"; do
        for mobility_cutoff in "${MOBILITY_CUTOFFS[@]}"; do
            VERSION="${data_source}_${mobility_source}_cutoff${mobility_cutoff}_early"
            DATE_CONFIG="${data_source}_${mobility_source}"
            MODEL_CONFIG="gcn_${data_source}"

            LOG_FILE="${LOG_DIR}/cgnn_${VERSION}_${TIMESTAMP}.log"
            ERR_FILE="${LOG_DIR}/cgnn_${VERSION}_${TIMESTAMP}.err"

            HYDRA_PARAMS="data=${DATE_CONFIG} model=${MODEL_CONFIG} data.version=${VERSION} data.mobility_cutoff=${mobility_cutoff} data.start_date=03/22/2020"

            echo "Submitting: ${VERSION}"
            echo "  Parameters: ${HYDRA_PARAMS}"
            echo "  Log: ${LOG_FILE}"

            JOB_ID=$(sbatch \
                --job-name="cgnn_${VERSION}" \
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
done

TRACKING_FILE="${LOG_DIR}/jobs_${TIMESTAMP}.txt"
{
    echo "Experiment Submission Summary"
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
echo "To check job status: squeue -u nhw2114"
echo "To cancel all jobs:  scancel ${JOB_IDS[*]}"
