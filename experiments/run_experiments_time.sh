#!/bin/bash
# Master script to submit all 8 experiments as separate SLURM jobs
# Each experiment is a combination of:
#   - data_source: hospital or case (2 options)
#   - mobility_source: advan or safegraph (2 options)
#   - mobility_cutoff: 500 or 1000 (2 options)
# Total: 2 × 2 × 2 = 8 experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/main.sh"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Define all experiment combinations
declare -a DATA_SOURCES=( "case")
declare -a MOBILITY_SOURCES=("advan" "safegraph")
declare -a MOBILITY_CUTOFFS=(500 1000)

# Track submitted jobs
JOB_IDS=()
EXPERIMENT_NAMES=()

echo "=========================================="
echo "Submitting CGNN Experiments"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Total experiments: 8"
echo ""

# Submit jobs for all combinations
for data_source in "${DATA_SOURCES[@]}"; do
    for mobility_source in "${MOBILITY_SOURCES[@]}"; do
        for mobility_cutoff in "${MOBILITY_CUTOFFS[@]}"; do
            # Generate version name
            VERSION="${data_source}_${mobility_source}_cutoff${mobility_cutoff}_early"
            
            # Determine date config based on data_source and mobility_source
            DATE_CONFIG="${data_source}_${mobility_source}"
            MODEL_CONFIG="gcn_${data_source}"
            
            # Generate unique log file names
            LOG_FILE="${LOG_DIR}/cgnn_${VERSION}_${TIMESTAMP}.log"
            ERR_FILE="${LOG_DIR}/cgnn_${VERSION}_${TIMESTAMP}.err"
            
            # Build Hydra parameter string
            HYDRA_PARAMS="data=${DATE_CONFIG} model=${MODEL_CONFIG} data.version=${VERSION} data.mobility_cutoff=${mobility_cutoff} data.start_date=07/13/2020"
            
            # Submit SLURM job
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

# Save job tracking information
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
echo "To check job status:"
echo "  squeue -u nhw2114"
echo ""
echo "To cancel all jobs:"
echo "  scancel ${JOB_IDS[*]}"

