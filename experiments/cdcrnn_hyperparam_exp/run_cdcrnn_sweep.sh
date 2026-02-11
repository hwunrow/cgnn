#!/bin/bash
# Submit CDCRNN sweep experiments over (transformation, target, horizon, objective)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/burg/apam/users/nhw2114/repos/cgnn"
SLURM_SCRIPT="${REPO_DIR}/experiments/cdcrnn_job.sh"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

# Good (transformation, target, objective) combinations
COMBOS=(
  "logp1 logp1 MAE"
  "logp1 logp1 RMSE"
)
HORIZONS=(1 2)
LEARNING_RATES=("1e-5" "1e-4")
HIDDEN_SIZES=(16 32 64)

JOB_IDS=()
EXP_NAMES=()

echo "=========================================="
echo "Submitting CDCRNN Sweep Experiments"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Total combinations: $(( ${#COMBOS[@]} * ${#HORIZONS[@]} * ${#LEARNING_RATES[@]} * ${#HIDDEN_SIZES[@]} ))"
echo ""

OUT_ROOT="${REPO_DIR}/nb/experiment_plots_${TIMESTAMP}"

for combo in "${COMBOS[@]}"; do
  read -r transformation target objective <<< "${combo}"
  for horizon in "${HORIZONS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for hs in "${HIDDEN_SIZES[@]}"; do
        EXP_NAME="${transformation}_${target}_h${horizon}_${objective}_lr${lr}_hs${hs}"
        OUT_DIR="${OUT_ROOT}"
        LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.out"
        ERR_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.err"

        echo "Submitting: ${EXP_NAME}"
        echo "  Out dir: ${OUT_DIR}"
        echo "  Log: ${LOG_FILE}"

        JOB_ID=$(sbatch \
          --job-name="cdcrnn_${EXP_NAME}" \
          --output="${LOG_FILE}" \
          --error="${ERR_FILE}" \
          --export=ALL,TRANSFORMATION=${transformation},TARGET=${target},HORIZON=${horizon},OBJECTIVE=${objective},LEARNING_RATE=${lr},HIDDEN_SIZE=${hs},OUT_DIR=${OUT_DIR} \
          "${SLURM_SCRIPT}" 2>&1 | grep -oP '\\d+')

        if [ -n "${JOB_ID}" ]; then
          JOB_IDS+=("${JOB_ID}")
          EXP_NAMES+=("${EXP_NAME}")
          echo "  Job ID: ${JOB_ID}"
        else
          echo "  ERROR: Failed to submit job"
        fi
        echo ""
      done
    done
  done
done

TRACKING_FILE="${LOG_DIR}/jobs_cdcrnn_sweep_${TIMESTAMP}.txt"
{
  echo "CDCRNN Sweep Experiment Submission Summary"
  echo "Timestamp: ${TIMESTAMP}"
  echo "Total jobs submitted: ${#JOB_IDS[@]}"
  echo ""
  echo "Job ID | Experiment Name"
  echo "------ | ---------------"
  for i in "${!JOB_IDS[@]}"; do
    echo "${JOB_IDS[$i]} | ${EXP_NAMES[$i]}"
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

