# CGNN Experiment Management

This directory contains scripts for running and tracking CGNN experiments with different combinations of data sources, mobility sources, and mobility cutoffs.

## Experiment Configuration

Experiments are defined by combinations of:
- **data_source**: `hospital` or `case` (2 options)
- **mobility_source**: `advan` or `safegraph` (2 options)
- **mobility_cutoff**: `500` or `1000` (2 options)

**Total: 2 × 2 × 2 = 8 experiments**

Each experiment is automatically named: `{data_source}_{mobility_source}_cutoff{mobility_cutoff}`
Example: `hospital_advan_cutoff500`

## Running Experiments

### Option 1: Submit All Experiments as Separate Jobs (Recommended)

```bash
cd /burg/apam/users/nhw2114/repos/cgnn/experiments
./run_experiments.sh
```

This will:
- Submit 8 separate SLURM jobs (one per experiment)
- Generate unique log files for each experiment
- Create a tracking file with all job IDs

### Option 2: Single Job with Hydra Multi-Run

```bash
cd /burg/apam/users/nhw2114/repos/cgnn/experiments
./sweep.sh
```

**Note**: This approach may have limitations with date config selection. The separate jobs approach is recommended.

### Option 3: Run Individual Experiment

```bash
sbatch --job-name=cgnn_hospital_advan_cutoff500 \
       --output=/burg/home/nhw2114/cgnn_hospital_advan_cutoff500.log \
       --error=/burg/home/nhw2114/cgnn_hospital_advan_cutoff500.err \
       outputs/main.sh \
       data.data_source=hospital \
       data.mobility_source=advan \
       data.mobility_cutoff=500 \
       data_dates=hospital_advan
```

## Date Range Configuration

Date ranges for each data_source × mobility_source combination are configured in:
- `nb/configs/data_dates/hospital_advan.yaml`
- `nb/configs/data_dates/hospital_safegraph.yaml`
- `nb/configs/data_dates/case_advan.yaml`
- `nb/configs/data_dates/case_safegraph.yaml`

**Important**: Update these files with the appropriate date ranges for each combination based on your data availability.

## Output Organization

Each experiment saves outputs to:
- **Data**: `data/processed/{version}/`
- **Models**: `models/gcn_checkpoints/{version}/`
- **Plots**: `/plots/{version}/`

Where `{version}` is the auto-generated experiment name.

## Log Files

Experiment logs are stored in:
- `experiments/experiment_logs/`

Each experiment gets unique log files with timestamps:
- `cgnn_{version}_{timestamp}.log`
- `cgnn_{version}_{timestamp}.err`

## Checking Job Status

```bash
# View all your jobs
squeue -u nhw2114

# View specific job details
scontrol show job <job_id>
```

## Canceling Jobs

```bash
# Cancel all CGNN jobs
scancel -u nhw2114 --name=cgnn_*

# Cancel specific job
scancel <job_id>
```

