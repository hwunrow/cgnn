# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CGNN (COVID Graph Neural Network) is a spatio-temporal graph neural network for COVID-19 forecasting using census region (CBSA) mobility data.

## Scientific Development Guidelines

### Strict Temporal Splitting
- Never use random shuffling for train/test splits. This is time-series data.
- Always split by time so that the validation sets must effectively be "the future" relatie to the training set

### Data Integrity
- Normalization: If used, `StandardScaler` and other scalers must be `fit` ONLY on the training split. Apply the transform to test data using training statistics.
- Target leakage: Ensure that the target variable is strictly excluded from the input features at time `t`.
- Graph topology: Ensure that the adjacency matrix `A_t` corresponds to mobility at time `t` (or `t-k` is using lagged mobility)

### Reproducibility
- All experiments must set global random seeds (`torch`, `numpy`) at the start of `main.py`
- If using `DCRNN`, ensure determinism flags (`torch.use_deterministic_algorithms`) are active if exact replication is required.

## Commands

### Installation
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

### Running the Pipeline
```bash
# Full pipeline (process data → train → evaluate → explain)
python main.py

# With different data/model configs
python main.py data=hospital_safegraph model=gcn_hospital

# Override hyperparameters
python main.py training.learning_rate=1e-4 training.num_epochs=10000

# Multi-run hyperparameter sweep
python main.py -m training.learning_rate=1e-5,1e-4,1e-3
```

### Running Tests
```bash
python -m unittest test/test_process_data.py
```

### Linting
```bash
flake8 src/
```
Code style: max line length 90, ignored rules: E731, W503, E741, E203

### SLURM Job Submission
```bash
sbatch main.sh
```

## Architecture

### Main Entry Point
- `main.py` - Hydra-based pipeline orchestrator running 4 steps: data processing → training → evaluation → GNN explanation

### Core Modules (`src/cgnn/`)
- `model.py` - GCN model with skip connections, RMSLELoss criterion
- `process_data.py` - Data processing pipeline creating PyTorch Geometric graphs
- `dataloader.py` - ConfigurableDatasetLoader for PyTorch Geometric
- `explain.py` - GNNExplainer for model interpretability

### Data Processing Modules
- `process_safegraph/` - SafeGraph mobility data processing
- `process_advan/` - Advan mobility data processing (see `process_advan/README.md` for download instructions)
- `process_xwalk.py` - CBSA/FIPS code crosswalk utilities

### Utilities (`src/cgnn/utils/`)
- `utils.py` - Date range generation (`get_date_range`), CBSA list helpers (`get_cbsa_list`, `get_cbsa_info`)
- `codebook.py` - CBSA title mappings, HHS region mappings

### Configuration (Hydra)
- Config path: `experiments/conf/`
- Data configs: `experiments/conf/data/` (hospital_advan, hospital_safegraph, case_advan, case_safegraph, hosp_advanplus)
- Model configs: `experiments/conf/model/` (gcn_hospital, gcn_case)

## Key Data Structures

- **Node features**: Hospital data (1 feature) or Case/Death data (16 features)
- **Edge weights**: Visitor aggregation from mobility data (Advan or SafeGraph)
- **Temporal structure**: Weekly snapshots (Monday-based date ranges)
- **Graph format**: PyTorch Geometric `Data` object with `train_mask`/`test_mask`

## Tech Stack
- PyTorch Geometric (GCNConv)
- PyTorch Geometric Temporal (DynamicGraphTemporalSignal)
- Hydra (configuration management)
- Requires CUDA 11.8+ for GPU training

