# cgnn
## Deliverables
- Students report - [e6691-2024spring-project-cgnn-nhw2114](#deliverables)
- Students’ slides - [E6691.2024Spring.CGNN.nhw2114.presentationFinal](https://docs.google.com/presentation/d/1HNhcweCy0BCiZ24RSah8lsI6n9f_g-vdYaPZ-Pe3NV8/edit#slide=id.geb327816ab_0_0)
- Papers: [Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks (2020)](https://arxiv.org/abs/2007.03113)
- Github: [e6691-2024spring-project-cgnn-nhw2114](#deliverables)

## Installation

### With uv (recommended)
```
uv sync              # runtime deps only
uv sync --dev        # runtime + notebook/dev deps
```
`uv sync` reads `pyproject.toml` and `uv.lock` to create `.venv` with the exact pinned versions.

To use a venv at a custom path instead of `.venv`:
```
UV_PROJECT_ENVIRONMENT=/path/to/venv uv sync
```

To add a new dependency during development:
```
uv add <package>           # runtime
uv add --dev <package>     # dev/notebook only
```
This updates `pyproject.toml` and `uv.lock` automatically.

### With pip
```
pip install -r requirements.txt
pip install -e .
```
`requirements.txt` is generated from `uv.lock` via `uv export --format requirements-txt --no-hashes -o requirements.txt`. Regenerate it after any `uv add` / `uv remove`.

### With conda
```
conda env create -f environment.yml
```

Other requirements to run on GCP:
- Linux
- NVIDIA GPU
- PyTorch 2.1.*
- CUDA 11.8+

## Data acquisition

The pipeline expects raw inputs under `data/raw/`. Most files are fetched live by `src/cgnn/process_data.py` on first run; two large or access-restricted files must be downloaded manually.

### Fetched automatically

These resolve from URLs at runtime — no manual action required.

| File | Source |
|---|---|
| `time_series_covid19_confirmed_US.csv` | JHU CSSE COVID-19 repository — [github.com/CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19) |
| `time_series_covid19_deaths_US.csv` | JHU CSSE COVID-19 repository — same |
| `co-est2023-alldata.csv` (county population) | U.S. Census Bureau — [www2.census.gov/programs-surveys/popest](https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv). Cached at `data/raw/co-est2023-alldata.csv` if present. |

### Manual downloads required

Place each file at the exact path shown:

| Path | Source |
|---|---|
| `data/raw/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_20251026.csv` | HHS *COVID-19 Reported Patient Impact and Hospital Capacity by Facility*, snapshot dated 2025-10-26 — [healthdata.gov](https://healthdata.gov/) |
| `data/raw/mobility/advan_plus/all_advan_plus.csv` | **Zenodo archive accompanying this paper** (DOI: TBD). Preprocessed weekly Advan Weekly Patterns Plus aggregated to CBSA-pair visitor flows. Too large for git. |

The Advan extract is produced upstream from raw Dewey/Advan API downloads using scripts in `src/cgnn/process_advan/`. Those scripts are included for reference but **not required** to reproduce the paper — the aggregated CSV is provided directly via Zenodo.

After downloading, `data/raw/` should contain at minimum:

```
data/raw/
├── COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_20251026.csv  # manual (HHS)
└── mobility/
    └── advan_plus/
        └── all_advan_plus.csv                                                       # manual (Zenodo)
```

JHU and Census files will be fetched the first time you run `python main.py`.

## Reproducing the paper

### Headline configurations

The default Hydra config trains the primary COVID-19 hospitalization model from the paper. Other manuscript configurations select via Hydra overrides:

| Result | Command |
|---|---|
| Weekly COVID-19 hospital admissions (main result) | `python main.py` |
| Weekly COVID-19 case counts | `python main.py data=cdcrnn/case_advan_plus_full_cdcrnn model=cdcrnn_case_hs32` |
| Mobility-lag sensitivity (SI) | `python main.py data=cdcrnn/hospital_advan_plus_cdcrnn_moblag` |

Default = `data=cdcrnn/hospital_advan_plus_full_cdcrnn model=cdcrnn_hospital_hs16`. See `experiments/conf/config.yaml` for shared training/explainer hyperparameters (`learning_rate=1e-4`, `num_epochs=5000`, `explain.epochs=2000`).

### End-to-end recipe

From a clean clone to all paper outputs:

```bash
# 1. Clone & set up environment
git clone https://github.com/hwunrow/cgnn.git
cd cgnn
uv sync                                # creates .venv from uv.lock

# 2. Download the two manual data files (see "Data acquisition" above)
mkdir -p data/raw/mobility/advan_plus
# -> place data/raw/COVID-19_Reported_Patient_Impact_..._20251026.csv (HHS)
# -> place data/raw/mobility/advan_plus/all_advan_plus.csv             (Zenodo)

# 3. Train + evaluate + run DCRNNExplainer (default = hospital model)
uv run python main.py

# 4. (Optional) Reproduce the case-count and mobility-lag SI runs
uv run python main.py data=cdcrnn/case_advan_plus_full_cdcrnn model=cdcrnn_case_hs32
uv run python main.py data=cdcrnn/hospital_advan_plus_cdcrnn_moblag
```

JHU CSSE time series and the Census population file are fetched live on the first run (the Census file is then cached at `data/raw/co-est2023-alldata.csv` for subsequent runs).

### Outputs

For each run, `version` (set in the data config) determines output paths:

```
models/gcn_checkpoints/{version}/
├── cdcrnn_final_model.pt       # trained weights + best test loss
└── config.yaml                 # full Hydra config snapshot

plots/{version}/
├── eval_metrics.csv            # train/test RMSE + MAE in original space
├── {version}.pdf               # loss curves, aggregate forecast, 3 CBSA panels
├── {version}_explain.pdf       # explainable-edge count vs aggregate target
├── importance_masks.parquet    # per-edge importance per snapshot (DCRNNExplainer)
├── explain_count.csv           # explainable edges per snapshot
└── explain_edges.csv           # CBSA-pair edges above the importance threshold
```

### Hardware

A single CUDA 11.8+ GPU is recommended; CPU fallback works but the default `num_epochs=5000` plus `explain.epochs=2000` run is long.

### Paper figures

Figures in the manuscript are produced by notebooks under `nb/`. A figure-by-notebook mapping will be added before tagging the public release.

## Directory Tree Structure
```
e6691-2024spring-project-cgnn-nhw2114/
│
├── assets/                                            # Plots of processed data and model results
|
├── data/
│   ├── raw/                                           # Raw safegraph and covid data
│   └── processed/                                     # Processed data for PyTorch-Geometric format
│
├── src/
│   ├── experiments/                                   # Result csv's from experiments
|   ├── *.yaml                                         # Experiment yaml files for hyperparameter tuning
|   ├── *_experiment.py                                # Scripts to run experiments
|   ├── colab_process_safegraph_mobility.py            # Jupyter notebook for processing raw safegraph data
|   ├── model.py                                       # GNN model definitions
|   ├── process_data.py                                # Script to process raw data into PyTorch-Geometric format
│   └── main.ipynb                                     # Core jupyter notebook with model runs and outputs                        
|
├── test/                                              # Unit tests for data processing
|
├── utils/   
│   ├── codebook.py                                    # mapping dicts for borough and FIPS code
│   └── utils.py                                       # util functions for graph node mapping
│
├── requirements.txt                                   # List of Python dependencies for pip
├── environment.yml                                    # List of Python dependencies for conda
├── .flake8                                            # flake8 codestyle
└── README.md                                          # Project README file with an overview and setup instructions
```

## Usage
To reproduce all plots and results in the presentation and report run the following Jupyter notebook
```
src/main.ipynb
```
To rerun experiments
```
python src/cgnn_experiment.py
```

```
python src/a3tgcn_experiment.py
```
