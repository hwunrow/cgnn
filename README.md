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
