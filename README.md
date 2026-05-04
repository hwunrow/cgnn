# CGNN — Spatio-temporal graph neural networks for COVID-19 forecasting

This repository accompanies the paper *{TBD: paper title}* ({TBD: authors}, {TBD: year}). It implements a CDCRNN (Diffusion Convolutional Recurrent Neural Network with skip connections) for weekly COVID-19 hospitalization and case forecasting at the U.S. Core-Based Statistical Area (CBSA) level. Inter-CBSA mobility from Advan Weekly Patterns Plus provides the dynamic graph topology, and a GNNExplainer-style edge-importance procedure (`DCRNNExplainer`) recovers a sparse, time-varying subgraph of mobility flows the model relies on. Cross-correlation between explainable-edge counts and the forecast target yields the lead-lag analysis reported in the paper.

## Citation

If you use this code or the accompanying data, please cite:

```bibtex
@article{TBD,
  title   = {TBD},
  author  = {Wunrow, Han Yong and TBD},
  journal = {TBD},
  year    = {TBD},
  doi     = {TBD},
}
```

The Zenodo archive accompanying this repository (DOI: TBD) bundles the preprocessed Advan mobility extract used as input.

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

### GPU runtime requirements
- Linux
- NVIDIA GPU with CUDA 11.8+
- PyTorch 2.1.*

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

## Notebook / figure mapping

All manuscript figures are produced by **`nb/final_manuscript_plots.ipynb`**, which reads the outputs that `python main.py` writes under `plots/{version}/`. Run the headline configurations in "Reproducing the paper" first.

| Notebook section | Manuscript |
|---|---|
| `# Figure 1 - CCF` | Cross-correlation between hospitalizations / cases and explainable-edge counts (lead-lag analysis). |
| `# Figure 2 - HHS plots` | Per-HHS-region aggregate time series and CCFs (4×5 panel grid). |
| `# Figure 3 - Maps and Mobility Matrix` | Explainable-subgraph heatmaps overlaid on geographic maps for three peak dates. |
| `# Supplementary` | Bimodal edge-importance distribution and ancillary diagnostics. |

## Repository structure

```
cgnn/
├── main.py                       # CDCRNN pipeline: data → train → evaluate → DCRNNExplainer
├── pyproject.toml                # uv-managed dependencies
├── uv.lock                       # pinned dependency versions
├── requirements.txt              # generated from uv.lock for pip users
├── environment.yml               # conda environment alternative
├── LICENSE                       # MIT
├── data/
│   ├── raw/                      # external inputs (see Data acquisition)
│   └── processed/                # generated PyG Data objects (created on first run)
├── src/cgnn/
│   ├── dataloader.py             # ConfigurableDatasetLoader (PyTorch Geometric)
│   ├── model.py                  # CDCRNN + GCN baseline
│   ├── explain.py                # DCRNNExplainer (edge-importance for dynamic graphs)
│   ├── plot.py                   # figure helpers
│   ├── process_data.py           # raw inputs → PyG temporal signal
│   ├── process_xwalk.py          # CBSA / FIPS / ZIP crosswalks
│   ├── process_advan/            # upstream Advan API ingestion (not required to reproduce)
│   └── utils/                    # codebook (CBSA / HHS region maps), helpers
├── experiments/
│   └── conf/                     # Hydra configs (config.yaml + data/, model/)
├── nb/
│   └── final_manuscript_plots.ipynb  # all paper figures
└── test/                         # unit tests for process_data
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
