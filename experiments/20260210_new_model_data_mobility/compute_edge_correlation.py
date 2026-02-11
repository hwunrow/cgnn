"""
Compute shifted cross-correlation between explainable edge counts
and aggregate case/hospitalization time series for CDCRNN experiments.

For each experiment directory in plots/*cdcrnn*:
  1. Read explain_count.csv (num_explainable_edges per snapshot)
  2. Read eval_metrics.csv (train/test loss, hyperparams)
  3. Load aggregate target time series via ConfigurableDatasetLoader
  4. Compute cross-correlation and find lag with max |correlation|
  5. Output CSV with experiment params, best lag, correlation, losses

Usage:
    python experiments/compute_edge_correlation.py
"""

import sys
import re
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cgnn.dataloader import ConfigurableDatasetLoader

PLOTS_DIR = ROOT / "plots"
CONF_DIR = ROOT / "experiments" / "conf" / "data"


def parse_experiment_dir(dirname):
    """Extract params from directory name like
    hospital_safegraph_cdcrnn_h1_lr0.0001_hs32."""
    if dirname.startswith("hospital_"):
        data_source = "hospital"
        rest = dirname[len("hospital_"):]
    elif dirname.startswith("case_"):
        data_source = "case"
        rest = dirname[len("case_"):]
    else:
        return None

    for mob in ("advan_plus", "advan", "safegraph"):
        if rest.startswith(mob + "_"):
            mobility_source = mob
            rest = rest[len(mob) + 1:]
            break
    else:
        return None

    m = re.match(r"cdcrnn_h(\d+)_lr([\d.e+-]+)_hs(\d+)$", rest)
    if not m:
        return None

    return {
        "data_source": data_source,
        "mobility_source": mobility_source,
        "target_horizon": int(m.group(1)),
        "learning_rate": m.group(2),
        "hidden_size": int(m.group(3)),
    }


def load_aggregate_ts(config_name):
    """Load data via ConfigurableDatasetLoader and return per-snapshot
    aggregate target (sum across all CBSAs per date)."""
    cfg_path = CONF_DIR / f"{config_name}.yaml"
    raw_cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.create({"data": raw_cfg})

    loader = ConfigurableDatasetLoader(cfg=cfg)

    date_col = loader.date_col
    target_col = loader.target_col
    dates = loader.dates

    agg = (
        loader.data_df
        .groupby(date_col)[target_col]
        .sum()
        .sort_index()
    )

    # Build array aligned to the loader's date range
    ts = np.array(
        [agg.get(d, 0.0) for d in dates], dtype=np.float64
    )
    return ts


def compute_best_lag(x, y):
    """Return (lag, normalized_correlation) at max |cross-correlation|."""
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    corr = signal.correlate(xc, yc, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    best_idx = np.argmax(np.abs(corr))
    norm = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    if norm > 0:
        return int(lags[best_idx]), float(corr[best_idx] / norm)
    return int(lags[best_idx]), 0.0


def main():
    exp_dirs = sorted(
        d for d in PLOTS_DIR.iterdir()
        if d.is_dir() and "cdcrnn" in d.name
    )
    print(f"Found {len(exp_dirs)} CDCRNN experiment directories\n")

    # Cache aggregate time series per unique config (6 combos)
    agg_cache = {}

    results = []
    for exp_dir in exp_dirs:
        dirname = exp_dir.name
        params = parse_experiment_dir(dirname)
        if params is None:
            print(f"SKIP (parse): {dirname}")
            continue

        explain_path = exp_dir / "explain_count.csv"
        metrics_path = exp_dir / "eval_metrics.csv"
        if not explain_path.exists() or not metrics_path.exists():
            print(f"SKIP (missing files): {dirname}")
            continue

        explain_df = pd.read_csv(explain_path)
        metrics_df = pd.read_csv(metrics_path)

        train_row = metrics_df[metrics_df["split"] == "train"]
        test_row = metrics_df[metrics_df["split"] == "test"]
        train_rmse = train_row["rmse"].values[0] if len(train_row) else np.nan
        test_rmse = test_row["rmse"].values[0] if len(test_row) else np.nan
        train_mae = train_row["mae"].values[0] if len(train_row) else np.nan
        test_mae = test_row["mae"].values[0] if len(test_row) else np.nan

        # Load or retrieve cached aggregate time series
        config_key = (
            f"{params['data_source']}_{params['mobility_source']}_cdcrnn"
        )
        if config_key not in agg_cache:
            print(f"Loading aggregate for {config_key}...")
            agg_cache[config_key] = load_aggregate_ts(config_key)
            print(
                f"  -> {len(agg_cache[config_key])} snapshots\n"
            )

        agg_ts = agg_cache[config_key]
        edge_ts = explain_df[
            "num_explainable_edges"
        ].values.astype(np.float64)

        # Align lengths (explain_count may cover a subset of snapshots)
        n = min(len(edge_ts), len(agg_ts))
        lag, corr = compute_best_lag(edge_ts[:n], agg_ts[:n])

        results.append({
            "experiment": dirname,
            "data_source": params["data_source"],
            "mobility_source": params["mobility_source"],
            "target_horizon": params["target_horizon"],
            "learning_rate": params["learning_rate"],
            "hidden_size": params["hidden_size"],
            "best_lag": lag,
            "normalized_corr": round(corr, 6),
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
        })
        print(f"  {dirname}: lag={lag}, corr={corr:.4f}")

    out_df = pd.DataFrame(results)
    out_path = ROOT / "experiments" / "cdcrnn_correlation_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
