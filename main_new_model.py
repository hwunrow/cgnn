# main_new_model.py
"""
CDCRNN pipeline: temporal dataloader -> train -> evaluate -> DCRNNExplainer.
Uses log-log transforms, per-node StandardScaler, and self-loops.
Supports configurable horizon, learning rate, and hidden size via CLI.

Usage:
    python main_new_model.py data=hospital_advan_cdcrnn model=cdcrnn_hospital
    python main_new_model.py data=case_safegraph_cdcrnn model=cdcrnn_case
    python main_new_model.py data=hospital_advan_cdcrnn model=cdcrnn_hospital \
        data.target_horizon=2 training.learning_rate=1e-4 model.cdcrnn.hidden_size=64
"""

import os
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from cgnn.dataloader import ConfigurableDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from cgnn.model import CDCRNN
from cgnn.explain import DCRNNExplainer
from cgnn.utils import save_config_to_directory
from cgnn.utils.codebook import TITLE_CBSA_MAP

REPO_DIR = "/burg/apam/users/nhw2114/repos/cgnn"

# Transforms that benefit from per-node StandardScaler (log_diff is self-normalizing).
_SCALER_TRANSFORMS = {"none", "log1p", "diff"}

FOCUS_CBSAS = ["35620", "14460", "45940"]


def plot_experiment_page(
    train_losses,
    test_losses,
    train_preds_orig,
    train_targets_orig,
    test_preds_orig,
    test_targets_orig,
    loader,
    version,
    pdf_path,
    explain_losses=None,
):
    """Create a one-page PDF with loss curves, aggregate forecast, and 3 CBSA plots."""
    import matplotlib.pyplot as plt

    train_preds_sum = train_preds_orig.sum(axis=1)
    train_targets_sum = train_targets_orig.sum(axis=1)
    test_preds_sum = test_preds_orig.sum(axis=1)
    test_targets_sum = test_targets_orig.sum(axis=1)

    dates = pd.to_datetime([d.strftime("%Y-%m-%d") for d in loader.dates])
    train_size = len(train_preds_sum)
    train_dates = dates[:train_size]
    test_dates = dates[train_size : train_size + len(test_preds_sum)]
    split_date = train_dates[-1] if len(train_dates) > 0 else test_dates[0]

    cbsa_to_idx = {cbsa: i for i, cbsa in enumerate(loader.cbsa_list)}

    has_explain = explain_losses is not None and len(explain_losses) > 0

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.0, 1.2, 1.2])

    # Training loss curves (top 2 rows if explainer present, else full column)
    if has_explain:
        ax_loss = fig.add_subplot(gs[:2, 0])
    else:
        ax_loss = fig.add_subplot(gs[:, 0])
    epochs_arr = np.arange(1, len(train_losses) + 1)
    ax_loss.plot(epochs_arr, train_losses, label="Train", color="#348ABD", linewidth=2)
    if test_losses:
        eval_epochs, eval_vals = zip(*test_losses)
        ax_loss.plot(
            eval_epochs,
            eval_vals,
            label="Test",
            color="#E24A33",
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(frameon=True, framealpha=0.9, loc="best")
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    if has_explain:
        ax_exp = fig.add_subplot(gs[2, 0])
        keys = ['total', 'dist', 'size', 'ent']
        colors = {'total': '#333333', 'dist': '#348ABD',
                  'size': '#E24A33', 'ent': '#988ED5'}
        labels = {'total': 'Total', 'dist': 'Fidelity',
                  'size': 'Size', 'ent': 'Entropy'}
        n_epochs = len(explain_losses[0]['total'])
        exp_epochs = np.arange(1, n_epochs + 1)
        for key in keys:
            stacked = np.array([h[key] for h in explain_losses])
            mean_curve = stacked.mean(axis=0)
            ax_exp.plot(
                exp_epochs, mean_curve, label=labels[key],
                color=colors[key], linewidth=1.5,
                linestyle='--' if key != 'total' else '-',
            )
        ax_exp.set_xlabel("Explainer Epoch")
        ax_exp.set_ylabel("Loss")
        ax_exp.set_title("Explainer Loss (mean over snapshots)")
        ax_exp.legend(frameon=True, framealpha=0.9, loc="best", fontsize=7)
        ax_exp.grid(True, linestyle=":", alpha=0.6)

    # Aggregate forecast
    ax_agg = fig.add_subplot(gs[:, 1])
    ax_agg.plot(train_dates, train_targets_sum, label="Train Truth", color="#333333", linewidth=2, alpha=0.8)
    ax_agg.plot(train_dates, train_preds_sum, label="Train Pred", color="#348ABD", linewidth=2, linestyle="--", alpha=0.8)
    ax_agg.axvline(x=split_date, color="red", linestyle=":", linewidth=2, alpha=0.7, label="Train/Test Split")
    ax_agg.plot(test_dates, test_targets_sum, label="Test Truth", color="#E24A33", marker="o", markersize=3, linestyle="None", alpha=0.8)
    ax_agg.plot(test_dates, test_preds_sum, label="Test Pred", color="#E24A33", linewidth=2, linestyle=":", alpha=0.8)
    ax_agg.set_xlabel("Date")
    ax_agg.set_ylabel("Total Hospitalizations")
    ax_agg.set_title("Aggregate (National) Forecast")
    ax_agg.legend(frameon=True, framealpha=0.9, loc="best", fontsize=9)
    ax_agg.grid(True, linestyle=":", alpha=0.6)
    plt.setp(ax_agg.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # CBSA-specific plots
    axes_cbsa = [fig.add_subplot(gs[row, 2]) for row in range(3)]
    for ax, cbsa_code in zip(axes_cbsa, FOCUS_CBSAS):
        if cbsa_code not in cbsa_to_idx:
            ax.set_visible(False)
            continue
        idx = cbsa_to_idx[cbsa_code]
        name = TITLE_CBSA_MAP.get(cbsa_code, cbsa_code)

        ax.plot(train_dates, train_targets_orig[:, idx], label="Train Truth", color="#333333", linewidth=1.5, alpha=0.8)
        ax.plot(train_dates, train_preds_orig[:, idx], label="Train Pred", color="#348ABD", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(x=split_date, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.plot(test_dates, test_targets_orig[:, idx], label="Test Truth", color="#E24A33", marker="o", markersize=3, linestyle="None", alpha=0.8)
        ax.plot(test_dates, test_preds_orig[:, idx], label="Test Pred", color="#E24A33", linewidth=1.5, linestyle=":", alpha=0.8)

        ax.set_title(name)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax is axes_cbsa[-1]:
            ax.set_xlabel("Date")
        else:
            ax.set_xticklabels([])

        if ax is axes_cbsa[0]:
            ax.legend(frameon=True, framealpha=0.9, fontsize=8, loc="best")

    fig.suptitle(version, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_explain_page(
    num_explainable_list,
    train_targets_orig,
    loader,
    version,
    pdf_path,
):
    """Create a dual-axis PDF: explainable edge count vs aggregate target."""
    import matplotlib.pyplot as plt

    dates = pd.to_datetime([d.strftime("%Y-%m-%d") for d in loader.dates])
    train_dates = dates[:len(train_targets_orig)]
    train_targets_sum = train_targets_orig.sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_target = "#348ABD"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Aggregate Hospitalizations / Cases", color=color_target)
    ax1.plot(
        train_dates, train_targets_sum,
        color=color_target, linewidth=2, alpha=0.8, label="Aggregate Target",
    )
    ax1.tick_params(axis="y", labelcolor=color_target)
    ax1.grid(True, linestyle=":", alpha=0.4)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    color_edges = "#E24A33"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Explainable Edges", color=color_edges)
    ax2.plot(
        train_dates, num_explainable_list,
        color=color_edges, linewidth=2, alpha=0.8, label="Explainable Edges",
    )
    ax2.tick_params(axis="y", labelcolor=color_edges)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", frameon=True, framealpha=0.9,
    )

    fig.suptitle(f"Explainable Edges vs Aggregate Target\n{version}", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _fit_scaler(arrays, per_node_only=False):
    """Compute per-element mean/std over the time axis.

    Args:
        arrays: list of numpy arrays, each [num_nodes, ...].
        per_node_only: if True and individual arrays are 2-D, collapse the
            trailing dimension so the result is [num_nodes] (useful for
            multi-horizon targets so all horizons share one scale per node).
    """
    stacked = np.stack(arrays, axis=0)
    if per_node_only and stacked.ndim == 3:
        mean = stacked.mean(axis=(0, 2))
        std = stacked.std(axis=(0, 2))
    else:
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _scale(arr, mean, std):
    """Standardize *arr* using pre-computed *mean* / *std*."""
    if arr.ndim > mean.ndim:
        return (arr - mean[:, np.newaxis]) / std[:, np.newaxis]
    return (arr - mean) / std


def add_self_loops(dataset):
    """Add self-loops to ensure no zero in-degree nodes."""
    for i in range(len(dataset.edge_indices)):
        edge_index = dataset.edge_indices[i]
        edge_weight = dataset.edge_weights[i]
        num_nodes = dataset.features[i].shape[0]

        in_degree = np.zeros(num_nodes)
        np.add.at(in_degree, edge_index[1], 1)
        zero_in_nodes = np.where(in_degree == 0)[0]

        if len(zero_in_nodes) > 0:
            self_loops = np.stack([zero_in_nodes, zero_in_nodes], axis=0)
            self_loop_weights = np.ones(len(zero_in_nodes), dtype=np.float32)

            dataset.edge_indices[i] = np.concatenate([edge_index, self_loops], axis=1)
            dataset.edge_weights[i] = np.concatenate([edge_weight, self_loop_weights])

    return dataset


def get_last_available_mobility_date(forecast_date, available_dates):
    """For a forecast Monday in month M, return the latest mobility date in month
    M-1, simulating Advan's monthly refresh (M-1 data published on first Monday of M).

    Args:
        forecast_date: pd.Timestamp or date-like for the forecast week.
        available_dates: sorted pd.DatetimeIndex of all dates in mobility CSV.

    Returns:
        pd.Timestamp of the last available mobility date.
    """
    forecast_date = pd.to_datetime(forecast_date)
    prev_month_end = forecast_date.replace(day=1) - pd.Timedelta(days=1)
    eligible = available_dates[available_dates <= prev_month_end]
    if len(eligible) == 0:
        # No prior-month data available (e.g. first month of dataset);
        # fall back to the earliest available mobility date.
        return available_dates[0]
    return eligible[-1]


def apply_monthly_lag_to_dataset(dataset, loader):
    """Replace every snapshot's edges with the latest mobility available under
    Advan's monthly refresh schedule (M-1 data available on first Monday of M).

    Unlike apply_constant_mobility_to_test(), this applies to ALL snapshots
    (train and test), enabling full time-series explainer analysis with a
    single pre-trained checkpoint.

    Must be called BEFORE add_self_loops().

    Args:
        dataset: DynamicGraphTemporalSignal (full, unsplit).
        loader: ConfigurableDatasetLoader — provides mobility_df, dates,
                and _get_edges_for_date().

    Returns:
        dataset with all edge_indices and edge_weights replaced.
    """
    available_dates = pd.DatetimeIndex(
        loader.mobility_df["date_range_start"].unique()
    ).sort_values()
    for i, forecast_date in enumerate(loader.dates):
        lag_date = get_last_available_mobility_date(forecast_date, available_dates)
        edge_index, edge_weight = loader._get_edges_for_date(lag_date)
        dataset.edge_indices[i] = edge_index.copy()
        dataset.edge_weights[i] = edge_weight.copy()
        print(
            f"  snapshot {i} ({pd.Timestamp(forecast_date).date()}) "
            f"→ mobility {lag_date.date()}"
        )
    return dataset


def apply_constant_mobility_to_test(dataset, train_size, loader):
    """Replace test-period edges with the last available training-week mobility.

    Simulates the real-time setting where the most recent Advan data has not
    yet been published. All test snapshots receive the same edges: those from
    the final training snapshot date.

    Must be called BEFORE add_self_loops() so self-loops are correctly
    re-applied to the substituted edge sets.

    Args:
        dataset: DynamicGraphTemporalSignal (full, unsplit).
        train_size: number of training snapshots (test starts at this index).
        loader: ConfigurableDatasetLoader — provides _get_edges_for_date()
                and loader.dates.

    Returns:
        dataset with test-period edge_indices and edge_weights replaced.
    """
    last_train_date = pd.to_datetime(loader.dates[train_size - 1])
    edge_index_const, edge_weight_const = loader._get_edges_for_date(last_train_date)

    for i in range(train_size, len(dataset.edge_indices)):
        dataset.edge_indices[i] = edge_index_const.copy()
        dataset.edge_weights[i] = edge_weight_const.copy()

    return dataset


def get_loss_fn(objective_name: str):
    """Return loss function for the given objective name."""
    name = str(objective_name).upper()
    if name == "RMSE":
        return lambda pred, target: torch.sqrt(F.mse_loss(pred, target))
    if name == "RMSLE":
        def rmsle_loss(pred, target):
            pred = pred.clamp(min=0.0)
            target = target.clamp(min=0.0)
            return torch.sqrt(F.mse_loss(torch.log1p(pred), torch.log1p(target)))
        return rmsle_loss
    if name == "MAE":
        return lambda pred, target: F.l1_loss(pred, target)
    raise ValueError(f"Unknown objective: {objective_name}")


def reverse_transform_predictions(preds, targets, loader, initial_values=None):
    """Reverse transform predictions and targets to original space."""
    preds = np.array(preds)
    targets = np.array(targets)

    if preds.ndim == 3:
        preds = preds[:, :, 0]
        targets = targets[:, :, 0]

    y_transform = loader.y_transform

    if y_transform == "log1p":
        preds = np.expm1(preds)
        targets = np.expm1(targets)
        return preds, targets

    if y_transform in ("diff", "log_diff"):
        if initial_values is None:
            first_date = loader.dates[0]
            first_date_str = pd.to_datetime(first_date).strftime("%Y-%m-%d")
            first_df = loader.data_df[
                loader.data_df[loader.date_col].dt.strftime("%Y-%m-%d") == first_date_str
            ]
            initial_dict = dict(zip(first_df["CBSA"], first_df[loader.target_col]))
            initial_values = np.array(
                [initial_dict.get(cbsa, 0.0) for cbsa in loader.cbsa_list],
                dtype=np.float32,
            )

        if y_transform == "log_diff":
            log_initial = np.log1p(np.abs(initial_values))
            log_preds_cumsum = np.cumsum(preds, axis=0)
            log_targets_cumsum = np.cumsum(targets, axis=0)
            preds = np.expm1(log_initial[None, :] + log_preds_cumsum)
            targets = np.expm1(log_initial[None, :] + log_targets_cumsum)
        else:
            preds = initial_values[None, :] + np.cumsum(preds, axis=0)
            targets = initial_values[None, :] + np.cumsum(targets, axis=0)

    return preds, targets


@hydra.main(version_base=None, config_path="experiments/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Apply log-log transform defaults ---
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg
    if not OmegaConf.is_missing(data_cfg, "x_transform"):
        x_transform = data_cfg.get("x_transform", "log1p")
    else:
        x_transform = "log1p"
    if not OmegaConf.is_missing(data_cfg, "y_transform"):
        y_transform = data_cfg.get("y_transform", "log1p")
    else:
        y_transform = "log1p"
    OmegaConf.update(data_cfg, "x_transform", x_transform, force_add=True)
    OmegaConf.update(data_cfg, "y_transform", y_transform, force_add=True)

    # Read sweep parameters from Hydra config
    target_horizon = data_cfg.get("target_horizon", 1)
    lr = cfg.training.get("learning_rate", 1e-5)
    if hasattr(cfg, "model") and hasattr(cfg.model, "cdcrnn"):
        hidden_size = cfg.model.cdcrnn.get("hidden_size", 32)
        K = cfg.model.cdcrnn.get("K", 2)
    else:
        hidden_size = 32
        K = 2

    num_epochs = cfg.training.get("num_epochs", 1000)
    version = cfg.version

    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    print("CDCRNN Pipeline")
    print("=" * 70)
    print(f"Version: {version}")
    print(f"Device: {device}")
    print(f"Data source: {data_cfg.data_source}")
    print(f"Mobility source: {data_cfg.mobility_source}")
    print(f"x_transform: {x_transform}, y_transform: {y_transform}")
    print(f"Horizon: {target_horizon}, LR: {lr}, Hidden size: {hidden_size}")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n[1/4] Loading data...")
    loader = ConfigurableDatasetLoader(
        cbsa_list=None,
        cfg=cfg,
    )
    dataset = loader.get_dataset()

    # --- Optional: replace test-period mobility with last training-week edges ---
    if bool(data_cfg.get("use_lagged_mobility", False)):
        _train_size = int(float(data_cfg.get("train_ratio", 0.8)) * dataset.snapshot_count)
        n_test = dataset.snapshot_count - _train_size
        print(
            f"  use_lagged_mobility=True: substituting last training-week edges "
            f"for {n_test} test snapshots..."
        )
        dataset = apply_constant_mobility_to_test(dataset, _train_size, loader)

    if bool(data_cfg.get("use_monthly_lag_mobility", False)):
        print(
            "  use_monthly_lag_mobility=True: applying variable monthly-refresh "
            "lag to all snapshots..."
        )
        dataset = apply_monthly_lag_to_dataset(dataset, loader)

    dataset = add_self_loops(dataset)

    # --- Optional: expanding-window CV (used for flu seasons) ---
    cv_cfg = data_cfg.get("cv", None)
    print(f"cv_cfg: {cv_cfg}")
    cv_enabled = bool(cv_cfg.get("enabled", False)) if isinstance(cv_cfg, (dict, DictConfig)) else False
    if cv_enabled:
        skip_training = bool(cfg.get("skip_training", False))
        if skip_training:
            raise ValueError(
                "skip_training=true is not supported with data.cv.enabled=true yet. "
                "Please disable skip_training for CV runs."
            )

        initial_train_weeks = int(cv_cfg.get("initial_train_weeks", 10))
        test_weeks = int(cv_cfg.get("test_weeks", 2))
        step_weeks = int(cv_cfg.get("step_weeks", test_weeks))

        folds = loader.get_expanding_window_cv_splits(
            dataset=dataset,
            initial_train_weeks=initial_train_weeks,
            test_weeks=test_weeks,
            step_weeks=step_weeks,
        )
        if not folds:
            raise ValueError(
                "No expanding-window CV folds were generated. "
                "Check cv.initial_train_weeks/test_weeks/step_weeks and the season date range."
            )

        print(
            f"\n[CV] Expanding-window training enabled: folds={len(folds)} "
            f"(initial_train_weeks={initial_train_weeks}, test_weeks={test_weeks}, step_weeks={step_weeks})"
        )

        # Whether to fit scalers is determined by x_transform / y_transform.
        # (Exactly matches the non-CV code’s logic.)
        use_x_scaler = loader.x_transform in _SCALER_TRANSFORMS
        use_y_scaler = loader.y_transform in _SCALER_TRANSFORMS

        loss_fn = get_loss_fn(cfg.training.get("loss_function", "RMSE"))
        global_best_test_loss = float("inf")
        global_best_state = None
        global_best_fold_idx = None
        global_best_epoch = None
        global_best_model = None

        fold_results = []

        for fold in folds:
            fold_idx = fold["fold_idx"]
            train_dataset = fold["train_dataset"]
            test_dataset = fold["test_dataset"]

            # Per-fold scaling to avoid leakage (fit on fold train only).
            x_mean = x_std = y_mean = y_std = None
            if use_x_scaler:
                x_mean, x_std = _fit_scaler(list(train_dataset.features))
                train_dataset.features = [
                    _scale(f, x_mean, x_std) for f in train_dataset.features
                ]
                test_dataset.features = [
                    _scale(f, x_mean, x_std) for f in test_dataset.features
                ]

            if use_y_scaler:
                y_mean, y_std = _fit_scaler(list(train_dataset.targets), per_node_only=True)
                train_dataset.targets = [_scale(t, y_mean, y_std) for t in train_dataset.targets]
                test_dataset.targets = [_scale(t, y_mean, y_std) for t in test_dataset.targets]

            node_features = train_dataset.features[0].shape[1]
            predict_delta = loader.predict_delta or use_y_scaler

            model = CDCRNN(
                node_features=node_features,
                target_horizon=loader.target_horizon,
                predict_delta=predict_delta,
                hidden_size=hidden_size,
                K=K,
            ).to(device)

            print(
                f"\n[CV] Fold {fold_idx}: train_snapshots={train_dataset.snapshot_count}, "
                f"test_snapshots={test_dataset.snapshot_count}"
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=cfg.training.get("weight_decay", 5e-4),
            )

            best_test_loss_fold = float("inf")
            best_state_fold = None
            best_epoch_fold = None

            def _eval_test(h_init=None):
                model.eval()
                test_loss_val = 0.0
                test_batches = 0
                with torch.no_grad():
                    h_test = h_init
                    for snapshot in test_dataset:
                        snapshot = snapshot.to(device)
                        pred, h_test = model(
                            snapshot.x,
                            snapshot.edge_index,
                            snapshot.edge_attr,
                            h_test,
                        )
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(1)
                        if snapshot.y.dim() == 1:
                            target = snapshot.y.unsqueeze(1)
                        else:
                            target = snapshot.y
                        test_loss_val += loss_fn(pred, target).item()
                        test_batches += 1
                return test_loss_val / max(test_batches, 1)

            for epoch in tqdm(
                range(num_epochs),
                desc=f"Training CV fold {fold_idx}",
                miniters=1000,
            ):
                model.train()
                epoch_loss = 0.0
                h = None
                num_batches = 0

                for snapshot in train_dataset:
                    snapshot = snapshot.to(device)
                    pred, h = model(
                        snapshot.x,
                        snapshot.edge_index,
                        snapshot.edge_attr,
                        h,
                    )
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(1)
                    if snapshot.y.dim() == 1:
                        target = snapshot.y.unsqueeze(1)
                    else:
                        target = snapshot.y

                    loss = loss_fn(pred, target)
                    epoch_loss += loss.item()
                    num_batches += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    h = h.detach()

                if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
                    test_loss_val = _eval_test(h_init=h.detach())
                    if test_loss_val < best_test_loss_fold:
                        best_test_loss_fold = test_loss_val
                        best_epoch_fold = epoch + 1
                        best_state_fold = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # --- Per-fold evaluation in original space ---
            if best_state_fold is not None:
                model.load_state_dict(best_state_fold)
            model = model.to(device)
            model.eval()

            train_preds_fold, train_targets_fold = [], []
            h_eval = None
            with torch.no_grad():
                for snapshot in train_dataset:
                    snapshot = snapshot.to(device)
                    pred, h_eval = model(
                        snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_eval,
                    )
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(1)
                    if snapshot.y.dim() == 1:
                        tgt = snapshot.y.unsqueeze(1)
                    else:
                        tgt = snapshot.y
                    train_preds_fold.append(pred[:, 0].cpu().numpy())
                    train_targets_fold.append(tgt[:, 0].cpu().numpy())
                    h_eval = h_eval.detach()

            test_preds_fold, test_targets_fold = [], []
            with torch.no_grad():
                for snapshot in test_dataset:
                    snapshot = snapshot.to(device)
                    pred, h_eval = model(
                        snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_eval,
                    )
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(1)
                    if snapshot.y.dim() == 1:
                        tgt = snapshot.y.unsqueeze(1)
                    else:
                        tgt = snapshot.y
                    test_preds_fold.append(pred[:, 0].cpu().numpy())
                    test_targets_fold.append(tgt[:, 0].cpu().numpy())
                    h_eval = h_eval.detach()

            train_preds_arr = np.vstack(train_preds_fold)
            train_targets_arr = np.vstack(train_targets_fold)
            test_preds_arr = np.vstack(test_preds_fold)
            test_targets_arr = np.vstack(test_targets_fold)

            if y_mean is not None:
                train_preds_arr = train_preds_arr * y_std + y_mean
                train_targets_arr = train_targets_arr * y_std + y_mean
                test_preds_arr = test_preds_arr * y_std + y_mean
                test_targets_arr = test_targets_arr * y_std + y_mean

            train_preds_orig, train_targets_orig = reverse_transform_predictions(
                train_preds_arr, train_targets_arr, loader
            )

            test_initial = None
            if loader.y_transform in ("diff", "log_diff"):
                train_size_snap = train_preds_arr.shape[0]
                split_date = loader.dates[train_size_snap]
                split_date_str = pd.to_datetime(split_date).strftime("%Y-%m-%d")
                split_df = loader.data_df[
                    loader.data_df[loader.date_col].dt.strftime("%Y-%m-%d") == split_date_str
                ]
                split_dict = dict(zip(split_df["CBSA"], split_df[loader.target_col]))
                test_initial = np.array(
                    [split_dict.get(cbsa, 0.0) for cbsa in loader.cbsa_list],
                    dtype=np.float32,
                )

            test_preds_orig, test_targets_orig = reverse_transform_predictions(
                test_preds_arr, test_targets_arr, loader, initial_values=test_initial
            )

            fold_train_rmse = float(np.sqrt(np.mean((train_preds_orig - train_targets_orig) ** 2)))
            fold_test_rmse = float(np.sqrt(np.mean((test_preds_orig - test_targets_orig) ** 2)))
            fold_train_mae = float(np.mean(np.abs(train_preds_orig - train_targets_orig)))
            fold_test_mae = float(np.mean(np.abs(test_preds_orig - test_targets_orig)))

            dates_all = pd.to_datetime([d.strftime("%Y-%m-%d") for d in loader.dates])
            fold_train_date_start = str(dates_all[fold["train_start"]])
            fold_train_date_end = str(dates_all[fold["train_end"] - 1])
            fold_test_date_start = str(dates_all[fold["test_start"]])
            fold_test_date_end = str(dates_all[fold["test_end"] - 1])

            print(
                f"  Fold {fold_idx} eval: "
                f"train_rmse={fold_train_rmse:.4f} test_rmse={fold_test_rmse:.4f} "
                f"train_mae={fold_train_mae:.4f} test_mae={fold_test_mae:.4f}"
            )

            fold_results.append(
                {
                    "fold_idx": fold_idx,
                    "best_test_loss": best_test_loss_fold,
                    "best_epoch": best_epoch_fold,
                    "train_rmse": fold_train_rmse,
                    "test_rmse": fold_test_rmse,
                    "train_mae": fold_train_mae,
                    "test_mae": fold_test_mae,
                    "train_snapshots": train_dataset.snapshot_count,
                    "test_snapshots": test_dataset.snapshot_count,
                    "train_date_start": fold_train_date_start,
                    "train_date_end": fold_train_date_end,
                    "test_date_start": fold_test_date_start,
                    "test_date_end": fold_test_date_end,
                }
            )

            if best_state_fold is not None and best_test_loss_fold < global_best_test_loss:
                global_best_test_loss = best_test_loss_fold
                global_best_fold_idx = fold_idx
                global_best_epoch = best_epoch_fold
                global_best_state = best_state_fold
                global_best_model = model  # keep instance for checkpoint save

        # Aggregate fold metrics
        best_losses = [r["best_test_loss"] for r in fold_results]
        test_rmses = [r["test_rmse"] for r in fold_results]
        test_maes = [r["test_mae"] for r in fold_results]
        train_rmses = [r["train_rmse"] for r in fold_results]
        train_maes = [r["train_mae"] for r in fold_results]

        cv_loss_mean = float(np.mean(best_losses))
        cv_loss_std = float(np.std(best_losses))
        cv_test_rmse_mean = float(np.mean(test_rmses))
        cv_test_rmse_std = float(np.std(test_rmses))
        cv_test_mae_mean = float(np.mean(test_maes))
        cv_test_mae_std = float(np.std(test_maes))

        print(
            f"\n[CV] Done. best_fold={global_best_fold_idx} "
            f"best_test_loss={global_best_test_loss:.6f}\n"
            f"  mean(test_loss)={cv_loss_mean:.6f}±{cv_loss_std:.6f}  "
            f"mean(test_rmse)={cv_test_rmse_mean:.4f}±{cv_test_rmse_std:.4f}  "
            f"mean(test_mae)={cv_test_mae_mean:.4f}±{cv_test_mae_std:.4f}"
        )

        # Save the best fold checkpoint for later evaluation/explanation.
        checkpoint_version = cfg.get("checkpoint_version", version)
        best_checkpoint_version = f"{checkpoint_version}_fold{global_best_fold_idx}"
        model_save_dir = os.path.join(
            REPO_DIR, "models", "gcn_checkpoints", best_checkpoint_version
        )
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, "cdcrnn_final_model.pt")

        if global_best_model is None or global_best_state is None:
            raise RuntimeError("CV selected no best fold model checkpoint.")

        # Use the model instance from the best fold and load the best state dict.
        best_model_to_save = global_best_model
        best_model_to_save.load_state_dict(global_best_state)
        best_model_to_save = best_model_to_save.to(device)

        torch.save(
            {
                "epoch": global_best_epoch,
                "model_state_dict": best_model_to_save.state_dict(),
                "best_test_loss": global_best_test_loss,
                "hidden_size": hidden_size,
                "learning_rate": lr,
                "target_horizon": target_horizon,
            },
            save_path,
        )
        save_config_to_directory(cfg, model_save_dir)

        # Persist per-fold detail + averaged summary rows.
        plot_dir = os.path.join(REPO_DIR, "plots", version)
        os.makedirs(plot_dir, exist_ok=True)

        # Per-fold detail CSV (unchanged from before).
        pd.DataFrame(fold_results).to_csv(
            os.path.join(plot_dir, "cv_fold_results.csv"), index=False
        )

        # eval_metrics.csv: one row per fold + a "mean" summary row.
        eval_rows = []
        for r in fold_results:
            eval_rows.append(
                {
                    "split": f"fold{r['fold_idx']}",
                    "train_rmse": r["train_rmse"],
                    "test_rmse": r["test_rmse"],
                    "train_mae": r["train_mae"],
                    "test_mae": r["test_mae"],
                    "best_test_loss": r["best_test_loss"],
                    "best_epoch": r["best_epoch"],
                    "learning_rate": lr,
                    "hidden_size": hidden_size,
                    "target_horizon": target_horizon,
                    "train_snapshots": r["train_snapshots"],
                    "test_snapshots": r["test_snapshots"],
                    "train_date_start": r["train_date_start"],
                    "train_date_end": r["train_date_end"],
                    "test_date_start": r["test_date_start"],
                    "test_date_end": r["test_date_end"],
                }
            )
        eval_rows.append(
            {
                "split": "cv_mean",
                "train_rmse": float(np.mean(train_rmses)),
                "test_rmse": cv_test_rmse_mean,
                "train_mae": float(np.mean(train_maes)),
                "test_mae": cv_test_mae_mean,
                "best_test_loss": cv_loss_mean,
                "best_epoch": None,
                "learning_rate": lr,
                "hidden_size": hidden_size,
                "target_horizon": target_horizon,
                "train_snapshots": None,
                "test_snapshots": None,
                "train_date_start": None,
                "train_date_end": None,
                "test_date_start": None,
                "test_date_end": None,
            }
        )
        eval_rows.append(
            {
                "split": "cv_std",
                "train_rmse": float(np.std(train_rmses)),
                "test_rmse": cv_test_rmse_std,
                "train_mae": float(np.std(train_maes)),
                "test_mae": cv_test_mae_std,
                "best_test_loss": cv_loss_std,
                "best_epoch": None,
                "learning_rate": lr,
                "hidden_size": hidden_size,
                "target_horizon": target_horizon,
                "train_snapshots": None,
                "test_snapshots": None,
                "train_date_start": None,
                "train_date_end": None,
                "test_date_start": None,
                "test_date_end": None,
            }
        )
        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(os.path.join(plot_dir, "eval_metrics.csv"), index=False)

        print(f"  Saved best fold checkpoint to {model_save_dir}")
        print(f"  Saved eval_metrics.csv ({len(eval_rows)} rows) to {plot_dir}")
        return

    train_ratio = data_cfg.get("train_ratio", 0.8)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)

    # --- Per-node StandardScaler (skip for self-normalizing log_diff) ---
    use_x_scaler = loader.x_transform in _SCALER_TRANSFORMS
    use_y_scaler = loader.y_transform in _SCALER_TRANSFORMS
    x_mean = x_std = y_mean = y_std = None
    if use_x_scaler:
        x_mean, x_std = _fit_scaler(list(train_dataset.features))
        train_dataset.features = [_scale(f, x_mean, x_std) for f in train_dataset.features]
        test_dataset.features = [_scale(f, x_mean, x_std) for f in test_dataset.features]
        print(f"  Applied x StandardScaler (x_transform={loader.x_transform})")
    if use_y_scaler:
        y_mean, y_std = _fit_scaler(list(train_dataset.targets), per_node_only=True)
        train_dataset.targets = [_scale(t, y_mean, y_std) for t in train_dataset.targets]
        test_dataset.targets = [_scale(t, y_mean, y_std) for t in test_dataset.targets]
        print(f"  Applied y StandardScaler (y_transform={loader.y_transform})")

    node_features = train_dataset.features[0].shape[1]
    # When targets are standardized the model must be free to output negative
    # values, so force predict_delta=True to disable the final output ReLU.
    predict_delta = loader.predict_delta or use_y_scaler
    print(f"  Node features: {node_features}")
    print(f"  Train snapshots: {train_dataset.snapshot_count}")
    print(f"  Test snapshots: {test_dataset.snapshot_count}")
    print(f"  predict_delta: {predict_delta}")

    # --- 2. Train (or load) CDCRNN ---
    model = CDCRNN(
        node_features=node_features,
        target_horizon=loader.target_horizon,
        predict_delta=predict_delta,
        hidden_size=hidden_size,
        K=K,
    ).to(device)

    loss_fn = get_loss_fn(cfg.training.get("loss_function", "RMSE"))

    checkpoint_version = cfg.get("checkpoint_version", version)
    model_save_dir = os.path.join(
        REPO_DIR, "models", "gcn_checkpoints", checkpoint_version
    )
    checkpoint_path = os.path.join(model_save_dir, "cdcrnn_final_model.pt")
    skip_training = cfg.get("skip_training", False)

    if skip_training and os.path.exists(checkpoint_path):
        print(f"\n[2/4] Loading CDCRNN from checkpoint (skip_training=True)...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_test_loss = ckpt.get("best_test_loss", float("inf"))
        train_losses = []
        test_losses = []
        print(f"  Loaded from {checkpoint_path}")
        print(f"  best_test_loss={best_test_loss:.4f}")
    else:
        print("\n[2/4] Training CDCRNN...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.training.get("weight_decay", 5e-4),
        )

        train_losses = []
        test_losses = []
        best_test_loss = float("inf")
        best_state = None

        def _eval_test(h_init=None):
            model.eval()
            test_loss_val = 0.0
            test_batches = 0
            with torch.no_grad():
                h_test = h_init
                for snapshot in test_dataset:
                    snapshot = snapshot.to(device)
                    pred, h_test = model(
                        snapshot.x,
                        snapshot.edge_index,
                        snapshot.edge_attr,
                        h_test,
                    )
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(1)
                    if snapshot.y.dim() == 1:
                        target = snapshot.y.unsqueeze(1)
                    else:
                        target = snapshot.y
                    test_loss_val += loss_fn(pred, target).item()
                    test_batches += 1
            return test_loss_val / max(test_batches, 1)

        for epoch in tqdm(range(num_epochs), desc="Training", miniters=1000):
            model.train()
            epoch_loss = 0.0
            h = None
            num_batches = 0
            for snapshot in train_dataset:
                snapshot = snapshot.to(device)
                pred, h = model(
                    snapshot.x,
                    snapshot.edge_index,
                    snapshot.edge_attr,
                    h,
                )
                if pred.dim() == 1:
                    pred = pred.unsqueeze(1)
                if snapshot.y.dim() == 1:
                    target = snapshot.y.unsqueeze(1)
                else:
                    target = snapshot.y
                loss = loss_fn(pred, target)
                epoch_loss += loss.item()
                num_batches += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                h = h.detach()

            train_losses.append(epoch_loss / num_batches)

            if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
                test_loss_val = _eval_test(h_init=h.detach())
                test_losses.append((epoch + 1, test_loss_val))
                if test_loss_val < best_test_loss:
                    best_test_loss = test_loss_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if (epoch + 1) % 500 == 0 or epoch == num_epochs - 1:
                    print(
                        f"  Epoch {epoch+1}: train_loss={train_losses[-1]:.4f} "
                        f"test_loss={test_loss_val:.4f}"
                    )

        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.to(device)

        save_dir = os.path.join(REPO_DIR, "models", "gcn_checkpoints", version)
        save_path = os.path.join(save_dir, "cdcrnn_final_model.pt")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "epoch": num_epochs,
                "model_state_dict": model.state_dict(),
                "best_test_loss": best_test_loss,
                "hidden_size": hidden_size,
                "learning_rate": lr,
                "target_horizon": target_horizon,
            },
            save_path,
        )
        save_config_to_directory(cfg, save_dir)
        print(f"  Model saved to {save_dir}")

    # --- 3. Evaluate ---
    print("\n[3/4] Evaluating...")
    model.eval()

    # Collect predictions in transformed space
    train_preds, train_targets_list = [], []
    h = None
    with torch.no_grad():
        for snapshot in train_dataset:
            snapshot = snapshot.to(device)
            pred, h = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            if snapshot.y.dim() == 1:
                tgt = snapshot.y.unsqueeze(1)
            else:
                tgt = snapshot.y
            train_preds.append(pred[:, 0].cpu().numpy())
            train_targets_list.append(tgt[:, 0].cpu().numpy())
            h = h.detach()

    test_preds, test_targets_list = [], []
    with torch.no_grad():
        for snapshot in test_dataset:
            snapshot = snapshot.to(device)
            pred, h = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            if snapshot.y.dim() == 1:
                tgt = snapshot.y.unsqueeze(1)
            else:
                tgt = snapshot.y
            test_preds.append(pred[:, 0].cpu().numpy())
            test_targets_list.append(tgt[:, 0].cpu().numpy())
            h = h.detach()

    train_preds = np.vstack(train_preds)
    train_targets_arr = np.vstack(train_targets_list)
    test_preds = np.vstack(test_preds)
    test_targets_arr = np.vstack(test_targets_list)

    # Undo StandardScaler
    if y_mean is not None:
        train_preds = train_preds * y_std + y_mean
        train_targets_arr = train_targets_arr * y_std + y_mean
        test_preds = test_preds * y_std + y_mean
        test_targets_arr = test_targets_arr * y_std + y_mean

    # Reverse transform to original space
    train_preds_orig, train_targets_orig = reverse_transform_predictions(
        train_preds, train_targets_arr, loader
    )

    test_initial = None
    if loader.y_transform in ("diff", "log_diff"):
        train_size_snap = train_preds.shape[0]
        split_date = loader.dates[train_size_snap]
        split_date_str = pd.to_datetime(split_date).strftime("%Y-%m-%d")
        split_df = loader.data_df[
            loader.data_df[loader.date_col].dt.strftime("%Y-%m-%d") == split_date_str
        ]
        split_dict = dict(zip(split_df["CBSA"], split_df[loader.target_col]))
        test_initial = np.array(
            [split_dict.get(cbsa, 0.0) for cbsa in loader.cbsa_list],
            dtype=np.float32,
        )

    test_preds_orig, test_targets_orig = reverse_transform_predictions(
        test_preds, test_targets_arr, loader, initial_values=test_initial
    )

    # Metrics in original space
    train_rmse = np.sqrt(np.mean((train_preds_orig - train_targets_orig) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds_orig - test_targets_orig) ** 2))
    train_mae = np.mean(np.abs(train_preds_orig - train_targets_orig))
    test_mae = np.mean(np.abs(test_preds_orig - test_targets_orig))

    print(f"  Train RMSE: {train_rmse:.4f}  Train MAE: {train_mae:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}  Test MAE:  {test_mae:.4f}")

    plot_dir = os.path.join(REPO_DIR, "plots", version)
    os.makedirs(plot_dir, exist_ok=True)

    eval_df = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rmse": [train_rmse, test_rmse],
            "mae": [train_mae, test_mae],
            "best_test_loss": [None, best_test_loss],
            "learning_rate": [lr, lr],
            "hidden_size": [hidden_size, hidden_size],
            "target_horizon": [target_horizon, target_horizon],
        }
    )
    eval_df.to_csv(os.path.join(plot_dir, "eval_metrics.csv"), index=False)

    # --- 4. DCRNNExplainer ---
    print("\n[4/4] Running DCRNNExplainer...")
    explain_epochs = cfg.explain.get("epochs", 200)
    explain_lr = cfg.explain.get("lr", 0.01)
    edge_threshold = cfg.explain.get("edge_mask_threshold", 0.5)
    explainer = DCRNNExplainer(model, device)

    dates = pd.to_datetime([d.strftime("%Y-%m-%d") for d in loader.dates])
    cbsa_list = loader.cbsa_list

    def _explain_dataset(dataset, split_label, date_offset):
        """Run DCRNNExplainer on every snapshot in *dataset*.

        Args:
            dataset: temporal signal (train or test split).
            split_label: "train" or "test" (stored in outputs).
            date_offset: index into *dates* for the first snapshot.

        Returns:
            (num_explainable_list, explain_edge_rows, mask_records,
             explain_loss_histories)
        """
        num_explainable = []
        edge_rows = []
        records = []
        all_loss_histories = []
        for i, snapshot in enumerate(tqdm(dataset, desc=f"Explain ({split_label})", miniters=10)):
            snapshot = snapshot.to(device)
            importance_mask, loss_history = explainer.explain(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                epochs=explain_epochs,
                lr=explain_lr,
                verbose=False,
                track_loss=True,
            )
            all_loss_histories.append(loss_history)
            important_edges = (importance_mask > edge_threshold)
            num_explainable.append(important_edges.sum().item())

            edge_index_np = snapshot.edge_index.cpu().numpy()
            mask_np = importance_mask.cpu().numpy()
            snapshot_date = dates[date_offset + i]

            for j in range(edge_index_np.shape[1]):
                src, dst = edge_index_np[0, j], edge_index_np[1, j]
                records.append({
                    "date": snapshot_date,
                    "split": split_label,
                    "snapshot_idx": date_offset + i,
                    "edge_j": j,
                    "src_idx": src,
                    "dst_idx": dst,
                    "cbsa_orig": cbsa_list[src],
                    "cbsa_dest": cbsa_list[dst],
                    "importance": mask_np[j],
                })

            edge_idx = edge_index_np[:, important_edges.cpu().numpy()]
            for j in range(edge_idx.shape[1]):
                edge_rows.append({
                    "date": snapshot_date,
                    "split": split_label,
                    "cbsa_orig": cbsa_list[edge_idx[0, j]],
                    "cbsa_dest": cbsa_list[edge_idx[1, j]],
                })

        return num_explainable, edge_rows, records, all_loss_histories

    train_size = train_dataset.snapshot_count
    train_expl_counts, train_edge_rows, train_mask_records, train_explain_losses = (
        _explain_dataset(train_dataset, "train", date_offset=0)
    )
    test_expl_counts, test_edge_rows, test_mask_records, test_explain_losses = (
        _explain_dataset(test_dataset, "test", date_offset=train_size)
    )

    # --- Plot loss curves + predictions (after explainer so we have its losses) ---
    all_explain_losses = train_explain_losses + test_explain_losses
    pdf_path = os.path.join(plot_dir, f"{version}.pdf")
    print(f"  Saving plots to {pdf_path}")
    plot_experiment_page(
        train_losses=train_losses,
        test_losses=test_losses,
        train_preds_orig=train_preds_orig,
        train_targets_orig=train_targets_orig,
        test_preds_orig=test_preds_orig,
        test_targets_orig=test_targets_orig,
        loader=loader,
        version=version,
        pdf_path=pdf_path,
        explain_losses=all_explain_losses,
    )

    all_mask_records = train_mask_records + test_mask_records
    masks_df = pd.DataFrame(all_mask_records)
    masks_path = os.path.join(plot_dir, "importance_masks.parquet")
    masks_df.to_parquet(masks_path, index=False)
    print(f"  Full importance masks saved to {masks_path} ({len(masks_df)} rows)")

    all_expl_counts = train_expl_counts + test_expl_counts
    explain_count_df = pd.DataFrame({
        "snapshot_idx": range(len(all_expl_counts)),
        "split": ["train"] * len(train_expl_counts) + ["test"] * len(test_expl_counts),
        "num_explainable_edges": all_expl_counts,
    })
    explain_count_df.to_csv(os.path.join(plot_dir, "explain_count.csv"), index=False)
    print(f"  Explainable-edge counts saved to {plot_dir}/explain_count.csv")

    all_edge_rows = train_edge_rows + test_edge_rows
    explain_edges_df = pd.DataFrame(all_edge_rows)
    explain_edges_df.to_csv(os.path.join(plot_dir, "explain_edges.csv"), index=False)
    print(f"  Explainable edges saved to {plot_dir}/explain_edges.csv ({len(explain_edges_df)} rows)")

    explain_pdf = os.path.join(plot_dir, f"{version}_explain.pdf")
    plot_explain_page(
        num_explainable_list=train_expl_counts,
        train_targets_orig=train_targets_orig,
        loader=loader,
        version=version,
        pdf_path=explain_pdf,
    )
    print(f"  Explain plot saved to {explain_pdf}")

    print("\n" + "=" * 70)
    print("CDCRNN pipeline complete")
    print("=" * 70)
    print(f"  Data/version: {version}")
    print(f"  Model: {model_save_dir}")
    print(f"  Plots/metrics: {plot_dir}")


if __name__ == "__main__":
    main()
