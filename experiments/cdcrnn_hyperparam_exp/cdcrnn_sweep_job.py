import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from cgnn.dataloader import ConfigurableDatasetLoader
from cgnn.model import CDCRNN
from cgnn.utils.utils import get_cbsa_list
from cgnn.utils.codebook import TITLE_CBSA_MAP
from torch_geometric_temporal.signal import temporal_signal_split


_TRANSFORM_TO_CONFIG = {
    "raw": "none",
    "logp1": "log1p",
    "diff": "diff",
    "log_diff": "log_diff",
}

# Transforms that benefit from per-node StandardScaler (log_diff is self-normalizing).
_SCALER_TRANSFORMS = {"none", "log1p", "diff"}


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


def get_config_params(transformation: str, target: str, horizon: int):
    """Map high-level labels to dataloader config values."""
    return {
        "target_horizon": int(horizon),
        "x_transform": _TRANSFORM_TO_CONFIG[transformation],
        "y_transform": _TRANSFORM_TO_CONFIG[target],
    }


def reverse_transform_predictions(preds, targets, loader, initial_values=None):
    """
    Reverse transform predictions and targets to original space using loader.y_transform.
    """
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


FOCUS_CBSAS = ["35620", "14460", "45940"]


def plot_experiment_page(
    train_losses,
    test_losses,
    model,
    train_dataset,
    test_dataset,
    loader,
    device,
    experiment_config,
    pdf_path,
    y_mean=None,
    y_std=None,
):
    """Create a one-page PDF (loss, aggregate, 3 CBSA plots)."""
    import matplotlib.pyplot as plt

    model.eval()
    train_preds = []
    train_targets = []
    h_train = None
    with torch.no_grad():
        for snapshot in train_dataset:
            snapshot = snapshot.to(device)
            pred, h_train = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_train)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            if snapshot.y.dim() == 1:
                target = snapshot.y.unsqueeze(1)
            else:
                target = snapshot.y
            train_preds.append(pred[:, 0].cpu().numpy())
            train_targets.append(target[:, 0].cpu().numpy())
            h_train = h_train.detach()

    test_preds = []
    test_targets = []
    h_test = None
    with torch.no_grad():
        for snapshot in test_dataset:
            snapshot = snapshot.to(device)
            pred, h_test = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_test)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            if snapshot.y.dim() == 1:
                target = snapshot.y.unsqueeze(1)
            else:
                target = snapshot.y
            test_preds.append(pred[:, 0].cpu().numpy())
            test_targets.append(target[:, 0].cpu().numpy())
            h_test = h_test.detach()

    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)

    # Undo StandardScaler before reverse-transforming back to original space.
    # Arrays are [T, num_nodes]; y_mean/y_std are [num_nodes] and broadcast.
    if y_mean is not None:
        train_preds = train_preds * y_std + y_mean
        train_targets = train_targets * y_std + y_mean
        test_preds = test_preds * y_std + y_mean
        test_targets = test_targets * y_std + y_mean

    train_preds_orig, train_targets_orig = reverse_transform_predictions(
        train_preds, train_targets, loader
    )

    # For diff/log_diff, the test cumsum must start from the value at the
    # split point (dates[train_size]), not dates[0].
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
        test_preds, test_targets, loader, initial_values=test_initial
    )

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

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.0, 1.2, 1.2])

    # Loss curves
    ax_loss = fig.add_subplot(gs[:, 0])
    epochs = np.arange(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, label="Train", color="#348ABD", linewidth=2)
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
    ax_loss.set_title("Loss Curves")
    ax_loss.legend(frameon=True, framealpha=0.9, loc="best")
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # Aggregate
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

    # CBSA-specific
    axes_cbsa = [fig.add_subplot(gs[row, 2]) for row in range(3)]
    for ax, cbsa_code in zip(axes_cbsa, FOCUS_CBSAS):
        if cbsa_code not in cbsa_to_idx:
            ax.set_visible(False)
            continue
        idx = cbsa_to_idx[cbsa_code]
        name = TITLE_CBSA_MAP.get(cbsa_code, cbsa_code)

        train_truth = train_targets_orig[:, idx]
        train_pred = train_preds_orig[:, idx]
        test_truth = test_targets_orig[:, idx]
        test_pred = test_preds_orig[:, idx]

        ax.plot(train_dates, train_truth, label="Train Truth", color="#333333", linewidth=1.5, alpha=0.8)
        ax.plot(train_dates, train_pred, label="Train Pred", color="#348ABD", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(x=split_date, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.plot(test_dates, test_truth, label="Test Truth", color="#E24A33", marker="o", markersize=3, linestyle="None", alpha=0.8)
        ax.plot(test_dates, test_pred, label="Test Pred", color="#E24A33", linewidth=1.5, linestyle=":", alpha=0.8)

        ax.set_title(name)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax is axes_cbsa[-1]:
            ax.set_xlabel("Date")
        else:
            ax.set_xticklabels([])

        if ax is axes_cbsa[0]:
            ax.legend(frameon=True, framealpha=0.9, fontsize=8, loc="best")

    fig.suptitle(
        f"{experiment_config['transformation']} / {experiment_config['target']}  "
        f"H={experiment_config['horizon']}  Obj={experiment_config['objective']}  "
        f"LR={experiment_config['learning_rate']}  HS={experiment_config['hidden_size']}",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(cbsa_list, cfg_modified, loss_fn, device, num_epochs, experiment_config, out_dir):
    """Single-experiment training + evaluation + plotting."""
    print(f"Training and evaluating experiment: {experiment_config}")
    print("Creating loader...")
    loader = ConfigurableDatasetLoader(cbsa_list=cbsa_list, cfg=cfg_modified)
    dataset = loader.get_dataset()
    dataset = add_self_loops(dataset)

    train_ratio = cfg_modified.get("train_ratio", 0.8)
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
    model = CDCRNN(
        node_features=node_features,
        target_horizon=loader.target_horizon,
        predict_delta=predict_delta,
        hidden_size=experiment_config["hidden_size"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_modified.get("learning_rate", 1e-5),
        weight_decay=cfg_modified.get("weight_decay", 5e-4),
    )

    train_losses = []
    test_losses = []
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        h = None
        num_batches = 0
        for snapshot in train_dataset:
            snapshot = snapshot.to(device)
            pred, h = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h)
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

        if (epoch + 1) % 25 == 0 or epoch == num_epochs - 1 or epoch == 0:
            model.eval()
            test_loss_val = 0.0
            test_batches = 0
            with torch.no_grad():
                h_test = None
                for snapshot in test_dataset:
                    snapshot = snapshot.to(device)
                    pred, h_test = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_test)
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(1)
                    if snapshot.y.dim() == 1:
                        target = snapshot.y.unsqueeze(1)
                    else:
                        target = snapshot.y
                    test_loss_val += loss_fn(pred, target).item()
                    test_batches += 1
            test_loss_val /= max(test_batches, 1)
            test_losses.append((epoch + 1, test_loss_val))

    exp_name = (
        f"{experiment_config['transformation']}_"
        f"{experiment_config['target']}_"
        f"h{experiment_config['horizon']}_"
        f"{experiment_config['objective']}_"
        f"lr{experiment_config['learning_rate']}_"
        f"hs{experiment_config['hidden_size']}"
    )
    pdf_path = os.path.join(out_dir, f"{exp_name}.pdf")
    print(f"Creating plots and saving to {pdf_path}...")
    plot_experiment_page(
        train_losses,
        test_losses,
        model,
        train_dataset,
        test_dataset,
        loader,
        device,
        experiment_config,
        pdf_path,
        y_mean=y_mean,
        y_std=y_std,
    )

    metrics = {
        "transformation": experiment_config["transformation"],
        "target": experiment_config["target"],
        "horizon": experiment_config["horizon"],
        "objective": experiment_config["objective"],
        "learning_rate": experiment_config["learning_rate"],
        "hidden_size": experiment_config["hidden_size"],
        "train_loss": train_losses[-1],
        "test_loss": test_losses[-1][1] if test_losses else float("inf"),
        "num_epochs": num_epochs,
        "timestamp": datetime.now().isoformat(),
    }
    metrics_path = os.path.join(out_dir, f"{exp_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def build_cbsa_list(regions: str):
    """Build CBSA list from comma-separated HHS region codes."""
    region_codes = [r.strip() for r in str(regions).split(",") if r.strip()]
    all_cbsas = []
    for r in region_codes:
        all_cbsas.extend(get_cbsa_list(hhs_region=r))
    return all_cbsas


def main():
    parser = argparse.ArgumentParser(description="Run a single CDCRNN experiment for one transform/target/horizon/objective.")
    parser.add_argument("--transformation", type=str, required=True, choices=list(_TRANSFORM_TO_CONFIG.keys()))
    parser.add_argument("--target", type=str, required=True, choices=list(_TRANSFORM_TO_CONFIG.keys()))
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--objective", type=str, required=True, choices=["RMSE", "RMSLE", "MAE"])
    parser.add_argument("--regions", type=str, default="1,2", help="Comma-separated HHS region codes (e.g. '1,2').")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for Adam optimizer.")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden state size for CDCRNN.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for plots and metrics.")
    parser.add_argument("--num-epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base data config
    base_cfg = OmegaConf.load("experiments/conf/data/hospital_advan_plus_cdcrnn.yaml")
    cfg_modified = OmegaConf.create(dict(base_cfg))
    cfg_overrides = get_config_params(args.transformation, args.target, args.horizon)
    for k, v in cfg_overrides.items():
        cfg_modified[k] = v
    cfg_modified["learning_rate"] = args.learning_rate
    print("=" * 70)
    print("Config:")
    print(OmegaConf.to_yaml(cfg_modified))
    print("=" * 70)

    loss_fn = get_loss_fn(args.objective)
    cbsa_list = build_cbsa_list(args.regions)

    os.makedirs(args.out_dir, exist_ok=True)

    experiment_config = {
        "transformation": args.transformation,
        "target": args.target,
        "horizon": args.horizon,
        "objective": args.objective,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
    }

    train_and_evaluate(
        cbsa_list=cbsa_list,
        cfg_modified=cfg_modified,
        loss_fn=loss_fn,
        device=device,
        num_epochs=args.num_epochs,
        experiment_config=experiment_config,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

