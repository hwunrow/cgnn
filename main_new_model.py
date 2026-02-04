# main_new_model.py
"""
CDCRNN pipeline: temporal dataloader -> train -> evaluate -> DCRNNExplainer.
Predicts delta (y_{t+1} - y_t), uses RMSE loss.
Uses ConfigurableDatasetLoader, CDCRNN, and DCRNNExplainer.

Usage:
    python main_new_model.py data=hospital_advan_cdcrnn model=gcn_hospital
    python main_new_model.py data=case_safegraph_cdcrnn model=gcn_case
"""

import os
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

REPO_DIR = "/burg/apam/users/nhw2114/repos/cgnn"


@hydra.main(version_base=None, config_path="experiments/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = cfg.data.get("version", f"{cfg.data.data_source}_{cfg.data.mobility_source}_cdcrnn")

    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    print("CDCRNN Pipeline")
    print("=" * 70)
    print(f"Version: {version}")
    print(f"Device: {device}")
    print(f"Data source: {cfg.data.data_source}")
    print(f"Mobility source: {cfg.data.mobility_source}")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n[1/4] Loading data...")
    num_historical = cfg.data.get("num_historical_features", 0)
    transform_xy = cfg.data.get("transform_xy", False)
    predict_delta = cfg.data.get("predict_delta", False)
    feature_transform = cfg.data.get("feature_transform", "none")

    loader = ConfigurableDatasetLoader(
        cbsa_list=None,
        cfg=cfg,
        num_historical_features=num_historical,
        transform_xy=transform_xy,
        predict_delta=predict_delta,
        feature_transform=feature_transform,
    )
    dataset = loader.get_dataset()
    train_ratio = cfg.data.get("train_ratio", 0.8)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)

    node_features = train_dataset.features[0].shape[1]
    # Index of target column in snapshot.x (current y_t; delta = y_{t+1} - y_t)
    target_feature_idx = list(loader.feature_cols).index(loader.target_col)
    print(f"  Node features: {node_features}")
    print(f"  Target feature index: {target_feature_idx} ({loader.target_col})")
    print(f"  Train snapshots: {train_dataset.snapshot_count}")
    print(f"  Test snapshots: {test_dataset.snapshot_count}")
    print("  Predicting delta (y_{t+1} - y_t), loss = RMSE")

    # --- 2. Train CDCRNN ---
    print("\n[2/4] Training CDCRNN...")
    model = CDCRNN(node_features=node_features).to(device)

    def _delta_target(snapshot):
        """Target delta = y_{t+1} - y_t (current target value in features)."""
        return snapshot.y - snapshot.x[:, target_feature_idx]

    def _rmse(pred_delta, delta_target):
        return torch.sqrt(F.mse_loss(pred_delta, delta_target))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.get("learning_rate", 1e-5),
        weight_decay=cfg.training.get("weight_decay", 5e-4),
    )
    num_epochs = cfg.training.get("num_epochs", 5000)

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_state = None

    def _eval_test():
        model.eval()
        test_loss_val = 0.0
        with torch.no_grad():
            h_test = None
            for time, snapshot in enumerate(test_dataset):
                snapshot = snapshot.to(device)
                pred, h_test = model(
                    snapshot.x,
                    snapshot.edge_index,
                    snapshot.edge_attr,
                    h_test,
                )
                delta_tgt = _delta_target(snapshot)
                test_loss_val += _rmse(pred.squeeze(), delta_tgt).item()
            test_loss_val /= max(time + 1, 1)
        return test_loss_val

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        h = None
        for time, snapshot in enumerate(train_dataset):
            snapshot = snapshot.to(device)
            pred, h = model(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                h,
            )
            delta_tgt = _delta_target(snapshot)
            loss = _rmse(pred.squeeze(), delta_tgt)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            h = h.detach()

        train_losses.append(epoch_loss / (time + 1))

        # Evaluate on test set
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            test_loss_val = _eval_test()
            test_losses.append(test_loss_val)
            if test_loss_val < best_test_loss:
                best_test_loss = test_loss_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 500 == 0 or epoch == num_epochs - 1:
                print(
                    f"  Epoch {epoch+1}: train_loss={train_losses[-1]:.4f} "
                    f"test_loss={test_loss_val:.4f}"
                )

    # Restore best model (or keep last if no eval was run yet)
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    model_save_dir = os.path.join(REPO_DIR, "models", "gcn_checkpoints", version)
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "best_test_loss": best_test_loss,
        },
        os.path.join(model_save_dir, "cdcrnn_final_model.pt"),
    )
    save_config_to_directory(cfg, model_save_dir)
    print(f"  Model saved to {model_save_dir}")

    # --- 3. Evaluate ---
    print("\n[3/4] Evaluating (delta = y_{t+1} - y_t, RMSE)...")
    model.eval()
    all_pred_delta = []
    all_true_delta = []
    all_pred_y = []
    all_true_y = []
    with torch.no_grad():
        h = None
        for snapshot in test_dataset:
            snapshot = snapshot.to(device)
            pred, h = model(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                h,
            )
            delta_tgt = _delta_target(snapshot)
            all_pred_delta.append(pred.squeeze().cpu().numpy())
            all_true_delta.append(delta_tgt.cpu().numpy())
            pred_y = snapshot.x[:, target_feature_idx].cpu().numpy() + pred.squeeze().cpu().numpy()
            all_pred_y.append(pred_y)
            all_true_y.append(snapshot.y.cpu().numpy())

    pred_delta_arr = np.concatenate(all_pred_delta, axis=0)
    true_delta_arr = np.concatenate(all_true_delta, axis=0)
    pred_y_arr = np.concatenate(all_pred_y, axis=0)
    true_y_arr = np.concatenate(all_true_y, axis=0)
    test_rmse_delta = np.sqrt(np.mean((pred_delta_arr - true_delta_arr) ** 2))
    test_rmse_y = np.sqrt(np.mean((pred_y_arr - true_y_arr) ** 2))
    test_corr_delta = pd.Series(pred_delta_arr.ravel()).corr(pd.Series(true_delta_arr.ravel()))

    # Train metrics
    all_train_pred_delta = []
    all_train_true_delta = []
    all_train_pred_y = []
    all_train_true_y = []
    with torch.no_grad():
        h = None
        for snapshot in train_dataset:
            snapshot = snapshot.to(device)
            pred, h = model(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                h,
            )
            delta_tgt = _delta_target(snapshot)
            all_train_pred_delta.append(pred.squeeze().cpu().numpy())
            all_train_true_delta.append(delta_tgt.cpu().numpy())
            pred_y = snapshot.x[:, target_feature_idx].cpu().numpy() + pred.squeeze().cpu().numpy()
            all_train_pred_y.append(pred_y)
            all_train_true_y.append(snapshot.y.cpu().numpy())
    train_pred_delta_arr = np.concatenate(all_train_pred_delta, axis=0)
    train_true_delta_arr = np.concatenate(all_train_true_delta, axis=0)
    train_pred_y_arr = np.concatenate(all_train_pred_y, axis=0)
    train_true_y_arr = np.concatenate(all_train_true_y, axis=0)
    train_rmse_delta = np.sqrt(np.mean((train_pred_delta_arr - train_true_delta_arr) ** 2))
    train_rmse_y = np.sqrt(np.mean((train_pred_y_arr - train_true_y_arr) ** 2))
    train_corr_delta = pd.Series(train_pred_delta_arr.ravel()).corr(
        pd.Series(train_true_delta_arr.ravel())
    )

    print(f"  Train RMSE (delta): {train_rmse_delta:.4f}  Train RMSE (y): {train_rmse_y:.4f}  corr(delta): {train_corr_delta:.4f}")
    print(f"  Test RMSE (delta):  {test_rmse_delta:.4f}  Test RMSE (y):  {test_rmse_y:.4f}  corr(delta): {test_corr_delta:.4f}")

    eval_df = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rmse_delta": [train_rmse_delta, test_rmse_delta],
            "rmse_y": [train_rmse_y, test_rmse_y],
            "correlation_delta": [train_corr_delta, test_corr_delta],
        }
    )
    plot_dir = os.path.join(REPO_DIR, "plots", version)
    os.makedirs(plot_dir, exist_ok=True)
    eval_df.to_csv(os.path.join(plot_dir, "eval_metrics.csv"), index=False)

    # --- 4. DCRNNExplainer ---
    print("\n[4/4] Running DCRNNExplainer...")
    explain_epochs = cfg.explain.get("epochs", 200)
    edge_threshold = cfg.explain.get("edge_mask_threshold", 0.5)
    explainer = DCRNNExplainer(model, device)

    num_explainable_list = []
    for i, snapshot in enumerate(tqdm(train_dataset, desc="Explain")):
        snapshot = snapshot.to(device)
        importance_mask = explainer.explain(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_attr,
            epochs=explain_epochs,
            lr=0.01,
            verbose=False,
        )
        num_explainable = (importance_mask > edge_threshold).sum().item()
        num_explainable_list.append(num_explainable)

    explain_df = pd.DataFrame(
        {"snapshot_idx": range(len(num_explainable_list)), "num_explainable_edges": num_explainable_list}
    )
    explain_df.to_csv(os.path.join(plot_dir, "explain_count.csv"), index=False)
    print(f"  Explainable-edge counts saved to {plot_dir}/explain_count.csv")

    print("\n" + "=" * 70)
    print("CDCRNN pipeline complete")
    print("=" * 70)
    print(f"  Data/version: {version}")
    print(f"  Model: {model_save_dir}")
    print(f"  Plots/metrics: {plot_dir}")


if __name__ == "__main__":
    main()
