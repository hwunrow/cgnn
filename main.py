# main.py
"""
Main pipeline script for CGNN using Hydra configuration.

Usage:
    # Run full pipeline
    python main.py

    # Run with different data source
    python main.py data.data_source=case

    # Override hyperparameters
    python main.py training.learning_rate=1e-4 training.num_epochs=10000

    # Multi-run for hyperparameter sweep
    python main.py -m training.learning_rate=1e-5,1e-4,1e-3
"""

import hydra
from omegaconf import DictConfig
import torch
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from cgnn import process_data
from cgnn.model import GCN, RMSLELoss
from cgnn.explain import get_explaination
from cgnn.utils import save_config_to_directory


@hydra.main(version_base=None, config_path="experiments/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline: process data -> train -> evaluate -> explain

    Args:
        cfg: Hydra configuration object
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-generate version name from config parameters
    # Format: {data_source}_{mobility_source}_cutoff{mobility_cutoff}
    version = f"{cfg.data.data_source}_{cfg.data.mobility_source}_cutoff{cfg.data.mobility_cutoff}"

    # Override config's version with generated name
    cfg.data.version = version

    print("=" * 70)
    print("CGNN Pipeline")
    print("=" * 70)
    print(f"Version: {version}")
    print(f"Device: {device}")
    print(f"Data source: {cfg.data.data_source}")
    print(f"Mobility source: {cfg.data.mobility_source}")
    print("=" * 70)

    # Step 1: Process data
    print("\n[1/4] Processing data...")
    data = process_data.create_torch_geometric_data(
        version=version,
        device=device,
        predict_delta=cfg.data.get("predict_delta", False),
        cbsa_list=cfg.data.get("cbsa_list", None),
        include_temporal_edges=cfg.data.get("include_temporal_edges", True),
        data_source=cfg.data.data_source,
        mobility_source=cfg.data.mobility_source,
        cfg=cfg,
    )
    print(f"✓ Data processed successfully!")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Train nodes: {data.train_mask.sum().item()}")
    print(f"  Test nodes: {data.test_mask.sum().item()}")

    # Save data as .pt file for easy loading
    data_path = f"data/processed/{version}/data.pt"
    torch.save(data, data_path)
    print(f"  Data saved to {data_path}")

    # Step 2: Train model
    print("\n[2/4] Training model...")
    graph = data.to(device)

    # Initialize model with config
    model = GCN(dropout=None, cfg=cfg).to(device)
    print(f"  Model: GCN with dropout={model.dropout}")

    # Setup optimizer from config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = RMSLELoss()

    print(f"  Learning rate: {cfg.training.learning_rate}")
    print(f"  Weight decay: {cfg.training.weight_decay}")
    print(f"  Epochs: {cfg.training.num_epochs}")
    print(f"  Training...")

    # Training loop
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")

    for epoch in tqdm(range(cfg.training.num_epochs), desc="Training"):
        # Train step
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index, graph.edge_weight)
        loss = criterion(out[graph.train_mask].squeeze(), graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Evaluate on test set periodically
        if (epoch + 1) % 100 == 0 or epoch == cfg.training.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_out = model(graph.x, graph.edge_index, graph.edge_weight)
                test_loss = criterion(
                    test_out[graph.test_mask].squeeze(), graph.y[graph.test_mask]
                )
            test_losses.append(test_loss.item())

            if test_loss.item() < best_test_loss:
                best_test_loss = test_loss.item()

            if (epoch + 1) % 500 == 0 or epoch == cfg.training.num_epochs - 1:
                print(
                    f"  Epoch {epoch+1}: Train={loss.item():.4f}, Test={test_loss.item():.4f}"
                )

    # Save model
    model_save_dir = f"models/gcn_checkpoints/{version}"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "gcn_final_model.pt")

    torch.save(
        {
            "epoch": cfg.training.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1],
            "test_loss": test_losses[-1] if test_losses else None,
            "best_test_loss": best_test_loss,
        },
        model_path,
    )

    # Save config to model checkpoint directory
    save_config_to_directory(cfg, model_save_dir)
    print(f"✓ Model saved to {model_save_dir}")
    print(f"  Best test loss: {best_test_loss:.4f}")

    # Step 3: Evaluate
    print("\n[3/4] Evaluating model...")
    model.eval()
    graph = data.to(device)

    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_weight)
        criterion = RMSLELoss()
        train_loss = criterion(
            out[graph.train_mask].squeeze(), graph.y[graph.train_mask]
        )
        test_loss = criterion(out[graph.test_mask].squeeze(), graph.y[graph.test_mask])

        # Calculate correlation
        train_pred = out[graph.train_mask].squeeze().cpu().numpy()
        train_true = graph.y[graph.train_mask].cpu().numpy()
        test_pred = out[graph.test_mask].squeeze().cpu().numpy()
        test_true = graph.y[graph.test_mask].cpu().numpy()

        train_corr = pd.Series(train_pred).corr(pd.Series(train_true))
        test_corr = pd.Series(test_pred).corr(pd.Series(test_true))

    print(f"✓ Evaluation results:")
    print(f"  Train Loss (RMSLE): {train_loss.item():.4f}")
    print(f"  Test Loss (RMSLE): {test_loss.item():.4f}")
    print(f"  Train Correlation: {train_corr:.4f}")
    print(f"  Test Correlation: {test_corr:.4f}")

    # Step 4: Explain
    print("\n[4/4] Running explainer...")

    # Load node_dict
    node_dict_path = f"data/processed/{version}/node_dict.pkl"
    if not os.path.exists(node_dict_path):
        raise FileNotFoundError(f"Node dict not found at {node_dict_path}")

    with open(node_dict_path, "rb") as f:
        node_dict = pickle.load(f)

    # Get target date from config or auto-select from test set
    explain_dates = cfg.explain.get("explain_dates", None)
    if explain_dates is None:
        explain_dates = pd.date_range(
            start=cfg.data.start_date, end=cfg.data.end_date, freq="W-MON"
        )

    swapped_dict = {value: key for key, value in node_dict.items()}

    print(f"  Explain dates: {explain_dates}")
    print(f"  Explainer epochs: {cfg.explain.epochs}")
    print(f"  Prev weeks: {cfg.explain.prev_weeks}")
    print(f"  Running explainer...")

    explaination_dict = {}
    self_edge_count = []
    inter_edge_count = []
    temporal_edge_count = []
    total_edge_count = []

    for target_date in tqdm(explain_dates):
        target_date = target_date.strftime("%Y-%m-%d")
        print(f"  Running explainer for {target_date}")
        explanation, subgraph_edges = get_explaination(
            model=model,
            target_date=target_date,
            node_dict=node_dict,
            data=graph,
            cbsa_list=None,
            prev_weeks=None,  # Will use config value
            verbose=True,
            cfg=cfg,
        )
        explaination_dict[target_date] = subgraph_edges

        # create key
        node_idx_start = [
            swapped_dict[item] for item in subgraph_edges[0, :].cpu().numpy()
        ]
        node_idx_end = [
            swapped_dict[item] for item in subgraph_edges[1, :].cpu().numpy()
        ]
        explainable_keys = [
            node_idx_start[i] + "-" + node_idx_end[i] for i in range(len(node_idx_end))
        ]

        fips_start = np.array([node.split("-")[0] for node in node_idx_start])
        fips_end = np.array([node.split("-")[0] for node in node_idx_end])

        date_start = np.array(
            [np.datetime64("".join(node.split("-")[1:])) for node in node_idx_start]
        )
        date_end = np.array(
            [np.datetime64("".join(node.split("-")[1:])) for node in node_idx_end]
        )

        # count types of edges
        inter_edge_count.append(
            np.sum(np.logical_and(fips_start != fips_end, date_start == date_end))
        )
        self_edge_count.append(
            np.sum(np.logical_and(fips_start == fips_end, date_start == date_end))
        )
        temporal_edge_count.append(
            np.sum(np.logical_and(fips_start == fips_end, date_start != date_end))
        )
        total_edge_count.append(subgraph_edges.shape[1])

    explain_count_df = pd.DataFrame(
        {
            "date": explain_dates,
            "self_edge_count": self_edge_count,
            "inter_edge_count": inter_edge_count,
            "temporal_edge_count": temporal_edge_count,
            "total_edge_count": total_edge_count,
        }
    )
    # save explain counts
    plot_save_dir = f"data/plots/{version}"
    os.makedirs(plot_save_dir, exist_ok=True)
    explain_count_df.to_csv(f"{plot_save_dir}/explain_count.csv", index=False)

    print(f"✓ Explanation completed!")
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)
    print(f"\nOutputs saved to:")
    print(f"  Data: data/processed/{version}/")
    print(f"  Model: models/gcn_checkpoints/{version}/")
    print(f"  Plots: plots/{version}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
