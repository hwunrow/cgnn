import torch
import numpy as np
from tqdm import tqdm
from model import AttentionGCN, RMSLELoss
import process_data
from torch_geometric_temporal.signal import temporal_signal_split


import yaml
import os
import csv


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def run_model(out_dir, gnn_params, optim_params):
    # create output directory
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.csv"), "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(
            [
                "periods",
                "out_channels",
                "dropout",
                "base_lr",
                "train_loss",
                "test_loss",
            ]
        )

    temporal_data = process_data.create_torch_geometric_temporal_data()
    train_dataset, test_dataset = temporal_signal_split(
        temporal_data, train_ratio=60 / 92
    )
    for periods in tqdm(gnn_params["periods"]):
        for out_channels in tqdm(gnn_params["out_channels"], leave=False):
            for dropout in tqdm(gnn_params["dropout"], leave=False):
                for base_lr in tqdm(optim_params["base_lr"], leave=False):
                    train_loss = []
                    test_loss = []
                    for _ in range(2):  # repeat 3 times
                        model = model = AttentionGCN(
                            node_features=22,
                            periods=periods,
                            dropout=dropout,
                            out_channels=out_channels,
                        )
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=base_lr,
                            weight_decay=optim_params["weight_decay"],
                        )
                        criterion = RMSLELoss()

                        for epoch in range(optim_params["max_epoch"]):
                            # train
                            cost = 0
                            for t in range(model.periods, train_dataset.snapshot_count):
                                snapshots = train_dataset[t - model.periods : t]
                                y_hat, _ = model(
                                    torch.stack(
                                        [snapshot.x for snapshot in snapshots],
                                        dim=2,
                                    ),
                                    snapshots[-1].edge_index,
                                    snapshots[-1].edge_attr,
                                )
                                cost = cost + criterion(
                                    y_hat.squeeze(), snapshots[-1].y
                                )
                            cost = cost / (t + 1)
                            cost.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        train_loss.append(cost.cpu().detach().numpy())

                        # test
                        model.eval()
                        cost = 0
                        for t in range(model.periods, test_dataset.snapshot_count):
                            snapshots = test_dataset[t - model.periods : t]
                            y_hat, _ = model(
                                torch.stack(
                                    [snapshot.x for snapshot in snapshots], dim=2
                                ),
                                snapshots[-1].edge_index,
                                snapshots[-1].edge_attr,
                            )
                            cost = cost + criterion(y_hat.squeeze(), snapshots[-1].y)
                        cost = cost / (t + 1)
                        cost = cost.item()
                        test_loss.append(cost)

                    # save results
                    print(
                        f"periods: {periods}, out_channels: {out_channels}, dropout: {dropout}, base_lr: {base_lr}"
                    )
                    with open(os.path.join(out_dir, "results.csv"), "a") as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=",")
                        csv_writer.writerow(
                            [
                                periods,
                                out_channels,
                                dropout,
                                base_lr,
                                np.mean(train_loss),
                                np.mean(test_loss),
                            ]
                        )


if __name__ == "__main__":
    config = load_config("a3tgcn.yaml")
    run_model(config["out_dir"], config["gnn"], config["optim"])
