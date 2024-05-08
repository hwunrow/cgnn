import torch
from tqdm import tqdm
from model import GCN, RMSLELoss
import process_data

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
                "dropout",
                "base_lr",
                "max_epoch",
                "weight_decay",
                "train_loss",
                "test_loss",
            ]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = process_data.create_torch_geometric_data("gcn", device)

    graph = data.to(device)

    for dropout in gnn_params["dropout"]:
        for base_lr in optim_params["base_lr"]:
            for max_epoch in optim_params["max_epoch"]:
                for weight_decay in optim_params["weight_decay"]:
                    model = GCN(dropout=dropout).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=base_lr, weight_decay=weight_decay
                    )
                    criterion = RMSLELoss()

                    train_loss = []
                    test_loss = []
                    for epoch in tqdm(range(max_epoch)):
                        # train
                        model.train()
                        optimizer.zero_grad()
                        out, _ = model(graph.x.to(device), graph.edge_index.to(device))
                        loss = criterion(
                            out[graph.train_mask].squeeze(), graph.y[graph.train_mask]
                        )
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.cpu().detach().numpy())

                        # test
                        model.eval()
                        out, _ = model(graph.x, graph.edge_index)
                        loss = criterion(
                            out[graph.test_mask].squeeze(), graph.y[graph.test_mask]
                        )
                        test_loss.append(loss.cpu().detach().numpy())

                    # save results
                    print(
                        f"dropout: {dropout}, base_lr: {base_lr}, max_epoch: {max_epoch}, weight_decay: {weight_decay}"
                    )
                    with open(os.path.join(out_dir, "results.csv"), "a") as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=",")
                        csv_writer.writerow(
                            [
                                dropout,
                                base_lr,
                                max_epoch,
                                weight_decay,
                                train_loss[-1],
                                test_loss[-1],
                            ]
                        )


if __name__ == "__main__":
    config = load_config("cgnn.yaml")
    run_model(config["out_dir"], config["gnn"], config["optim"])
