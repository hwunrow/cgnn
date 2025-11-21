import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

# from torch_geometric_temporal.nn.recurrent.attentiontemporalgcn import A3TGCN

import pandas as pd
from cgnn import process_data
import numpy as np
from omegaconf import DictConfig

# Default values
_DEFAULT_NODE_FEATURES = 1
_DEFAULT_TARGET_FEATURE_IDX = 0
_DEFAULT_OUT_DIM = 1
_DEFAULT_DROPOUT = 0.5


def _get_config_value(cfg, key, default):
    """Helper function to get config value or default."""
    if cfg is None:
        return default
    return cfg.get(key, default)


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        rmsle = torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
        if torch.isnan(rmsle):
            import pdb

            pdb.set_trace()
        return rmsle


class GCN(nn.Module):
    def __init__(self, dropout=None, cfg=None):
        super().__init__()
        # Get values from config or parameters
        if cfg is not None and hasattr(cfg, "model"):
            model_cfg = cfg.model
            node_features = _get_config_value(
                model_cfg, "node_features", _DEFAULT_NODE_FEATURES
            )
            out_dim = _get_config_value(model_cfg, "out_dim", _DEFAULT_OUT_DIM)
            if hasattr(model_cfg, "gcn"):
                gcn_cfg = model_cfg.gcn
                dropout = (
                    dropout
                    if dropout is not None
                    else _get_config_value(gcn_cfg, "dropout", _DEFAULT_DROPOUT)
                )
                hidden_dim = _get_config_value(gcn_cfg, "hidden_dim", 32)
            else:
                dropout = (
                    dropout
                    if dropout is not None
                    else _get_config_value(model_cfg, "dropout", _DEFAULT_DROPOUT)
                )
                hidden_dim = 32
        else:
            node_features = _DEFAULT_NODE_FEATURES
            out_dim = _DEFAULT_OUT_DIM
            dropout = dropout if dropout is not None else _DEFAULT_DROPOUT
            hidden_dim = 32

        self.MLP_embed = nn.Linear(node_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim * 2, hidden_dim)
        self.MLP_pred = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = dropout
        # Get target_feature_idx from config
        if cfg is not None and hasattr(cfg, "model"):
            self.target_feature_idx = _get_config_value(
                cfg.model, "target_feature_idx", _DEFAULT_TARGET_FEATURE_IDX
            )
        else:
            self.target_feature_idx = _DEFAULT_TARGET_FEATURE_IDX

    def forward(self, x, edge_index, edge_weight):
        h = self.MLP_embed(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h0 = h

        h = self.conv1(h, edge_index, edge_weight)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.relu()
        h = torch.cat((h, h0), dim=1)  # skip connection

        h = self.conv2(h, edge_index, edge_weight)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.cat((h, h0), dim=1)  # skip connection

        delta = self.MLP_pred(h)
        # Handle both single-feature (hospitalization) and multi-feature (case/death) data
        # For single feature, use x[:, 0]; for multi-feature, use x[:, 1] (7-day average)
        h = delta + x[:, self.target_feature_idx].unsqueeze(1)
        out = h.relu()

        return out


class prevCase(nn.Module):
    """
    Predicts that the delta in the number of cases will be 0 (and that the actual number
    of cases will be the same as the previous day).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        return x[:, 1]


class prevDelta(nn.Module):
    """
    Predicts that the delta in the number of cases will be the same as the delta from the
    previous day.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        node_dict = process_data.create_node_key()
        x = x.numpy()
        pred_df = pd.DataFrame()
        pred_df["key"] = np.array(list(node_dict.keys()))
        pred_df[["fips", "date"]] = pred_df["key"].str.split("-", n=1, expand=True)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df["CASE_COUNT"] = x[:, 0]
        pred_df["CASE_COUNT_7DAY_AVG"] = x[:, 1]

        pred_df["CASE_DELTA"] = (
            pred_df.groupby(["fips"])["CASE_COUNT_7DAY_AVG"].diff(-1).fillna(0)
        )
        pred_df["CASE_DELTA"] = pred_df["CASE_DELTA"] * -1
        pred_df["CASE_DELTA"] = (
            pred_df.groupby(["fips"])["CASE_DELTA"].shift(1).fillna(0)
        )
        pred_df["pred"] = pred_df["CASE_DELTA"] + pred_df["CASE_COUNT_7DAY_AVG"]
        pred_df.loc[pred_df["pred"] < 0, "pred"] = 0.0

        out = torch.tensor(pred_df["pred"].values, dtype=torch.float32)
        delta = torch.tensor(pred_df["CASE_DELTA"].values, dtype=torch.float32)

        return out
