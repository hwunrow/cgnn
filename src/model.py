import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

# from torch_geometric_temporal.nn.recurrent.attentiontemporalgcn import A3TGCN

import pandas as pd
import process_data
import numpy as np


NODE_FEATURES = 22
OUT_DIM = 1


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


class AttentionGCN(torch.nn.Module):
    def __init__(self, node_features, periods, dropout, out_channels=32):
        super(AttentionGCN, self).__init__()
        self.attention = A3TGCN(node_features, out_channels, periods=periods)
        self.MLP_pred = nn.Linear(out_channels, 1)
        self.periods = periods
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        h = self.attention(x, edge_index, edge_weight)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.relu()

        delta = self.MLP_pred(h)
        h = delta + x[:, 1, -1].unsqueeze(1)
        out = h.relu()

        return out


class GCN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.MLP_embed = nn.Linear(NODE_FEATURES, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(64, 32)
        self.MLP_pred = nn.Linear(64, OUT_DIM)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.MLP_embed(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h0 = h

        h = self.conv1(h, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.relu()
        h = torch.cat((h, h0), dim=1)  # skip connection

        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.cat((h, h0), dim=1)  # skip connection

        delta = self.MLP_pred(h)
        h = delta + x[:, 1].unsqueeze(1)
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
