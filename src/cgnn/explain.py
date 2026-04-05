from datetime import datetime
import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.utils import subgraph
from torch_geometric.explain import Explainer, GNNExplainer
from omegaconf import DictConfig, OmegaConf

# Default values
_DEFAULT_EXPLAINER_EPOCHS = 200
_DEFAULT_EDGE_MASK_THRESHOLD = 0.5


class DCRNNExplainer:
    """
    Explains a single temporal snapshot prediction by learning an edge mask.
    Used with CDCRNN (temporal) models; one snapshot at a time.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def explain(
        self,
        x,
        edge_index,
        edge_weight=None,
        epochs=500,
        lr=0.01,
        verbose=False,
        track_loss=False,
    ):
        """
        Learn an edge importance mask for one snapshot.

        Args:
            x: Node features (num_nodes, num_features).
            edge_index: Graph connectivity (2, num_edges).
            edge_weight: Edge weights (num_edges,). Optional.
            epochs: Number of optimization steps for the mask.
            lr: Learning rate for mask logits.
            verbose: If True, print loss components at end.
            track_loss: If True, return per-epoch loss histories.

        Returns:
            If track_loss is False:
                Tensor of shape (num_edges,) with values in [0, 1].
            If track_loss is True:
                (mask_tensor, loss_dict) where loss_dict has keys
                'total', 'dist', 'size', 'ent', each a list of floats.
        """
        num_edges = edge_index.shape[1]
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        mask_logits = torch.randn(
            num_edges, requires_grad=True, device=self.device
        )
        if edge_weight is None:
            base_weights = torch.ones(num_edges, device=self.device)
        else:
            base_weights = edge_weight.to(self.device)

        optimizer = Adam([mask_logits], lr=lr)

        with torch.no_grad():
            base_pred, _ = self.model(x, edge_index, base_weights)
            base_pred = torch.expm1(base_pred)

        loss_history = {
            'total': [], 'dist': [], 'size': [], 'ent': [],
        } if track_loss else None

        for _ in range(epochs):
            optimizer.zero_grad()
            mask = torch.sigmoid(mask_logits)
            masked_weights = base_weights * mask
            masked_pred, _ = self.model(x, edge_index, masked_weights)
            masked_pred = torch.expm1(masked_pred)

            loss_dist = torch.mean((masked_pred - base_pred) ** 2)
            loss_size = 0.000005 * mask.sum()
            ent = -mask * torch.log(mask + 1e-15) - (1 - mask) * torch.log(
                1 - mask + 1e-15
            )
            loss_ent = 0.1 * ent.mean()
            loss = loss_dist + loss_size + loss_ent
            loss.backward()
            optimizer.step()

            if track_loss:
                loss_history['total'].append(loss.item())
                loss_history['dist'].append(loss_dist.item())
                loss_history['size'].append(loss_size.item())
                loss_history['ent'].append(loss_ent.item())

        if verbose:
            print(
                f"  explain loss_dist={loss_dist.item():.6f} "
                f"loss_size={loss_size.item():.6f} loss_ent={loss_ent.item():.6f}"
            )

        mask_result = torch.sigmoid(mask_logits).detach()
        if track_loss:
            return mask_result, loss_history
        return mask_result


def _get_config_value(cfg, key, default):
    """Helper function to get config value or default."""
    if cfg is None:
        return default
    return cfg.get(key, default)


def get_mask(target_date, node_dict, data, cbsa_list=None, prev_weeks=None):

    def get_indices(node_dict, target_date, cbsa_list):
        suffix = "-" + target_date
        if cbsa_list:
            return [node_dict.get(f"{cbsa}{suffix}") for cbsa in cbsa_list]
        else:
            return [idx for key, idx in node_dict.items() if suffix in key]

    target_idxs = get_indices(node_dict, target_date, cbsa_list)
    # print(len(target_idxs))
    assert len(target_idxs) > 0, f"target_idxs empty for {target_date}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_idxs = torch.tensor(target_idxs).to(device)

    # historical_nodes = torch.arange(max(target_idxs), device=device)
    historical_nodes = []
    for node_key, node_idx in node_dict.items():
        date_parts = node_key.split("-")[1:]
        node_date_str = "-".join(date_parts)
        node_date = datetime.strptime(node_date_str, "%Y-%m-%d")

        if node_date <= datetime.strptime(target_date, "%Y-%m-%d"):
            if prev_weeks:
                # check that node is within certain number of weeks of target date
                diff = datetime.strptime(target_date, "%Y-%m-%d") - node_date
                if diff.days // 7 <= prev_weeks:
                    historical_nodes.append(node_idx)
            else:
                historical_nodes.append(node_idx)
    historical_nodes = torch.tensor(historical_nodes, dtype=torch.long).to(device)

    sub_edge_index, sub_edge_weight = subgraph(
        subset=historical_nodes,
        edge_index=data.edge_index,
        edge_attr=data.edge_weight,
        relabel_nodes=True,
        return_edge_mask=False,
    )
    sub_x = data.x[historical_nodes]

    target_node_map = {
        original_idx.item(): new_idx
        for new_idx, original_idx in enumerate(historical_nodes)
    }
    relabeled_target_idxs = torch.tensor(
        [target_node_map[idx.item()] for idx in target_idxs]
    ).to(device)

    return (
        sub_x,
        sub_edge_index,
        sub_edge_weight,
        relabeled_target_idxs,
        historical_nodes,
    )


def get_explaination(
    model,
    target_date,
    node_dict,
    data,
    cbsa_list=None,
    prev_weeks=None,
    verbose=False,
    cfg=None,
):
    """
    Get explanation for model predictions using GNNExplainer.

    Args:
        model: The trained model to explain.
        target_date: Target date string in format 'YYYY-MM-DD'.
        node_dict: Dictionary mapping node keys to indices.
        data: PyTorch Geometric Data object.
        cbsa_list: Optional list of CBSA codes.
        prev_weeks: Optional number of previous weeks to include (None means all).
        verbose: Whether to print verbose output.
        cfg: Optional Hydra config object.

    Returns:
        tuple: (explanation, original_subgraph_edges)
    """
    # Get config values or use defaults
    if cfg is not None and hasattr(cfg, "explain"):
        explain_cfg = cfg.explain
        epochs = _get_config_value(explain_cfg, "epochs", _DEFAULT_EXPLAINER_EPOCHS)
        explanation_type = _get_config_value(explain_cfg, "explanation_type", "model")
        node_mask_type = _get_config_value(explain_cfg, "node_mask_type", "attributes")
        edge_mask_type = _get_config_value(explain_cfg, "edge_mask_type", "object")
        edge_mask_threshold = _get_config_value(
            explain_cfg, "edge_mask_threshold", _DEFAULT_EDGE_MASK_THRESHOLD
        )

        # Get model_config from config or use defaults
        if hasattr(explain_cfg, "model_config"):
            model_config = dict(explain_cfg.model_config)
        else:
            model_config = dict(
                mode="regression",
                task_level="node",
                return_type="raw",
            )

        # Override prev_weeks and verbose from config if not provided
        if prev_weeks is None:
            prev_weeks = _get_config_value(explain_cfg, "prev_weeks", None)
        if verbose is False:
            verbose = _get_config_value(explain_cfg, "verbose", False)
    else:
        epochs = _DEFAULT_EXPLAINER_EPOCHS
        explanation_type = "model"
        node_mask_type = "attributes"
        edge_mask_type = "object"
        edge_mask_threshold = _DEFAULT_EDGE_MASK_THRESHOLD
        model_config = dict(
            mode="regression",
            task_level="node",
            return_type="raw",
        )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )
    # sub_x: data x historical_nodes_idx
    sub_x, sub_edge_index, sub_edge_weight, target_idxs, historical_nodes = get_mask(
        target_date, node_dict, data, cbsa_list, prev_weeks
    )
    assert all(
        idx <= sub_x.shape[0] for idx in target_idxs
    ), f"Some target idx out of bounds: {target_idxs}, sub_x.shape={sub_x.shape}"

    explanation = explainer(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_weight=sub_edge_weight,
        index=target_idxs,
    )
    if verbose:
        print("average explainable edge mask:", explanation.edge_mask.mean())

    subgraph_edges = sub_edge_index[:, explanation.edge_mask > edge_mask_threshold]
    original_subgraph_edges = historical_nodes.cpu()[subgraph_edges.cpu()]

    if verbose:
        print("number of edges in edge mask:", original_subgraph_edges.shape[1])

    return explanation, original_subgraph_edges


def load_aggregate_ts_from_cfg(cfg_file_path):
    """Load aggregate hospitalization or case time series from a config yaml.

    Handles both checkpoint configs (with a top-level 'data' key) and
    standalone data-only configs.

    Args:
        cfg_file_path: Path to a Hydra config yaml file.

    Returns:
        (ts, dates) where ts is a float64 numpy array of aggregate target
        values and dates is the list of snapshot dates from the loader.
    """
    from cgnn.dataloader import ConfigurableDatasetLoader

    raw_cfg = OmegaConf.load(cfg_file_path)
    # Checkpoint configs already have a 'data' sub-key; data-only configs don't.
    if "data" in raw_cfg:
        cfg = raw_cfg
    else:
        cfg = OmegaConf.create({"data": raw_cfg})

    loader = ConfigurableDatasetLoader(cfg=cfg)
    dates = loader.dates

    agg = (
        loader.data_df
        .groupby(loader.date_col)[loader.target_col]
        .sum()
        .sort_index()
    )
    ts = np.array([agg.get(d, 0.0) for d in dates], dtype=np.float64)
    return ts, dates


def compute_lead_lag(edge_counts, aggregate_ts, max_lag=None):
    """Compute shifted cross-correlation between explainable edge counts
    and an aggregate target time series.

    A positive best_lag means edge counts lead the target (edge counts peak
    *best_lag* snapshots before the target peaks).

    Args:
        edge_counts: Array-like of explainable edge counts per snapshot.
        aggregate_ts: Array-like of aggregate target values per snapshot.
        max_lag: If set, restrict returned lags to [-max_lag, max_lag].

    Returns:
        (lags, correlations, best_lag, best_corr, x_transformed, y_transformed)
        where lags and correlations are numpy arrays of the full cross-correlation,
        best_lag/best_corr are the lag and normalized correlation at max |correlation|,
        and x_transformed/y_transformed are the mean-centered input series.
    """
    from scipy import signal

    x = np.asarray(edge_counts, dtype=np.float64)
    y = np.asarray(aggregate_ts, dtype=np.float64)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    xc = x - x.mean()
    yc = y - y.mean()
    corr = signal.correlate(xc, yc, mode="full")
    lags = signal.correlation_lags(n, n, mode="full")

    x_norm = np.sqrt((xc ** 2).sum())
    y_norm = np.sqrt((yc ** 2).sum())
    norm = x_norm * y_norm
    corr = corr / norm if norm > 0 else corr
    x_std = x.std()
    y_std = y.std()
    xc = xc / x_std if x_std > 0 else xc
    yc = yc / y_std if y_std > 0 else yc

    if max_lag is not None:
        mask = np.abs(lags) <= max_lag
        lags = lags[mask]
        corr = corr[mask]

    best_idx = np.argmax(np.abs(corr))
    return lags, corr, int(lags[best_idx]), float(corr[best_idx]), xc, yc
