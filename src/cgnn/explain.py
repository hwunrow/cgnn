from datetime import datetime
import torch
from torch_geometric.utils import subgraph
from torch_geometric.explain import Explainer, GNNExplainer
from omegaconf import DictConfig

# Default values
_DEFAULT_EXPLAINER_EPOCHS = 200
_DEFAULT_EDGE_MASK_THRESHOLD = 0.5


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
