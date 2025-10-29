from datetime import datetime
import torch
from torch_geometric.utils import subgraph
from torch_geometric.explain import Explainer, GNNExplainer


def get_mask(target_date, node_dict, data, cbsa_list=None, prev_weeks=None):

    def get_indices(node_dict, target_date, cbsa_list):
        suffix = "-" + target_date
        if cbsa_list:
            return [node_dict.get(f"{cbsa}{suffix}") for cbsa in cbsa_list]
        else:
            return [idx for key, idx in node_dict.items() if suffix in key]
    
    target_idxs = get_indices(node_dict, target_date, cbsa_list)
    print(len(target_idxs))
    assert len(target_idxs) > 0, f"target_idxs empty for {target_date}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_idxs = torch.tensor(target_idxs).to(device)
    
    # historical_nodes = torch.arange(max(target_idxs), device=device)
    historical_nodes = []
    for node_key, node_idx in node_dict.items():
        date_parts = node_key.split('-')[1:]
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
    
    sub_edge_index, sub_edge_weight  = subgraph(
        subset=historical_nodes,
        edge_index=data.edge_index,
        edge_attr=data.edge_weight,
        relabel_nodes=True,
        return_edge_mask=False
    )
    sub_x = data.x[historical_nodes]

    target_node_map = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(historical_nodes)}
    relabeled_target_idxs = torch.tensor([target_node_map[idx.item()] for idx in target_idxs]).to(device)

    return sub_x, sub_edge_index, sub_edge_weight, relabeled_target_idxs, historical_nodes


def get_explaination(model, target_date, node_dict, data, cbsa_list=None, prev_weeks=None, verbose=False):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    # sub_x: data x historical_nodes_idx
    sub_x, sub_edge_index, sub_edge_weight, target_idxs, historical_nodes = get_mask(target_date, node_dict, data, cbsa_list, prev_weeks)
    assert all(idx <= sub_x.shape[0] for idx in target_idxs), f"Some target idx out of bounds: {target_idxs}, sub_x.shape={sub_x.shape}"

    explanation = explainer(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_weight=sub_edge_weight,
        index=target_idxs,
    )
    if verbose:
        print("average explainable edge mask:", explanation.edge_mask.mean())
    
    subgraph_edges = sub_edge_index[:,explanation.edge_mask > 0.5]
    original_subgraph_edges = historical_nodes.cpu()[subgraph_edges.cpu()]
    
    if verbose:
        print("number of edges in edge mask:", original_subgraph_edges.shape[1])
        print("number of self edges:", sum(original_subgraph_edges[0,:] == original_subgraph_edges[1,:]))
        print("number of inter edges:", sum(original_subgraph_edges[0,:] != original_subgraph_edges[1,:]))

    return explanation, original_subgraph_edges