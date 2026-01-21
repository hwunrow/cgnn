"""
Dataloader for PyTorch Geometric Temporal DynamicGraphTemporalSignal.
Handles hospitalization data with edges from Advan Plus mobility data.
"""

import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from tqdm import tqdm

from cgnn.process_data import (
    process_hospitalization_data,
    get_advan_plus_mobility_data,
    get_date_range,
    get_cbsa_list,
    _get_config_value,
    _DEFAULT_START_DATE,
    _DEFAULT_END_DATE,
    _DEFAULT_HOSP_COL,
    _DEFAULT_MOBILITY_CUTOFF,
    _DEFAULT_RAW_ADVAN_PLUS_FILE,
    _DEFAULT_RAW_HOSPITALZATION_FILE,
    _DEFAULT_MAX_MISSING_WEEKS,
)
from omegaconf import DictConfig


class HospitalizationAdvanPlusDatasetLoader:
    """
    Dataloader for hospitalization data with dynamic edges from Advan Plus mobility data.
    
    Creates a DynamicGraphTemporalSignal where:
    - Node features: hospitalization counts for each CBSA at each time step
    - Targets: next time step hospitalization counts (shifted by 1)
    - Edges: spatial edges between CBSAs based on Advan Plus mobility data
    - Edge weights: visitor_home_aggregation from Advan Plus data
    
    Example:
        >>> from cgnn.dataloader import HospitalizationAdvanPlusDatasetLoader
        >>> from torch_geometric_temporal.signal import temporal_signal_split
        >>> 
        >>> # Initialize loader
        >>> loader = HospitalizationAdvanPlusDatasetLoader(cbsa_list=None, cfg=None)
        >>> 
        >>> # Get dataset
        >>> dataset = loader.get_dataset()
        >>> 
        >>> # Split into train/test
        >>> train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        >>> 
        >>> # Iterate over snapshots
        >>> for snapshot in train_dataset:
        >>>     print(snapshot.x.shape)  # [num_nodes, num_features]
        >>>     print(snapshot.edge_index.shape)  # [2, num_edges]
        >>>     print(snapshot.edge_attr.shape)  # [num_edges]
        >>>     print(snapshot.y.shape)  # [num_nodes]
    """
    
    def __init__(
        self,
        cbsa_list=None,
        cfg=None,
        normalize_edge_weights=True,
        edge_weight_transform="log1p",
        edge_weight_normalization="row_sum",
        transform_xy=False,
        xy_transform="log1p",
        predict_delta=False,
    ):
        """
        Initialize the dataloader.
        
        Args:
            cbsa_list (list, optional): List of CBSA codes to include. If None, uses all available.
            cfg (DictConfig, optional): Hydra config object for configuration.
        """
        self.cbsa_list = cbsa_list
        self.cfg = cfg
        self.normalize_edge_weights = normalize_edge_weights
        self.edge_weight_transform = edge_weight_transform
        self.edge_weight_normalization = edge_weight_normalization
        self.transform_xy = transform_xy
        self.xy_transform = xy_transform
        self.predict_delta = predict_delta

        # Get config values or defaults
        if cfg is not None and hasattr(cfg, "data"):
            self.start_date = _get_config_value(
                cfg.data, "start_date", _DEFAULT_START_DATE
            )
            self.end_date = _get_config_value(
                cfg.data, "end_date", _DEFAULT_END_DATE
            )
            self.hosp_col = _get_config_value(
                cfg.data, "hosp_col", _DEFAULT_HOSP_COL
            )
            self.mobility_cutoff = _get_config_value(
                cfg.data, "mobility_cutoff", _DEFAULT_MOBILITY_CUTOFF
            )
        else:
            self.start_date = _DEFAULT_START_DATE
            self.end_date = _DEFAULT_END_DATE
            self.hosp_col = _DEFAULT_HOSP_COL
            self.mobility_cutoff = _DEFAULT_MOBILITY_CUTOFF
        
        # Get date range
        self.dates = get_date_range(self.start_date, self.end_date)
        
        # Get CBSA list if not provided
        if self.cbsa_list is None:
            self.cbsa_list = get_cbsa_list()
        
        # Process data
        self._load_data()
        
    def _load_data(self):
        """Load and process hospitalization and mobility data."""
        print("Loading hospitalization data...")
        self.hosp_df = process_hospitalization_data(
            cbsa_list=self.cbsa_list, cfg=self.cfg
        )
        
        print("Loading Advan Plus mobility data...")
        self.mobility_df = get_advan_plus_mobility_data(
            cbsa_list=self.cbsa_list, cfg=self.cfg
        )
        
        # Ensure dates are datetime
        self.mobility_df["date_range_start"] = pd.to_datetime(
            self.mobility_df["date_range_start"]
        )
        
        # Create CBSA to index mapping (for each time step, nodes are CBSAs)
        self.cbsa_to_idx = {cbsa: idx for idx, cbsa in enumerate(sorted(self.cbsa_list))}
        self.num_nodes = len(self.cbsa_list)
        
        # Ensure hosp_df has collection_week as datetime
        if not pd.api.types.is_datetime64_any_dtype(self.hosp_df["collection_week"]):
            self.hosp_df["collection_week"] = pd.to_datetime(
                self.hosp_df["collection_week"]
            )
    
    def _get_edges_for_date(self, date):
        """
        Get edge indices and weights for a specific date.
        
        Args:
            date: datetime object or date string
            
        Returns:
            tuple: (edge_index, edge_weights) where:
                - edge_index: numpy array of shape [2, num_edges]
                - edge_weights: numpy array of shape [num_edges]
        """
        if isinstance(date, str):
            date_str = date
        else:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        
        # Filter mobility data for this date
        date_mobility = self.mobility_df[
            self.mobility_df["date_range_start"].dt.strftime("%Y-%m-%d") == date_str
        ]
        
        edge_list = []
        weight_list = []
        
        for _, row in date_mobility.iterrows():
            cbsa_orig = row["cbsa_orig"]
            cbsa_dest = row["cbsa_dest"]
            
            # Only include edges between CBSAs in our list
            if cbsa_orig in self.cbsa_to_idx and cbsa_dest in self.cbsa_to_idx:
                orig_idx = self.cbsa_to_idx[cbsa_orig]
                dest_idx = self.cbsa_to_idx[cbsa_dest]
                edge_list.append([orig_idx, dest_idx])
                weight_list.append(row["visitor_home_aggregation"])
        
        if len(edge_list) > 0:
            edge_index = np.array(edge_list, dtype=np.int64).T  # [2, num_edges]
            edge_weights = np.array(weight_list, dtype=np.float32)

            # Optional: transform + normalize edge weights to keep scales stable.
            # For DCRNN-style models, edge weights behave best like normalized
            # transition probabilities (row-stochastic / row-sum normalized).
            if self.normalize_edge_weights:
                if self.edge_weight_transform == "log1p":
                    edge_weights = np.log1p(edge_weights)
                elif self.edge_weight_transform in (None, "none"):
                    pass
                else:
                    raise ValueError(
                        f"Unsupported edge_weight_transform='{self.edge_weight_transform}'"
                    )

                if self.edge_weight_normalization == "row_sum":
                    # Normalize outgoing weights per origin node.
                    src = edge_index[0]
                    out_sum = np.zeros(self.num_nodes, dtype=np.float32)
                    np.add.at(out_sum, src, edge_weights)
                    denom = out_sum[src]
                    # Avoid division by zero (shouldn't happen if denom computed from edges).
                    denom = np.where(denom > 0, denom, 1.0)
                    edge_weights = edge_weights / denom
                elif self.edge_weight_normalization in (None, "none"):
                    pass
                else:
                    raise ValueError(
                        f"Unsupported edge_weight_normalization='{self.edge_weight_normalization}'"
                    )
        else:
            # No edges for this time step - create empty tensors
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_weights = np.empty((0,), dtype=np.float32)
        
        return edge_index, edge_weights
    
    def _get_features_and_target_for_date(self, date):
        """
        Get node features and targets for a specific date.
        
        Args:
            date: datetime object
            
        Returns:
            tuple: (features, target) where:
                - features: numpy array of shape [num_nodes, num_features]
                - target: numpy array of shape [num_nodes]
        """
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        
        # Get current date features
        date_hosp = self.hosp_df[
            self.hosp_df["collection_week"].dt.strftime("%Y-%m-%d") == date_str
        ]
        
        # Get next date for target (shift by 1)
        # Convert date to datetime if needed
        date_dt = pd.to_datetime(date)
        
        # Find index in DatetimeIndex using get_loc
        try:
            date_idx = self.dates.get_loc(date_dt)
            # get_loc can return an integer, slice, or boolean array
            if isinstance(date_idx, slice):
                date_idx = date_idx.start if date_idx.start is not None else 0
            elif isinstance(date_idx, np.ndarray):
                # Boolean array - find first True index
                date_idx = np.where(date_idx)[0][0] if date_idx.any() else len(self.dates) - 1
            # If it's already an integer, use it as is
        except (KeyError, IndexError):
            # Date not found - use get_indexer as fallback
            date_idx_array = self.dates.get_indexer([date_dt])
            date_idx = date_idx_array[0] if date_idx_array[0] >= 0 else len(self.dates) - 1
        
        if date_idx < len(self.dates) - 1:
            next_date = self.dates[date_idx + 1]
            next_date_str = pd.to_datetime(next_date).strftime("%Y-%m-%d")
            next_date_hosp = self.hosp_df[
                self.hosp_df["collection_week"].dt.strftime("%Y-%m-%d") == next_date_str
            ]
        else:
            # Last date - use zeros for target
            next_date_hosp = pd.DataFrame({
                "CBSA": self.cbsa_list,
                self.hosp_col: [0.0] * len(self.cbsa_list)
            })
        
        # Create features array: ensure all CBSAs are present and in correct order
        features_dict = dict(zip(date_hosp["CBSA"], date_hosp[self.hosp_col]))
        target_dict = dict(zip(next_date_hosp["CBSA"], next_date_hosp[self.hosp_col]))
        
        # Build arrays in CBSA order
        features = []
        targets = []
        for cbsa in sorted(self.cbsa_list):
            features.append(features_dict.get(cbsa, 0.0))
            if self.predict_delta:
                targets.append(target_dict.get(cbsa, 0.0) - features_dict.get(cbsa, 0.0))
            else:
                targets.append(target_dict.get(cbsa, 0.0))
        
        features = np.array(features, dtype=np.float32).reshape(-1, 1)  # [num_nodes, 1]
        targets = np.array(targets, dtype=np.float32)  # [num_nodes]

        if self.transform_xy:
            if self.xy_transform == "log1p":
                features = np.log1p(features)
                targets = np.log1p(targets)
            elif self.xy_transform in (None, "none"):
                pass
            else:
                raise ValueError(f"Unsupported xy_transform='{self.xy_transform}'")
        
        return features, targets
    
    def get_dataset(self):
        """
        Create and return a DynamicGraphTemporalSignal dataset.
        
        Returns:
            DynamicGraphTemporalSignal: Dataset containing temporal snapshots
        """
        edge_indices = []
        edge_weights_list = []
        features_list = []
        targets_list = []
        
        print("Creating temporal snapshots...")
        for date in tqdm(self.dates, desc="Processing snapshots"):
            # Get edges for this date
            edge_index, edge_weight = self._get_edges_for_date(date)
            edge_indices.append(edge_index)
            edge_weights_list.append(edge_weight)
            
            # Get features and target for this date
            features, target = self._get_features_and_target_for_date(date)
            features_list.append(features)
            targets_list.append(target)
        
        print(f"Created {len(features_list)} snapshots")
        print(f"Number of nodes per snapshot: {self.num_nodes}")
        print(f"Average edges per snapshot: {np.mean([ei.shape[1] for ei in edge_indices]):.1f}")
        
        return DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights_list,
            features=features_list,
            targets=targets_list
        )
