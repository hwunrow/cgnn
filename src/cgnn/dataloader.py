"""
Dataloader for PyTorch Geometric Temporal DynamicGraphTemporalSignal.
Provides a config-driven loader that can combine different node data sources
(e.g. hospitalization vs case/death) with different mobility/edge sources
(e.g. Advan, Advan Plus, SafeGraph).
"""

import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from tqdm import tqdm

from cgnn.process_data import (
    process_hospitalization_data,
    process_case_death_data,
    get_mobility_df_by_source,
    get_date_range,
    get_cbsa_list,
    _get_config_value,
    _DEFAULT_START_DATE,
    _DEFAULT_END_DATE,
    _DEFAULT_HOSP_COL,
    _DEFAULT_MOBILITY_CUTOFF,
)


def _resolve_data_cfg(cfg):
    """
    Return the config node that actually contains the data keys.

    Supports both:
    - Hydra-style: cfg.data.<keys>
    - Direct YAML load: cfg.<keys>
    - "double nested" YAMLs that have cfg.data.data.<keys>
    """
    if cfg is None:
        return None

    data_cfg = cfg.data if hasattr(cfg, "data") else cfg
    # Some files in this repo include an extra "data:" nesting.
    if (
        data_cfg is not None
        and hasattr(data_cfg, "get")
        and data_cfg.get("data", None) is not None
        and not any(k in data_cfg for k in ("start_date", "end_date", "data_source"))
    ):
        data_cfg = data_cfg.get("data")
    return data_cfg


class ConfigurableDatasetLoader:
    """
    Configurable dataloader that creates a `DynamicGraphTemporalSignal`.
    
    Target (e.g. hospitalization vs case) and mobility source (e.g. Advan, Advan Plus,
    SafeGraph) are determined by config (data_source, mobility_source).
    
    Creates a DynamicGraphTemporalSignal where:
    - **Node features**: determined by `data_source` and `feature_cols`.
      - If `data_source="hospital"`, defaults to `[hosp_col]`.
      - If `data_source="case"`, defaults to the config's `feature_cols` (or a
        built-in default list compatible with `process_case_death_data`).
      - If `num_historical_features=K > 0`, features are expanded by concatenating
        the current date block with the previous K date blocks (most recent to oldest).
        When historical dates don't exist (at start of date range), the earliest
        available value is repeated.
      - Transform applied to features is controlled by ``x_transform`` (none, diff, log1p, log_diff).
        With diff or log_diff, each block is replaced by first (log-)difference; snapshot.x then
        no longer contains levels.
    - **Targets**:
      - `data_source="hospital"`: next time step of `hosp_col`
      - `data_source="case"`: next time step of `CASE_COUNT_7DAY_AVG
      - Can be overridden via config `target_col`
      - If `target_horizon=k` (default 1), targets are for t+1, t+2, ..., t+k. Then
        `snapshot.y` has shape `[num_nodes, k]`; with `target_horizon=1` it remains
        `[num_nodes]` for backward compatibility.
    - **Edges**: spatial edges between CBSAs based on `mobility_source`
      (`advan`, `advan_plus`, or `safegraph`)
    - **Edge weights**: `visitor_home_aggregation` from the chosen mobility data
    
    Example:
        >>> from cgnn.dataloader import ConfigurableDatasetLoader
        >>> from torch_geometric_temporal.signal import temporal_signal_split
        >>> 
        >>> # Initialize loader (parameters read from cfg)
        >>> loader = ConfigurableDatasetLoader(
        ...     cbsa_list=None, 
        ...     cfg=cfg,
        ... )
        >>> 
        >>> # Get dataset
        >>> dataset = loader.get_dataset()
        >>> 
        >>> # Split into train/test
        >>> train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        >>> 
        >>> # Iterate over snapshots
        >>> for snapshot in train_dataset:
        >>>     print(snapshot.x.shape)  # [num_nodes, 4] (current + 3 historical)
        >>>     print(snapshot.edge_index.shape)  # [2, num_edges]
        >>>     print(snapshot.edge_attr.shape)  # [num_edges]
        >>>     print(snapshot.y.shape)  # [num_nodes] or [num_nodes, target_horizon]
    """
    
    def __init__(
        self,
        cbsa_list=None,
        cfg=None,
    ):
        """
        Initialize the dataloader.

        Args:
            cbsa_list (list, optional): List of CBSA codes to include. If None, uses all available.
            cfg (DictConfig, optional): Hydra config object for configuration.
                Transformation and edge parameters read from config:
                - x_transform (str, default: "none"): feature space; one of "none", "diff", "log1p", "log_diff"
                - y_transform (str, default: "none"): target space; one of "none", "diff", "log1p", "log_diff"
                - normalize_edge_weights (bool, default: True)
                - edge_weight_transform (str, default: "log1p")
                - edge_weight_normalization (str, default: "row_sum")
                - num_historical_features (int, default: 0)
                - target_horizon (int, default: 1)
                predict_delta is derived from y_transform (True when y_transform in ("diff", "log_diff")).
        """
        self.cbsa_list = cbsa_list
        self.cfg = cfg

        # Get config values or defaults
        data_cfg = _resolve_data_cfg(cfg)
        
        # Read transformation and target parameters from config
        self.normalize_edge_weights = _get_config_value(data_cfg, "normalize_edge_weights", True)
        self.edge_weight_transform = _get_config_value(data_cfg, "edge_weight_transform", "log1p")
        self.edge_weight_normalization = _get_config_value(data_cfg, "edge_weight_normalization", "row_sum")

        # x_transform and y_transform (none | diff | log1p | log_diff)
        _VALID_TRANSFORMS = ("none", "diff", "log1p", "log_diff")
        x_raw = _get_config_value(data_cfg, "x_transform", "none")
        y_raw = _get_config_value(data_cfg, "y_transform", "none")
        self.x_transform = str(x_raw).lower() if x_raw else "none"
        self.y_transform = str(y_raw).lower() if y_raw else "none"
        if self.x_transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"Unsupported x_transform='{self.x_transform}'. "
                "Expected 'none', 'diff', 'log1p', or 'log_diff'."
            )
        if self.y_transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"Unsupported y_transform='{self.y_transform}'. "
                "Expected 'none', 'diff', 'log1p', or 'log_diff'."
            )

        self.num_historical_features = _get_config_value(data_cfg, "num_historical_features", 0)
        self.num_historical_features = int(self.num_historical_features)

        self.start_date = _get_config_value(data_cfg, "start_date", _DEFAULT_START_DATE)
        self.end_date = _get_config_value(data_cfg, "end_date", _DEFAULT_END_DATE)
        self.hosp_col = _get_config_value(data_cfg, "hosp_col", _DEFAULT_HOSP_COL)
        self.mobility_cutoff = _get_config_value(
            data_cfg, "mobility_cutoff", _DEFAULT_MOBILITY_CUTOFF
        )
        self.data_source = _get_config_value(data_cfg, "data_source", "hospital")
        self.mobility_source = _get_config_value(
            data_cfg, "mobility_source", "advan_plus"
        )
        self.target_horizon = _get_config_value(
            data_cfg, "target_horizon", 1
        )
        self.target_horizon = int(self.target_horizon)
        if self.target_horizon < 1:
            raise ValueError("target_horizon must be >= 1.")

        # Optional: explicit feature/target schema (useful for case/death configs)
        self.feature_cols = _get_config_value(data_cfg, "feature_cols", None)
        self.target_col = _get_config_value(data_cfg, "target_col", None)

        if self.data_source is not None:
            self.data_source = str(self.data_source).lower()
        if self.mobility_source is not None:
            self.mobility_source = str(self.mobility_source).lower()
        
        # Get date range
        self.dates = get_date_range(self.start_date, self.end_date)
        
        # Get CBSA list if not provided
        if self.cbsa_list is None:
            self.cbsa_list = get_cbsa_list()
        
        # Process data
        self._load_data()

    @property
    def predict_delta(self):
        """True when y_transform is diff or log_diff (for CDCRNN and callers)."""
        return self.y_transform in ("diff", "log_diff")
        
    def _load_data(self):
        """Load and process node data and mobility/edge data."""
        if self.data_source not in {"hospital", "case"}:
            raise ValueError(
                f"Unsupported data_source='{self.data_source}'. Expected 'hospital' or 'case'."
            )
        if self.mobility_source not in {"advan", "advan_plus", "safegraph"}:
            raise ValueError(
                f"Unsupported mobility_source='{self.mobility_source}'. "
                "Expected one of {'advan','advan_plus','safegraph'}."
            )

        if self.data_source == "hospital":
            print("Loading hospitalization data...")
            self.data_df = process_hospitalization_data(cbsa_list=self.cbsa_list, cfg=self.cfg)
            self.date_col = "collection_week"
            if self.feature_cols is None:
                self.feature_cols = [self.hosp_col]
            if self.target_col is None:
                self.target_col = self.hosp_col
        else:
            print("Loading case/death data...")
            death_df, case_df = process_case_death_data(cbsa_list=self.cbsa_list, cfg=self.cfg)
            # Merge case + death features into a single frame.
            self.data_df = case_df.merge(death_df, on=["date", "CBSA", "node_key"], how="inner")
            self.date_col = "date"
            if self.feature_cols is None:
                # Default feature set matches `create_torch_geometric_data` in process_data.py
                self.feature_cols = [
                    "CASE_COUNT",
                    "CASE_COUNT_7DAY_AVG",
                    "CASE_COUNT_PREV_0",
                    "CASE_COUNT_PREV_1",
                    "CASE_COUNT_PREV_2",
                    "CASE_COUNT_PREV_3",
                    "CASE_COUNT_PREV_4",
                    "CASE_COUNT_PREV_5",
                    "DEATH_COUNT",
                    "DEATH_COUNT_7DAY_AVG",
                    "DEATH_COUNT_PREV_0",
                    "DEATH_COUNT_PREV_1",
                    "DEATH_COUNT_PREV_2",
                    "DEATH_COUNT_PREV_3",
                    "DEATH_COUNT_PREV_4",
                    "DEATH_COUNT_PREV_5",
                ]
            if self.target_col is None:
                self.target_col = "CASE_COUNT_7DAY_AVG"
                # self.target_col = "CASE_DELTA" if self.predict_delta else "CASE_COUNT_7DAY_AVG"

        print(f"Loading mobility data (source={self.mobility_source})...")
        self.mobility_df = get_mobility_df_by_source(
            self.mobility_source, cbsa_list=self.cbsa_list, cfg=self.cfg
        )
        
        # Ensure dates are datetime
        self.mobility_df["date_range_start"] = pd.to_datetime(self.mobility_df["date_range_start"])
        if self.date_col not in self.data_df.columns:
            raise KeyError(
                f"Expected date column '{self.date_col}' in data_df (data_source={self.data_source})."
            )
        if not pd.api.types.is_datetime64_any_dtype(self.data_df[self.date_col]):
            self.data_df[self.date_col] = pd.to_datetime(self.data_df[self.date_col])

        # Validate configured feature/target columns exist.
        missing_feature_cols = [c for c in self.feature_cols if c not in self.data_df.columns]
        if missing_feature_cols:
            raise KeyError(
                f"Missing feature_cols in processed data: {missing_feature_cols}. "
                f"Available columns include: {sorted(self.data_df.columns.tolist())[:25]}..."
            )
        if self.target_col not in self.data_df.columns:
            raise KeyError(
                f"Missing target_col='{self.target_col}' in processed data. "
                f"Available columns include: {sorted(self.data_df.columns.tolist())[:25]}..."
            )
        
        # Restrict to CBSAs that appear in both node data and mobility edges.
        data_cbsa_set = set(self.data_df["CBSA"].astype(str).unique())
        mobility_cbsa_set = set(self.mobility_df["cbsa_orig"].astype(str).unique()).intersection(
            set(self.mobility_df["cbsa_dest"].astype(str).unique())
        )
        requested_cbsa_set = set(map(str, self.cbsa_list))
        cbsa_set = requested_cbsa_set.intersection(data_cbsa_set).intersection(mobility_cbsa_set)
        if not cbsa_set:
            raise ValueError(
                "No overlapping CBSAs between requested CBSA list, processed node data, and mobility data."
            )
        self.cbsa_list = sorted(cbsa_set)
        self.data_df["CBSA"] = self.data_df["CBSA"].astype(str)
        self.data_df = self.data_df.loc[self.data_df["CBSA"].isin(self.cbsa_list)].copy()
        self.mobility_df["cbsa_orig"] = self.mobility_df["cbsa_orig"].astype(str)
        self.mobility_df["cbsa_dest"] = self.mobility_df["cbsa_dest"].astype(str)
        self.mobility_df = self.mobility_df.loc[
            (self.mobility_df["cbsa_orig"].isin(self.cbsa_list))
            & (self.mobility_df["cbsa_dest"].isin(self.cbsa_list))
        ].copy()

        # Create CBSA to index mapping (for each time step, nodes are CBSAs)
        self.cbsa_to_idx = {cbsa: idx for idx, cbsa in enumerate(self.cbsa_list)}
        self.num_nodes = len(self.cbsa_list)
    
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
                  If num_historical_features=0: [num_nodes, 1] (current date only)
                  If num_historical_features=K: [num_nodes, K+1] (current + K previous dates)
                - target: numpy array of shape [num_nodes] if target_horizon==1, else
                  [num_nodes, target_horizon] for multi-horizon (t+1, t+2, ..., t+k).
        """
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        
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

        # Current date dataframe (for features, and for delta targets if requested)
        curr_df = self.data_df[
            self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == date_str
        ]
        curr_target_dict = dict(zip(curr_df["CBSA"], curr_df[self.target_col]))
        curr_vals = np.array(
            [curr_target_dict.get(cbsa, 0.0) for cbsa in self.cbsa_list], dtype=np.float32
        )

        # Build targets: first compute levels per horizon, then apply y_transform
        level_arrays = []  # level at t+(h+1) for each h
        for h in range(self.target_horizon):
            future_idx = date_idx + 1 + h
            if future_idx < len(self.dates):
                future_date = self.dates[future_idx]
                future_date_str = pd.to_datetime(future_date).strftime("%Y-%m-%d")
                future_df = self.data_df[
                    self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == future_date_str
                ]
                future_target_dict = dict(zip(future_df["CBSA"], future_df[self.target_col]))
                future_vals = np.array(
                    [future_target_dict.get(cbsa, 0.0) for cbsa in self.cbsa_list],
                    dtype=np.float32,
                )
            else:
                future_vals = np.zeros(len(self.cbsa_list), dtype=np.float32)
            level_arrays.append(future_vals)

        # Apply y_transform to get target array
        if self.y_transform == "none":
            horizon_targets = level_arrays
        elif self.y_transform == "log1p":
            if any((a < -1).any() for a in level_arrays):
                raise ValueError(
                    "Cannot apply log1p to targets with values less than -1. "
                    "Use y_transform='none', 'diff', or 'log_diff'."
                )
            horizon_targets = [np.log1p(a) for a in level_arrays]
        elif self.y_transform == "diff":
            horizon_targets = []
            for h in range(self.target_horizon):
                prev_idx = date_idx + h
                if prev_idx < len(self.dates):
                    prev_date = self.dates[prev_idx]
                    prev_date_str = pd.to_datetime(prev_date).strftime("%Y-%m-%d")
                    prev_df = self.data_df[
                        self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == prev_date_str
                    ]
                    prev_dict = dict(zip(prev_df["CBSA"], prev_df[self.target_col]))
                    prev_vals = np.array(
                        [prev_dict.get(cbsa, 0.0) for cbsa in self.cbsa_list], dtype=np.float32
                    )
                else:
                    prev_vals = np.zeros(len(self.cbsa_list), dtype=np.float32)
                horizon_targets.append(level_arrays[h] - prev_vals)
        else:  # log_diff
            horizon_targets = []
            for h in range(self.target_horizon):
                prev_idx = date_idx + h
                if prev_idx < len(self.dates):
                    prev_date = self.dates[prev_idx]
                    prev_date_str = pd.to_datetime(prev_date).strftime("%Y-%m-%d")
                    prev_df = self.data_df[
                        self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == prev_date_str
                    ]
                    prev_dict = dict(zip(prev_df["CBSA"], prev_df[self.target_col]))
                    prev_vals = np.array(
                        [prev_dict.get(cbsa, 0.0) for cbsa in self.cbsa_list], dtype=np.float32
                    )
                else:
                    prev_vals = np.zeros(len(self.cbsa_list), dtype=np.float32)
                horizon_targets.append(
                    np.log1p(np.abs(level_arrays[h])) - np.log1p(np.abs(prev_vals))
                )

        targets = np.stack(horizon_targets, axis=1).astype(np.float32)  # [num_nodes, target_horizon]
        if self.target_horizon == 1:
            targets = targets.squeeze(axis=1)  # [num_nodes] for backward compatibility

        # Features: concatenate current + K previous blocks (each block has len(feature_cols) features)
        # When x_transform is diff or log_diff, we need one extra prior block to compute first differences.
        num_blocks = self.num_historical_features + 1
        if self.x_transform in ("diff", "log_diff"):
            num_blocks += 1  # extra block for (t - (K+1)) to diff the last historical block
        blocks = []
        earliest_date = self.dates[0]
        earliest_date_str = pd.to_datetime(earliest_date).strftime("%Y-%m-%d")
        earliest_df = self.data_df[
            self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == earliest_date_str
        ]

        for k in range(0, num_blocks):
            hist_idx = date_idx - k
            if hist_idx >= 0:
                hist_date = self.dates[hist_idx]
                hist_date_str = pd.to_datetime(hist_date).strftime("%Y-%m-%d")
                hist_df = self.data_df[
                    self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == hist_date_str
                ]
            else:
                hist_df = earliest_df

            # Build [num_nodes, num_features] for this time block
            feature_dicts = {
                col: dict(zip(hist_df["CBSA"], hist_df[col])) for col in self.feature_cols
            }
            block = np.array(
                [[feature_dicts[col].get(cbsa, 0.0) for col in self.feature_cols] for cbsa in self.cbsa_list],
                dtype=np.float32,
            )
            blocks.append(block)

        if self.x_transform == "diff":
            # Replace each block by first difference: block_k becomes (block_k - block_{k+1}).
            num_output_blocks = self.num_historical_features + 1
            diff_blocks = []
            for k in range(num_output_blocks):
                curr_block = blocks[k]
                prev_block = blocks[k + 1]  # we built one extra block
                diff_blocks.append(curr_block - prev_block)
            blocks = diff_blocks
        elif self.x_transform == "log_diff":
            # Apply log1p before differencing: log(x_t + 1) - log(x_{t-1} + 1)
            num_output_blocks = self.num_historical_features + 1
            diff_blocks = []
            for k in range(num_output_blocks):
                curr_block = blocks[k]
                prev_block = blocks[k + 1]
                curr_log = np.log1p(np.abs(curr_block))
                prev_log = np.log1p(np.abs(prev_block))
                diff_blocks.append(curr_log - prev_log)
            blocks = diff_blocks

        # blocks are [current, prev1, prev2, ...] (increasing k)
        features = np.concatenate(blocks, axis=1)  # [num_nodes, (K+1)*num_features]

        if self.x_transform == "log1p":
            if (features < 0).any():
                raise ValueError(
                    "Cannot apply log1p to features with negative values. "
                    "Use x_transform='none', 'diff', or 'log_diff'."
                )
            features = np.log1p(features)

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

    def get_expanding_window_cv_splits(
        self,
        dataset,
        initial_train_weeks,
        test_weeks,
        step_weeks,
    ):
        """
        Create expanding-window train/test folds by slicing temporal snapshots.

        Notes:
        - Assumes `dataset` is a `DynamicGraphTemporalSignal` built from this loader,
          so snapshot ordering matches `self.dates`.
        - Returns contiguous prefix train windows + contiguous test windows.
        """
        initial_train_weeks = int(initial_train_weeks)
        test_weeks = int(test_weeks)
        step_weeks = int(step_weeks)

        if initial_train_weeks < 1:
            raise ValueError("initial_train_weeks must be >= 1.")
        if test_weeks < 1:
            raise ValueError("test_weeks must be >= 1.")
        if step_weeks < 1:
            raise ValueError("step_weeks must be >= 1.")

        snapshot_total = getattr(dataset, "snapshot_count", None)
        if snapshot_total is None:
            snapshot_total = len(dataset.features)

        folds = []
        train_end = initial_train_weeks  # number of snapshots in train prefix
        fold_idx = 0
        while train_end + test_weeks <= snapshot_total:
            test_end = train_end + test_weeks

            train_dataset = DynamicGraphTemporalSignal(
                edge_indices=dataset.edge_indices[:train_end],
                edge_weights=dataset.edge_weights[:train_end],
                features=dataset.features[:train_end],
                targets=dataset.targets[:train_end],
            )
            test_dataset = DynamicGraphTemporalSignal(
                edge_indices=dataset.edge_indices[train_end:test_end],
                edge_weights=dataset.edge_weights[train_end:test_end],
                features=dataset.features[train_end:test_end],
                targets=dataset.targets[train_end:test_end],
            )

            folds.append(
                {
                    "fold_idx": fold_idx,
                    "train_start": 0,
                    "train_end": train_end,
                    "test_start": train_end,
                    "test_end": test_end,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                }
            )

            fold_idx += 1
            train_end += step_weeks

        return folds


# Backwards compatibility alias.
HospitalizationAdvanPlusDatasetLoader = ConfigurableDatasetLoader
