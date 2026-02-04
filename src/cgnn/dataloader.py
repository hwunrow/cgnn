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
      - If `feature_transform="diff"`, each block is replaced by its first difference
        (x_t - x_{t-1}), which can help avoid the model converging to a persistence
        predictor (y_{t+1} = y_t). Note: with diff, ``snapshot.x`` no longer contains
        levels; training code that uses ``snapshot.x`` as current levels (e.g. to form
        delta targets) must be updated (e.g. predict next-step level and use MSE, or
        pass current levels separately).
    - **Targets**:
      - `data_source="hospital"`: next time step of `hosp_col`
      - `data_source="case"`: next time step of `CASE_COUNT_7DAY_AVG` (or `CASE_DELTA`
        if `predict_delta=True`)
      - Can be overridden via config `target_col`
    - **Edges**: spatial edges between CBSAs based on `mobility_source`
      (`advan`, `advan_plus`, or `safegraph`)
    - **Edge weights**: `visitor_home_aggregation` from the chosen mobility data
    
    Example:
        >>> from cgnn.dataloader import ConfigurableDatasetLoader
        >>> from torch_geometric_temporal.signal import temporal_signal_split
        >>> 
        >>> # Initialize loader with 3 historical features (total 4 features per node)
        >>> loader = ConfigurableDatasetLoader(
        ...     cbsa_list=None, 
        ...     cfg=None,
        ...     num_historical_features=3
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
        num_historical_features=0,
        feature_transform="none",
    ):
        """
        Initialize the dataloader.
        
        Args:
            cbsa_list (list, optional): List of CBSA codes to include. If None, uses all available.
            cfg (DictConfig, optional): Hydra config object for configuration.
            num_historical_features (int, optional): Number of previous dates to include as features.
                Default is 0 (only current date). If K > 0, features will have shape [num_nodes, K+1]
                where column 0 is current date and columns 1..K are previous dates (most recent to oldest).
            feature_transform (str, optional): How to transform node features. "none" (default) uses
                raw values. "diff" uses first differences (x_t - x_{t-1}) per feature. "log_diff" applies
                log1p before differencing: log(x_t + 1) - log(x_{t-1} + 1), which compresses large values
                and represents percentage change. Uses signed log to handle negatives.
        """
        self.cbsa_list = cbsa_list
        self.cfg = cfg
        self.normalize_edge_weights = normalize_edge_weights
        self.edge_weight_transform = edge_weight_transform
        self.edge_weight_normalization = edge_weight_normalization
        self.transform_xy = transform_xy
        self.xy_transform = xy_transform
        self.predict_delta = predict_delta
        self.num_historical_features = num_historical_features
        self.feature_transform = str(feature_transform).lower() if feature_transform else "none"
        if self.feature_transform not in ("none", "diff", "log_diff"):
            raise ValueError(
                f"Unsupported feature_transform='{self.feature_transform}'. "
                "Expected 'none', 'diff', or 'log_diff'."
            )

        # Get config values or defaults
        data_cfg = _resolve_data_cfg(cfg)

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
                self.target_col = "CASE_DELTA" if self.predict_delta else "CASE_COUNT_7DAY_AVG"

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
                - target: numpy array of shape [num_nodes]
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
        
        if date_idx < len(self.dates) - 1:
            next_date = self.dates[date_idx + 1]
            next_date_str = pd.to_datetime(next_date).strftime("%Y-%m-%d")
            next_df = self.data_df[
                self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == next_date_str
            ]
        else:
            # Last date - use zeros for target
            next_df = pd.DataFrame({"CBSA": self.cbsa_list, self.target_col: [0.0] * len(self.cbsa_list)})

        # Current date dataframe (for features, and for delta targets if requested)
        curr_df = self.data_df[
            self.data_df[self.date_col].dt.strftime("%Y-%m-%d") == date_str
        ]

        # Targets (next time step)
        next_target_dict = dict(zip(next_df["CBSA"], next_df[self.target_col]))
        if self.predict_delta and self.data_source == "hospital":
            # For hospital, delta target means next - current for the hospital column.
            curr_target_dict = dict(zip(curr_df["CBSA"], curr_df[self.target_col]))
            targets = np.array(
                [next_target_dict.get(cbsa, 0.0) - curr_target_dict.get(cbsa, 0.0) for cbsa in self.cbsa_list],
                dtype=np.float32,
            )
        else:
            targets = np.array([next_target_dict.get(cbsa, 0.0) for cbsa in self.cbsa_list], dtype=np.float32)

        # Features: concatenate current + K previous blocks (each block has len(feature_cols) features)
        # When feature_transform=="diff" or "log_diff", we need one extra prior block to compute first differences.
        num_blocks = self.num_historical_features + 1
        if self.feature_transform in ("diff", "log_diff"):
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

        if self.feature_transform == "diff":
            # Replace each block by first difference: block_k becomes (block_k - block_{k+1}).
            # When no prior exists, use zeros (so first difference is zero at the boundary).
            num_output_blocks = self.num_historical_features + 1
            diff_blocks = []
            for k in range(num_output_blocks):
                curr_block = blocks[k]
                prev_block = blocks[k + 1]  # we built one extra block
                diff_blocks.append(curr_block - prev_block)
            blocks = diff_blocks
        elif self.feature_transform == "log_diff":
            # Apply log1p BEFORE differencing: log(x_t + 1) - log(x_{t-1} + 1)
            # This represents percentage change and compresses large values.
            # Use signed log to handle negative values: sign(x) * log1p(|x|)
            num_output_blocks = self.num_historical_features + 1
            diff_blocks = []
            for k in range(num_output_blocks):
                curr_block = blocks[k]
                prev_block = blocks[k + 1]  # we built one extra block
                # Signed log1p: preserves sign, applies log1p to absolute value
                curr_log = np.sign(curr_block) * np.log1p(np.abs(curr_block))
                prev_log = np.sign(prev_block) * np.log1p(np.abs(prev_block))
                diff_blocks.append(curr_log - prev_log)
            blocks = diff_blocks

        # blocks are [current, prev1, prev2, ...] (increasing k)
        features = np.concatenate(blocks, axis=1)  # [num_nodes, (K+1)*num_features]

        if self.transform_xy:
            if self.xy_transform == "log1p":
                if (features < 0).any() or (targets < 0).any():
                    raise ValueError(
                        "Cannot apply log1p transform with negative feature/target values. "
                        "Disable transform_xy or use a different transform."
                    )
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


# Backwards compatibility alias.
HospitalizationAdvanPlusDatasetLoader = ConfigurableDatasetLoader
