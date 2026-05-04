import pandas as pd
import pickle
from cgnn.utils.codebook import TITLE_CBSA_MAP, HHS_REGION_MAP
from datetime import datetime
import math
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_date_range(start, end, day_of_week="MON"):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    dates = pd.date_range(
        start=start_date, end=end_date, freq=f"W-{day_of_week.upper()}"
    )
    return dates


def get_cbsa_list(hhs_region=None):
    all_cbsas = list(TITLE_CBSA_MAP.keys())
    if hhs_region is None:
        return all_cbsas

    df = pd.read_csv(_REPO_ROOT / "data" / "raw" / "list1_2023.csv")
    df = df.iloc[:-3]  # Remove footer rows

    cbsa_states = df.groupby("CBSA Code")["State Name"].apply(set).to_dict()

    cbsa_to_region = {}
    for cbsa, states in cbsa_states.items():
        regions = set()
        for state in states:
            if state in HHS_REGION_MAP:
                regions.add(HHS_REGION_MAP[state])
        if regions:
            # If CBSA spans multiple regions, take the first one
            cbsa_to_region[cbsa] = next(iter(regions))

    filtered_cbsas = [
        cbsa
        for cbsa, region in cbsa_to_region.items()
        if cbsa in TITLE_CBSA_MAP and region == hhs_region
    ]

    return filtered_cbsas


def get_cbsa_info(cbsa_list=None):
    if cbsa_list is None:
        cbsa_list = get_cbsa_list()

    df = pd.read_csv(
        _REPO_ROOT / "data" / "raw" / "list1_2023.csv",
        dtype={"CBSA Code": str},
    )
    df = df.iloc[:-3]  # Remove footer rows

    return df.loc[df["CBSA Code"].isin(cbsa_list)]


def get_node_date(version, idx):
    path = f"../data/processed/{version}/"
    with open(f"{path}/node_dict.pkl", "rb") as f:
        node_dict = pickle.load(f)

    return list(node_dict.keys())[idx].split("-", 1)[1]


def get_node_borough(version, idx):
    path = f"../data/processed/{version}/"
    with open(f"{path}/node_dict.pkl", "rb") as f:
        node_dict = pickle.load(f)

    return list(node_dict.keys())[idx].split("-", 1)[0]


def get_node_pos(version, idx):
    date = get_node_date(version, idx)
    borough = get_node_borough(version, idx)

    borough_positions = {
        "36005": (math.cos(0), math.sin(0)),
        "36047": (math.cos(2 * math.pi / 5), math.sin(2 * math.pi / 5)),
        "36061": (math.cos(4 * math.pi / 5), math.sin(4 * math.pi / 5)),
        "36081": (math.cos(6 * math.pi / 5), math.sin(6 * math.pi / 5)),
        "36085": (math.cos(8 * math.pi / 5), math.sin(8 * math.pi / 5)),
    }

    pos_x, pos_y = borough_positions[borough]

    # Increase pos_x for larger dates
    reference_date = datetime(2020, 2, 29)
    date = datetime.strptime(date, "%Y-%m-%d")
    delta_days = (date - reference_date).days

    pos_x += 2 * delta_days

    return (pos_x, pos_y)


def save_config_to_directory(cfg, directory, filename="config.yaml"):
    """
    Save a Hydra config to a directory.

    Args:
        cfg (DictConfig): The Hydra config object to save.
        directory (str): The directory path where to save the config.
        filename (str): The filename for the config file (default: "config.yaml").

    Returns:
        str: The full path to the saved config file.
    """
    os.makedirs(directory, exist_ok=True)
    config_path = os.path.join(directory, filename)
    OmegaConf.save(cfg, config_path)
    return config_path


def get_node_idx(node_dict, fips, date):
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    key = f"{fips}-{date}"
    return node_dict[key]


def compare_configs(cfg_a, cfg_b):
    """
    Compare two Hydra configs and return keys where they differ.

    Accepts either file paths (str) or DictConfig objects. Nested keys are
    represented in dot notation (e.g. ``"training.learning_rate"``).

    Args:
        cfg_a: Path to a YAML config file or a DictConfig.
        cfg_b: Path to a YAML config file or a DictConfig.

    Returns:
        dict: Mapping of dotted key -> {"a": value_in_a, "b": value_in_b}
              for every key whose value differs between the two configs.
              Keys present in only one config are included with the missing
              side set to ``None``.
    """
    if isinstance(cfg_a, str):
        cfg_a = OmegaConf.load(cfg_a)
    if isinstance(cfg_b, str):
        cfg_b = OmegaConf.load(cfg_b)

    def _flatten(cfg, prefix=""):
        items = {}
        for key, value in cfg.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if OmegaConf.is_config(value):
                items.update(_flatten(value, full_key))
            else:
                items[full_key] = value
        return items

    flat_a = _flatten(cfg_a)
    flat_b = _flatten(cfg_b)

    all_keys = set(flat_a) | set(flat_b)
    diffs = {}
    for key in sorted(all_keys):
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)
        if val_a != val_b:
            diffs[key] = {"a": val_a, "b": val_b}
    return diffs
