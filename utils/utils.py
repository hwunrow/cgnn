import pandas as pd
import pickle
from codebook import BOROUGH_FIPS_MAP
from datetime import datetime
import math


def get_date_range(start, end):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return dates


def get_fips_list():
    fips_list = list(BOROUGH_FIPS_MAP.values())
    return fips_list


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


def get_node_idx(node_dict, fips, date):
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    key = f"{fips}-{date}"
    return node_dict[key]
