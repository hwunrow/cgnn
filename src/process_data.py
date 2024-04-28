import numpy as np
import pandas as pd

import glob
import os
import pickle
from tqdm import tqdm
import sys

import torch
from torch_geometric.data import Data

sys.path.append("../utils/")
from utils import get_date_range, get_fips_list
from codebook import BOROUGH_FIPS_MAP, BOROUGH_FULL_FIPS_DICT

# TODO don't hardcode these values, make them a YAML file that is saved
START_DATE = "02/29/2020"
END_DATE = "05/30/2020"
TRAIN_SPLIT_IDX = 60
TIME_WINDOW_SIZE = 7
VERSION = "gcn"


RAW_DEATH_URL = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/deaths-by-day.csv"
RAW_CASE_URL = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/cases-by-day.csv"
RAW_SAFEGRAPH_DIR = "../data/raw/mobility/"
RAW_MOBILITY_REPORT_FILE = "../data/raw/2020_US_Region_Mobility_Report.csv"


def create_torch_geometric_data(device="cpu", processed_path=None):
    if processed_path:
        death_subset_df = pd.read_csv(f"{processed_path}/death_data.csv")
        case_subset_df = pd.read_csv(f"{processed_path}/case_data.csv")
        nyc_mobility_report_df = pd.read_csv(
            f"{processed_path}/mobility_report_data.csv"
        )
        coo_df = pd.read_csv(f"{processed_path}/coo_edge_index.csv")
        edge_weights = np.load(f"{processed_path}/edge_weights.npy")
        train_mask = np.load(f"{processed_path}/train_mask.npy")
        test_mask = np.load(f"{processed_path}/test_mask.npy")
    else:
        dates = get_date_range(START_DATE, END_DATE)
        fips_list = get_fips_list()

        # process data
        death_subset_df, case_subset_df = process_case_death_data()
        node_dict = create_node_key()
        nyc_mobility_report_df = process_mobility_report()
        coo_df = create_edge_index(node_dict, dates, fips_list)
        edge_weights = process_safegraph_data(dates, node_dict, coo_df)
        train_mask, test_mask = create_train_test_mask(node_dict, dates, fips_list)
        save_data(
            death_subset_df,
            case_subset_df,
            nyc_mobility_report_df,
            coo_df,
            edge_weights,
            train_mask,
            test_mask,
            node_dict,
            VERSION,
        )

    # make everything a tensor
    coo_t = torch.tensor(coo_df.values, dtype=torch.int64)
    coo_t = coo_t.reshape((2, len(coo_df.values)))

    edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)

    x_t = case_subset_df.merge(
        death_subset_df, on=["date_of_interest", "FIPS", "node_key"]
    )
    x_t = x_t.merge(
        nyc_mobility_report_df,
        left_on=["date_of_interest", "FIPS"],
        right_on=["date", "FIPS"],
    )
    # correct the 7 day average so that it's not rounding to integers
    x_t["CASE_COUNT_7DAY_AVG"] = x_t[
        [f"CASE_COUNT_PREV_{i}" for i in range(6)] + ["CASE_COUNT"]
    ].mean(axis=1)
    x_t["DEATH_COUNT_7DAY_AVG"] = x_t[
        [f"DEATH_COUNT_PREV_{i}" for i in range(6)] + ["DEATH_COUNT"]
    ].mean(axis=1)
    x_t_cols = [
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
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    x_t[x_t_cols].to_csv(f"../data/processed/{VERSION}/x_t.csv", index=False)
    x_t = torch.tensor(x_t[x_t_cols].values, dtype=torch.float32)
    y_t = torch.tensor(
        case_subset_df["CASE_DELTA"].values
        + case_subset_df["CASE_COUNT_7DAY_AVG"].values,
        dtype=torch.float32,
    )
    data = Data(
        x=x_t.to(device),
        y=y_t.to(device),
        edge_index=coo_t.to(device),
        edge_weight=edge_weight_tensor.to(device),
    )

    data.train_mask = torch.tensor(np.array(train_mask), dtype=torch.bool).to(device)
    data.test_mask = torch.tensor(np.array(test_mask), dtype=torch.bool).to(device)

    return data


def save_data(
    death_subset_df,
    case_subset_df,
    nyc_mobility_report_df,
    coo_df,
    edge_weights,
    train_mask,
    test_mask,
    node_dict,
    VERSION,
):
    path = f"../data/processed/{VERSION}/"
    os.makedirs(path, exist_ok=True)

    death_subset_df.to_csv(f"{path}/death_data.csv", index=False)
    case_subset_df.to_csv(f"{path}/case_data.csv", index=False)
    nyc_mobility_report_df.to_csv(f"{path}/mobility_report_data.csv", index=False)
    coo_df.to_csv(f"{path}/coo_edge_index.csv", index=False)
    np.save(f"{path}/edge_weights.npy", edge_weights)
    np.save(f"{path}/train_mask.npy", train_mask)
    np.save(f"{path}/test_mask.npy", test_mask)
    with open(f"{path}/node_dict.pkl", "wb") as f:
        pickle.dump(node_dict, f)

    print("Processed data saved to", path)


def create_node_key():
    """
    Creates a dictionary mapping node keys to vertex indices.

    Each key is formatted as {FIPS}-{YYYY-MM-DD}. For example, "36061-2020-03-01"
    is the key for node associated with Manhattan on March 1, 2020.
    The index values are orded by date and then FIPS.

    Returns:
        dict: A dictionary mapping node keys to vertex indices.
    """
    dates = get_date_range(START_DATE, END_DATE)
    fips_list = get_fips_list()

    node_dict = dict()

    curr_idx = 0
    for d in dates:
        for f in fips_list:
            key_str = f"{f}-{d.strftime('%Y-%m-%d')}"
            node_dict[key_str] = curr_idx
            curr_idx += 1

    return node_dict


def create_train_test_mask(node_dict, dates, fips_list):
    """
    Creates train and test masks.
    Train mask includes data up to the TRAIN_SPLIT_IDX'th date.

    Args:
        node_dict (dict): A dictionary mapping node keys to vertex indices.
        dates (list): A list of datetime objects representing dates.
        fips_list (list): A list of FIPS codes for each borough.

    Returns:
        tuple: A tuple containing two lists, of the train mask and the test mask.
            Each list has the same length as the number of nodes,
            with 1s indicating inclusion, and 0s indicating exclusion.
    """
    train_mask = [0 for _ in range(len(node_dict))]
    test_mask = [0 for _ in range(len(node_dict))]

    for i in range(len(dates)):
        for fips in fips_list:
            date_str = dates[i].strftime("%Y-%m-%d")
            key_str = "{}-{}".format(fips, date_str)

            idx = node_dict[key_str]
            if i < TRAIN_SPLIT_IDX:
                train_mask[idx] = 1
            else:
                test_mask[idx] = 1

    return train_mask, test_mask


def create_edge_index(node_dict, dates, fips_list):
    """
    Creates edge index DataFrame representing spatial and temporal edges.
    The function first creates spatial edges, connecting all boroughs to each other
    for each date.
    Then, it creates temporal edges, linking each borough to itself for previous days
    in a time window. The time window size is defined by the variable TIME_WINDOW_SIZE.

    Args:
        node_dict (dict): A dictionary mapping node keys to to vertex indices.
        dates (list): A list of datetime objects representing dates.
        fips_list (list): A list of FIPS codes for each borough.

    Returns:
        pandas.DataFrame: Each row of the DataFrame corresponds to an edge,
            with two columns representing the source and target node indices.
    """
    coo_list = []

    # create spatial edges (all boroughs are connected to each other)
    for d in dates:
        for u in fips_list:
            for v in fips_list:
                u_key = f"{u}-{d.strftime('%Y-%m-%d')}"
                v_key = f"{v}-{d.strftime('%Y-%m-%d')}"
                u_idx = node_dict[u_key]
                v_idx = node_dict[v_key]
                coo_list.append([u_idx, v_idx])
    print(len(coo_list), "spatial edges")

    # create temporal edges
    temp_count = 0
    for base_day_idx in range(0, len(dates) - TIME_WINDOW_SIZE):
        base_day = dates[base_day_idx]
        base_str = base_day.strftime("%Y-%m-%d")
        for future_day in dates[base_day_idx + 1 : base_day_idx + TIME_WINDOW_SIZE + 1]:
            future_str = future_day.strftime("%Y-%m-%d")

            # iterate over each county fips
            for f in fips_list:

                # Need a link from base_day to future_day
                u_key = f"{f}-{base_str}"
                v_key = f"{f}-{future_str}"

                u_idx = node_dict[u_key]
                v_idx = node_dict[v_key]
                # Only add past->future link.
                coo_list.append([u_idx, v_idx])
                temp_count += 1

    print(temp_count, "temporal edges")
    coo_df = pd.DataFrame(coo_list)

    return coo_df


def process_mobility_report():
    """
    Processes Google Community Mobility Reports data for New York City.

    Reads the raw mobility report data from RAW_MOBILITY_REPORT_FILE, converts the 'date'
    column to datetime format, and extracts data for New York City counties within the
    specified date range (START_DATE to END_DATE).

    Returns:
        pandas.DataFrame: Columns include FIPS code, date, and mobility indicators.
    """
    DTYPE = {
        "census_fips_code": "Int64",
        "date": "str",
    }
    mobility_report_df = pd.read_csv(RAW_MOBILITY_REPORT_FILE, dtype=DTYPE)
    mobility_report_df["date"] = pd.to_datetime(mobility_report_df["date"])

    counties = [
        "Bronx County",
        "Kings County",
        "New York County",
        "Queens County",
        "Richmond County",
    ]
    subset_cols = [
        "census_fips_code",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]

    nyc_mobility_report_df = mobility_report_df.loc[
        (mobility_report_df.sub_region_1 == "New York")
        & (mobility_report_df.sub_region_2.isin(counties))
        & (START_DATE <= mobility_report_df["date"])
        & (mobility_report_df["date"] <= END_DATE),
        subset_cols,
    ]

    nyc_mobility_report_df.rename(columns={"census_fips_code": "FIPS"}, inplace=True)

    return nyc_mobility_report_df


def process_safegraph_data(dates, node_dict, coo_df):
    """
    Processes SafeGraph mobility data to extract edge weights for temporal edges.

    Args:
        dates (list): A list of datetime objects representing dates.
        node_dict (dict): A dictionary mapping node keys to vertex indices.
        coo_df (pandas.DataFrame): A DataFrame containing edge indices.

    Returns:
        list: A list containing edge weights for temporal edges.

    Note:
        The function reads SafeGraph mobility data from CSV files in RAW_SAFEGRAPH_DIR.
        It extracts edge weights based on visitor home aggregation for temporal edges,
        with weights representing the number of visitors from the origin borough to the
        destination borough.
        Since the mobility data is only provided on a weekly basis, the function assigns
        edge weights to the next Monday for each date.
    """
    mobility_files = glob.glob(f"{RAW_SAFEGRAPH_DIR}/*.csv")
    mobility_dates = [os.path.basename(f).split("_")[0] for f in mobility_files]
    mobility_dates = pd.to_datetime(mobility_dates)
    mobility_dates = mobility_dates.sort_values()

    day_mobility_dict = dict()

    for d in dates:
        next_sunday = d + pd.offsets.Week(n=0, weekday=0)
        day_key = d.strftime("%Y-%m-%d")
        day_mobility_dict[day_key] = next_sunday.strftime("%Y-%m-%d") + "_mobility.csv"

    node_keys = list(node_dict.keys())

    edge_weights = []

    for row in tqdm(coo_df.iterrows()):
        orig = row[1][0]
        dest = row[1][1]

        orig_key = node_keys[orig]
        dest_key = node_keys[dest]

        orig_fips, orig_date = orig_key.split("-", maxsplit=1)
        dest_fips, dest_date = dest_key.split("-", maxsplit=1)

        if orig_date != dest_date:
            # temportal edge with no edge weight
            edge_weights.append(1)
        else:
            df = pd.read_csv(RAW_SAFEGRAPH_DIR + day_mobility_dict[orig_date])
            ew = df.loc[
                (df.origin == BOROUGH_FULL_FIPS_DICT[orig_fips])
                & (df.destination == BOROUGH_FULL_FIPS_DICT[dest_fips]),
                "visitor_home_aggregation",
            ].values[0]
            edge_weights.append(ew)

    return edge_weights


def process_case_death_data():
    """
    Processes case and death data for New York City boroughs.

    Reads raw case and death data from RAW_CASE_URL and RAW_DEATH_URL, converts the
    'date_of_interest' column to datetime format, and extracts relevant subsets of data
    within the specified date range (START_DATE to END_DATE).

    Returns:
        tuple: A tuple containing two DataFrames:
            - death_subset_df: A DataFrame of processed death data, including FIPS code,
              date, death counts, 7-day average death counts, and delta death counts.
            - case_subset_df: A DataFrame of processed case data, including FIPS code,
              date, case counts, 7-day average case counts, and delta case counts.

    Note:
        The function computes delta values for case and death counts as the change in the
        next day's 7-day average.
        It also computes previous case counts for each day within the time window.
    """
    death_df = pd.read_csv(RAW_DEATH_URL)
    case_df = pd.read_csv(RAW_CASE_URL)

    death_df["date_of_interest"] = pd.to_datetime(death_df["date_of_interest"])
    case_df["date_of_interest"] = pd.to_datetime(case_df["date_of_interest"])

    subset_cols = [
        "date_of_interest",
        "BX_DEATH_COUNT",
        "BX_DEATH_COUNT_7DAY_AVG",
        "BK_DEATH_COUNT",
        "BK_DEATH_COUNT_7DAY_AVG",
        "MN_DEATH_COUNT",
        "MN_DEATH_COUNT_7DAY_AVG",
        "QN_DEATH_COUNT",
        "QN_DEATH_COUNT_7DAY_AVG",
        "SI_DEATH_COUNT",
        "SI_DEATH_COUNT_7DAY_AVG",
    ]
    death_subset_df = death_df.loc[
        (START_DATE <= death_df["date_of_interest"])
        & (death_df["date_of_interest"] <= END_DATE),
        subset_cols,
    ]
    death_subset_df = pd.melt(death_subset_df, id_vars=["date_of_interest"])
    death_subset_df[["borough", "metric"]] = death_subset_df["variable"].str.split(
        "_", n=1, expand=True
    )
    death_subset_df = death_subset_df.pivot(
        index=["date_of_interest", "borough"], columns="metric", values="value"
    ).reset_index()

    death_subset_df["FIPS"] = death_subset_df["borough"].map(BOROUGH_FIPS_MAP)

    death_subset_df["node_key"] = (
        death_subset_df["FIPS"].astype(str)
        + "-"
        + death_subset_df["date_of_interest"].astype("str")
    )
    long_cols = [
        "date_of_interest",
        "FIPS",
        "node_key",
        "DEATH_COUNT",
        "DEATH_COUNT_7DAY_AVG",
    ]
    death_subset_df = death_subset_df[long_cols]

    # compute deltas
    death_subset_df = death_subset_df.sort_values(by=["date_of_interest", "FIPS"])

    death_subset_df["DEATH_DELTA"] = (
        death_subset_df.groupby(["FIPS"])["DEATH_COUNT_7DAY_AVG"].diff(-1).fillna(0)
    )

    death_subset_df["DEATH_DELTA"] = death_subset_df["DEATH_DELTA"] * -1

    subset_cols = [
        "date_of_interest",
        "BX_CASE_COUNT",
        "BX_CASE_COUNT_7DAY_AVG",
        "BK_CASE_COUNT",
        "BK_CASE_COUNT_7DAY_AVG",
        "MN_CASE_COUNT",
        "MN_CASE_COUNT_7DAY_AVG",
        "QN_CASE_COUNT",
        "QN_CASE_COUNT_7DAY_AVG",
        "SI_CASE_COUNT",
        "SI_CASE_COUNT_7DAY_AVG",
    ]
    case_subset_df = case_df.loc[
        (START_DATE <= case_df["date_of_interest"])
        & (case_df["date_of_interest"] <= END_DATE),
        subset_cols,
    ]
    case_subset_df = pd.melt(case_subset_df, id_vars=["date_of_interest"])
    case_subset_df[["borough", "metric"]] = case_subset_df["variable"].str.split(
        "_", n=1, expand=True
    )
    case_subset_df = case_subset_df.pivot(
        index=["date_of_interest", "borough"], columns="metric", values="value"
    ).reset_index()

    case_subset_df["FIPS"] = case_subset_df["borough"].map(BOROUGH_FIPS_MAP)
    case_subset_df["node_key"] = (
        case_subset_df["FIPS"].astype(str)
        + "-"
        + case_subset_df["date_of_interest"].astype("str")
    )
    long_cols = [
        "date_of_interest",
        "FIPS",
        "node_key",
        "CASE_COUNT",
        "CASE_COUNT_7DAY_AVG",
    ]

    case_subset_df = case_subset_df[long_cols]

    # compute deltas
    case_subset_df = case_subset_df.sort_values(by=["date_of_interest", "FIPS"])

    case_subset_df["CASE_DELTA"] = (
        case_subset_df.groupby(["FIPS"])["CASE_COUNT_7DAY_AVG"].diff(-1).fillna(0)
    )

    case_subset_df["CASE_DELTA"] = case_subset_df["CASE_DELTA"] * -1

    deltaT = pd.Timedelta(value=1, unit="D")
    dates = get_date_range(START_DATE, END_DATE)

    for f in list(BOROUGH_FIPS_MAP.values()):
        for d in dates:
            for dd in range(TIME_WINDOW_SIZE - 1):
                prev = pd.to_datetime(d) - deltaT * (dd + 1)

                selection_current = (case_subset_df["FIPS"] == f) & (
                    case_subset_df["date_of_interest"] == d
                )
                selection_prev = (case_subset_df["FIPS"] == f) & (
                    case_subset_df["date_of_interest"] == prev
                )
                if prev < pd.to_datetime(START_DATE):
                    prev_cases = 0
                else:
                    prev_cases = case_subset_df.loc[
                        selection_prev, "CASE_COUNT"
                    ].values[0]
                case_subset_df.loc[selection_current, f"CASE_COUNT_PREV_{dd}"] = (
                    prev_cases
                )

                selection_current = (death_subset_df["FIPS"] == f) & (
                    death_subset_df["date_of_interest"] == d
                )
                selection_prev = (death_subset_df["FIPS"] == f) & (
                    death_subset_df["date_of_interest"] == prev
                )
                if prev < pd.to_datetime(START_DATE):
                    prev_deaths = 0
                else:
                    prev_deaths = case_subset_df.loc[
                        selection_prev, "CASE_COUNT"
                    ].values[0]
                death_subset_df.loc[selection_current, f"DEATH_COUNT_PREV_{dd}"] = (
                    prev_deaths
                )

    death_subset_df.reset_index(inplace=True)
    case_subset_df.reset_index(inplace=True)

    return death_subset_df, case_subset_df
