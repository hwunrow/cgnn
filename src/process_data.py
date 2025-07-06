import numpy as np
import pandas as pd

import glob
import os
import pickle
from tqdm import tqdm
import sys

from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
from torch_geometric.data import Data

# from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

sys.path.append("../utils/")
from utils import get_date_range, get_cbsa_list
from process_xwalk import get_county_cbsa_map

# TODO don't hardcode these values, make them a YAML file that is saved
START_DATE = "02/29/2020"
END_DATE = "12/31/2022"
TRAIN_SPLIT_IDX = 672
TIME_WINDOW_SIZE = 7
TEMPORAL_EDGE_WINDOW_SIZE = 1

SAFEGRAPH_MOBILITY_CUTOFF = 500

RAW_DEATH_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
RAW_CASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
RAW_SAFEGRAPH_FILE = "../data/raw/mobility/all_harddrive_us.csv"
RAW_MOBILITY_REPORT_DIR = "../data/raw/google_mobility_reports/"

POP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv"


def create_edge_indices(dates):
    """
    Creates edge indices for a dynamic graph given a list of dates.

    Args:
        dates (list of datetime): List of dates.

    Returns:
        list of numpy.ndarray: List of edge indices arrays for each snapshot.
    """
    num_nodes = 5
    nodes = np.arange(num_nodes)
    edges_upper = np.tile(nodes, (num_nodes, 1))
    edges_lower = np.transpose(edges_upper)
    edge_indices = np.concatenate(
        (edges_upper.reshape(1, -1), edges_lower.reshape(1, -1)), axis=0
    )
    edge_indices %= num_nodes

    edge_indices_list = [edge_indices for _ in range(len(dates))]

    return edge_indices_list


def create_torch_geometric_data(version, device="cpu", predict_delta=False):
    dates = get_date_range(START_DATE, END_DATE)
    cbsa_list = get_cbsa_list()

    # process data
    death_subset_df, case_subset_df = process_case_death_data()
    node_dict = create_node_key()
    mobility_report_df = process_mobility_report()
    print("creating coo_df")
    coo_df = create_edge_index(node_dict, dates, cbsa_list)
    print("processing safegraph data")
    edge_weights = process_safegraph_data(dates, node_dict, coo_df)
    train_mask, test_mask = create_train_test_mask(node_dict, dates, cbsa_list)
    save_data(
        death_subset_df,
        case_subset_df,
        mobility_report_df,
        coo_df,
        edge_weights,
        train_mask,
        test_mask,
        node_dict,
        version,
    )

    # make everything a tensor
    coo_t = torch.tensor(coo_df.values, dtype=torch.int64)
    coo_t = torch.transpose(coo_t, 0, 1)

    edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)

    x_t = case_subset_df.merge(
        death_subset_df, on=["date_of_interest", "FIPS", "node_key"]
    )
    x_t = x_t.merge(
        mobility_report_df,
        left_on=["date_of_interest", "FIPS"],
        right_on=["date", "FIPS"],
    )
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
        "mobility_pc1_full_dat",
        "mobility_pc2_full_dat",
    ]
    x_t[x_t_cols].to_csv(f"../data/processed/{version}/x_t.csv", index=False)
    x_t = torch.tensor(x_t[x_t_cols].values, dtype=torch.float32)
    if predict_delta:
        y_t = case_subset_df["CASE_DELTA"]
    else:
        y_t = case_subset_df["CASE_DELTA"] + case_subset_df["CASE_COUNT_7DAY_AVG"]
    y_t.to_csv(f"../data/processed/{version}/y_t.csv", index=False)
    y_t = torch.tensor(
        y_t.values,
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
    version,
):
    path = f"../data/processed/{version}/"
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

    Each key is formatted as {CBSA}-{YYYY-MM-DD}. For example, "36061-2020-03-01"
    is the key for node associated with Manhattan on March 1, 2020.
    The index values are orded by date and then CBSA.

    Returns:
        dict: A dictionary mapping node keys to vertex indices.
    """
    dates = get_date_range(START_DATE, END_DATE)
    cbsa_list = get_cbsa_list()

    node_dict = dict()

    curr_idx = 0
    for d in dates:
        for c in cbsa_list:
            key_str = f"{c}-{d.strftime('%Y-%m-%d')}"
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


def get_safegraph_mobility_data():

    mobility_df = pd.read_csv(
        RAW_SAFEGRAPH_FILE, dtype={"cbsa_orig": str, "cbsa_dest": str}
    )
    mobility_df["date_range_start"] = pd.to_datetime(mobility_df["date_range_start"])
    mobility_df = mobility_df.loc[
        (mobility_df["date_range_start"] >= START_DATE)
        & (mobility_df["date_range_start"] <= END_DATE)
    ]
    mobility_df = mobility_df.loc[
        mobility_df["visitor_home_aggregation"] > SAFEGRAPH_MOBILITY_CUTOFF
    ]
    cbsa_list = get_cbsa_list()
    mobility_df = mobility_df.loc[
        (mobility_df["cbsa_orig"].isin(cbsa_list))
        & (mobility_df["cbsa_dest"].isin(cbsa_list))
    ]

    return mobility_df


def create_edge_index(node_dict, dates, cbsa_list):
    """
    Creates edge index DataFrame representing spatial and temporal edges.
    The function first creates spatial edges, connecting all boroughs to each other
    for each date.
    Then, it creates temporal edges, linking each borough to itself for previous days
    in a time window. The time window size is defined by the variable TIME_WINDOW_SIZE.

    Args:
        node_dict (dict): A dictionary mapping node keys to to vertex indices.
        dates (list): A list of datetime objects representing dates.
        cbsa_list (list): A list of CBSA codes for each borough.

    Returns:
        pandas.DataFrame: Each row of the DataFrame corresponds to an edge,
            with two columns representing the source and target node indices.
    """
    coo_list = []

    mobility_df = get_safegraph_mobility_data()

    print("Creating spatial edges...")
    for index, row in tqdm(
        mobility_df.iterrows(), total=mobility_df.shape[0], desc="Spatial Edges"
    ):
        date_str = row["date_range_start"].strftime("%Y-%m-%d")
        cbsa_orig = row["cbsa_orig"]
        cbsa_dest = row["cbsa_dest"]

        u_key = f"{cbsa_orig}-{date_str}"
        v_key = f"{cbsa_dest}-{date_str}"

        # Ensure nodes exist before adding an edge
        if u_key in node_dict and v_key in node_dict:
            u_idx = node_dict[u_key]
            v_idx = node_dict[v_key]
            coo_list.append([u_idx, v_idx])
        else:
            print(
                f"Warning: Missing node key for spatial edge: {u_key} or {v_key}. Skipping."
            )
    print(len(coo_list), "spatial edges")

    print("Creating temporal edges...")
    temp_count = 0
    for base_day_idx in range(0, len(dates) - TEMPORAL_EDGE_WINDOW_SIZE):
        base_day = dates[base_day_idx]
        base_str = base_day.strftime("%Y-%m-%d")
        for future_day in dates[
            base_day_idx + 1 : base_day_idx + TEMPORAL_EDGE_WINDOW_SIZE + 1
        ]:
            future_str = future_day.strftime("%Y-%m-%d")

            # iterate over each county fips
            for f in cbsa_list:

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
    Processes Google Community Mobility Reports data for all counties.

    Reads the raw mobility report data from RAW_MOBILITY_REPORT_DIR, converts the 'date'
    column to datetime format, and extracts data for all counties counties within the
    specified date range (START_DATE to END_DATE).

    Imputes missing values then performs PCA and returns df with first two components.

    Returns:
        pandas.DataFrame: Columns include FIPS code, date, and mobility indicators.
    """
    DTYPE = {
        "census_fips_code": "str",
        "date": "str",
    }
    files = glob.glob(RAW_MOBILITY_REPORT_DIR + "/*.csv")
    li = []
    for file in files:
        df = pd.read_csv(file, dtype=DTYPE)
        df["date"] = pd.to_datetime(df["date"])
        li.append(df)

    mobility_report_df = pd.concat(li, axis=0, ignore_index=True)

    subset_cols = [
        "sub_region_1",
        "iso_3166_2_code",
        "census_fips_code",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        # "parks_percent_change_from_baseline",  # not used in PCA
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]

    county_mobility_report_df = mobility_report_df.loc[
        ~(mobility_report_df["census_fips_code"].isna())
        & (START_DATE <= mobility_report_df["date"])
        & (mobility_report_df["date"] <= END_DATE),
        subset_cols,
    ]
    county_mobility_report_df.rename(columns={"census_fips_code": "FIPS"}, inplace=True)

    county_cbsa_map = get_county_cbsa_map()
    county_mobility_report_df = county_mobility_report_df.merge(
        county_cbsa_map, left_on="FIPS", right_on="COUNTY", how="left"
    )

    # fill in NaN rows so each FIPS has a complete set of dates
    all_dates = pd.date_range(
        start=county_mobility_report_df["date"].min(),
        end=county_mobility_report_df["date"].max(),
    )
    all_fips = county_mobility_report_df["FIPS"].unique()
    all_combinations = pd.MultiIndex.from_product(
        [all_fips, all_dates], names=["FIPS", "date"]
    ).to_frame(index=False)
    county_mobility_report_df = all_combinations.merge(
        county_mobility_report_df, on=["FIPS", "date"], how="left"
    )
    county_mobility_report_df = county_mobility_report_df.sort_values(
        by=["FIPS", "date"]
    ).reset_index(drop=True)

    # 7-day moving average
    percent_change_columns = [
        col for col in county_mobility_report_df.columns if "percent_change" in col
    ]
    # for col in percent_change_columns:
    #     county_mobility_report_df[f"{col}_avg"] = (
    #         county_mobility_report_df.groupby("FIPS")
    #         .rolling(window=7, min_periods=1)[col]
    #         .mean()
    #         .reset_index(level=0, drop=True)
    #     )

    # Impute missing values using IterativeImputer
    # avg_percent_change_columns = [f"{col}_avg" for col in percent_change_columns]
    print("imputing using columns", percent_change_columns)
    impute_df = county_mobility_report_df.drop(
        columns=["sub_region_1", "iso_3166_2_code"]
    ).loc[:, percent_change_columns]
    df_copy = impute_df.copy()
    missing_mask = df_copy.isna()

    imputer = IterativeImputer(max_iter=100, random_state=1994)
    imputed_values = imputer.fit_transform(df_copy)
    imputed_df = pd.DataFrame(imputed_values, columns=df_copy.columns)

    # PCA to account for correlation
    df = imputed_df[percent_change_columns].dropna()

    df_centered = df - df.mean()
    assert np.all(np.abs(df_centered.mean()) < 1e-10)

    pca = PCA()
    pca.fit(df_centered)

    perc_explained = pca.explained_variance_ratio_[0]
    print(
        f"Percent of variation explained with first component: {perc_explained * 100} %"
    )
    print(pca.explained_variance_ratio_)

    # Force sign to be positive for workplace mobility
    sgn_fix = -1 if pca.components_[0, 3] < 0 else 1

    # Calculate the first principal component
    pc1_train = sgn_fix * df_centered.values.dot(pca.components_[0])

    # Calculate the second principal component
    sgn_fix = -1 if pca.components_[1, 3] < 0 else 1
    pc2_train = sgn_fix * df_centered.values.dot(pca.components_[1])

    # Add the first and second principal component to the original DataFrame
    df["mobility_pc1_full_dat"] = pc1_train
    df["mobility_pc2_full_dat"] = pc2_train

    # Ensure the absolute values match
    assert np.all(
        np.abs(np.abs(pc1_train) - np.abs(pca.transform(df_centered)[:, 0])) < 1e-10
    )

    county_mobility_report_df_pca = county_mobility_report_df.join(
        df[["mobility_pc1_full_dat", "mobility_pc2_full_dat"]]
    )

    # temporal aggregation (just Mondays)
    county_mobility_report_df_pca = county_mobility_report_df_pca[
        county_mobility_report_df_pca["date"].dt.weekday == 0  #  Monday=0
    ].reset_index(drop=True)

    # spatial aggregation (county-weighted average)
    county_mobility_report_df_pca = county_mobility_report_df_pca.loc[
        county_mobility_report_df_pca["CBSA"] != 99999
    ]
    fields = [
        "SUMLEV",
        "REGION",
        "DIVISION",
        "STATE",
        "COUNTY",
        "STNAME",
        "CTYNAME",
        "POPESTIMATE2021",
        "POPESTIMATE2022",
    ]
    converters = {
        col: str
        for col in [
            "SUMLEV",
            "REGION",
            "DIVISION",
            "STATE",
            "COUNTY",
            "STNAME",
            "CTYNAME",
        ]
    }
    print("Reading population data from", POP_URL)
    pop_df = pd.read_csv(
        POP_URL, usecols=fields, converters=converters, encoding="ISO-8859-1"
    )
    pop_df = pop_df.loc[pop_df["SUMLEV"] == "050"]
    pop_df["FIPS"] = pop_df["STATE"] + pop_df["COUNTY"]

    # fix connecticut
    # https://developer.ap.org/ap-elections-api/docs/CT_FIPS_Codes_forPlanningRegions.htm
    ct_fips_mapping = {
        "09190": "09001",
        "09110": "09003",
        "09190": "09005",
        "09130": "09007",
        "09140": "09009",
        "09180": "09011",
        "09110": "09013",
        "09150": "09015",
        "09120": "09001",
        "09160": "09005",
        "09170": "09009",
    }
    pop_df = pop_df.replace({"FIPS": ct_fips_mapping})
    pop_df = pop_df.groupby("FIPS")["POPESTIMATE2021"].sum().reset_index()

    assert pop_df.FIPS.nunique() == pop_df.shape[0]

    county_mobility_report_df_pca = county_mobility_report_df_pca.merge(
        pop_df[["FIPS", "POPESTIMATE2021"]], on="FIPS", how="left"
    )

    # Group by date and CBSA, compute population-weighted average for PCA columns and percent change columns
    pca_columns = ["mobility_pc1_full_dat", "mobility_pc2_full_dat"]
    agg_columns = pca_columns + percent_change_columns

    def weighted_avg(x):
        weights = x["POPESTIMATE2021"]
        return np.average(x[agg_columns], weights=weights, axis=0)

    county_mobility_report_df_pca = (
        county_mobility_report_df_pca.groupby(["date", "CBSA"])
        .apply(lambda x: pd.Series(weighted_avg(x), index=agg_columns))
        .reset_index()
    )

    return county_mobility_report_df_pca


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
    mobility_df = get_safegraph_mobility_data()

    # Create a lookup dictionary for fast access
    mobility_df["date_str"] = mobility_df["date_range_start"].dt.strftime("%Y-%m-%d")
    mobility_lookup = {}
    for _, row in mobility_df.iterrows():
        key = (row["date_str"], row["cbsa_orig"], row["cbsa_dest"])
        mobility_lookup[key] = row["visitor_home_aggregation"]

    # Create reverse lookup for node indices to keys
    node_keys = list(node_dict.keys())

    coo_array = coo_df.values
    orig_keys = [node_keys[idx] for idx in coo_array[:, 0]]
    dest_keys = [node_keys[idx] for idx in coo_array[:, 1]]

    # Split keys into components
    orig_components = [key.split("-", maxsplit=1) for key in orig_keys]
    dest_components = [key.split("-", maxsplit=1) for key in dest_keys]

    edge_weights = []

    print("Processing edge weights...")
    for i in tqdm(range(len(coo_array))):
        orig_cbsa, orig_date = orig_components[i]
        dest_cbsa, dest_date = dest_components[i]

        if orig_date != dest_date:
            # temporal edge with no edge weight
            edge_weights.append(1)
        else:
            lookup_key = (orig_date, orig_cbsa, dest_cbsa)
            ew = mobility_lookup.get(lookup_key)

            if ew is not None:
                edge_weights.append(ew)
            else:
                print(
                    f"No edge weight for {orig_date}, orig:{orig_cbsa}, dest:{dest_cbsa}"
                )

    return edge_weights


def process_case_death_data():
    """
    Processes case and death data for US CBSAs.

    Reads raw case and death data from RAW_CASE_URL and RAW_DEATH_URL, converts the
    'date_of_interest' column to datetime format, and extracts relevant subsets of data
    within the specified date range (START_DATE to END_DATE).

    Returns:
        tuple: A tuple containing two DataFrames:
            - death_subset_df: A DataFrame of processed death data, including CBSA code,
              date, death counts, 7-day average death counts, and delta death counts.
            - case_subset_df: A DataFrame of processed case data, including CBSA code,
              date, case counts, 7-day average case counts, and delta case counts.

    Note:
        The function computes delta values for case and death counts as the change in the
        next day's 7-day average.
        It also computes previous case counts for each day within the time window.
    """
    death_df = pd.read_csv(RAW_DEATH_URL, dtype={"UID": str})
    # case_df = pd.read_csv(RAW_CASE_URL, dtype={"UID": str})

    # # fill in FIPS code for using the last 5 digits of the UID code
    death_df["FIPS"] = death_df["UID"].astype(str).str[-5:]
    # case_df["FIPS"] = case_df["UID"].astype(str).str[-5:]

    # # melt from wide to long
    death_df = pd.melt(
        death_df,
        id_vars=death_df.columns[:12],
        var_name="date",
        value_name="DEATH_COUNT",
    )
    # case_df = pd.melt(
    #     case_df, id_vars=case_df.columns[:11], var_name="date", value_name="CASE_COUNT"
    # )

    # # filter between START_DATE and END_DATE
    death_df["date"] = pd.to_datetime(death_df["date"], format="%m/%d/%y")
    # case_df["date"] = pd.to_datetime(case_df["date"], format="%m/%d/%y")

    death_df = death_df.loc[
        (
            pd.to_datetime(START_DATE) - pd.Timedelta(days=TIME_WINDOW_SIZE + 1)
            <= death_df["date"]
        )
        & (death_df["date"] <= END_DATE)
    ]
    # case_df = case_df.loc[
    #     (
    #         pd.to_datetime(START_DATE) - pd.Timedelta(days=TIME_WINDOW_SIZE + 1)
    #         <= case_df["date"]
    #     )
    #     & (case_df["date"] <= END_DATE)
    # ]

    # # fix since the counts are reported as cummulative
    death_df.sort_values(by=["FIPS", "date"], inplace=True)
    # case_df.sort_values(by=["FIPS", "date"], inplace=True)

    death_df["DEATH_COUNT"] = death_df.groupby("FIPS")["DEATH_COUNT"].diff().fillna(0)
    # case_df["CASE_COUNT"] = case_df.groupby("FIPS")["CASE_COUNT"].diff().fillna(0)

    # ## ADHOC FIXES
    # def _apply_neg_ad_hoc_fixes(df, fixes):
    #     """
    #     Applies ad-hoc fixes with large negative values in case/ death counts.

    #     Args:
    #         df (pd.DataFrame): The DataFrame to apply the fixes to.
    #         fixes (list of dict): A list of dictionaries, where each dictionary defines a fix.
    #             Each dictionary should have the following keys:
    #             - 'fips': The FIPS code (string).
    #             - 'date_to_zero': The date where the CASE_COUNT should be set to 0 (string).
    #             - 'date_to_add': The date where the negative value should be added (string).

    #     Returns:
    #         pd.DataFrame: The DataFrame with the ad-hoc fixes applied.
    #     """

    #     for fix in fixes:
    #         fips = fix["fips"]
    #         date_to_zero = fix["date_to_zero"]
    #         date_to_add = fix["date_to_add"]

    #         try:
    #             neg_val = df.loc[
    #                 (df["FIPS"] == fips) & (df["date"] == date_to_zero),
    #                 "CASE_COUNT",
    #             ].values[0]
    #             df.loc[
    #                 (df["FIPS"] == fips) & (df["date"] == date_to_zero),
    #                 "CASE_COUNT",
    #             ] = 0
    #             df.loc[
    #                 (df["FIPS"] == fips) & (df["date"] == date_to_add),
    #                 "CASE_COUNT",
    #             ] += neg_val
    #         except IndexError:
    #             print(
    #                 f"Warning: FIPS {fips}, date {date_to_zero} or {date_to_add} not found. Skipping fix."
    #             )

    #     return df

    # case_fixes = [
    #     {"fips": "12099", "date_to_zero": "2022-08-17", "date_to_add": "2022-08-13"},
    #     {"fips": "48061", "date_to_zero": "2021-08-22", "date_to_add": "2021-08-21"},
    #     {"fips": "06029", "date_to_zero": "2023-01-12", "date_to_add": "2023-01-13"},
    #     {"fips": "06077", "date_to_zero": "2020-12-23", "date_to_add": "2020-12-24"},
    #     {"fips": "01073", "date_to_zero": "2022-01-26", "date_to_add": "2022-01-25"},
    #     {"fips": "31055", "date_to_zero": "2022-05-11", "date_to_add": "2022-02-28"},
    #     {"fips": "34041", "date_to_zero": "2022-07-18", "date_to_add": "2022-07-19"},
    # ]
    # case_df = _apply_neg_ad_hoc_fixes(case_df, case_fixes)

    ##################################################################################
    ## READ in Teresa's data created from compiledata.m
    ## TODO: replace this with the ad-hoc fixes coded here
    ##################################################################################
    case_df = pd.read_csv("../data/processed/teresa_case_df.csv", dtype={"FIPS": str})
    case_df["FIPS"] = case_df["FIPS"].str.replace("'", "")
    case_df = pd.melt(
        case_df, id_vars=case_df.columns[:5], var_name="date", value_name="CASE_COUNT"
    )
    case_df["date"] = case_df["date"].str.replace("'", "")
    case_df.drop(columns=["fipsnum"], inplace=True)
    case_df["date"] = pd.to_datetime(case_df["date"], format="%m/%d/%Y")
    case_df = case_df.loc[
        (
            pd.to_datetime(START_DATE) - pd.Timedelta(days=TIME_WINDOW_SIZE + 1)
            <= case_df["date"]
        )
        & (case_df["date"] <= END_DATE)
    ]
    ##################################################################################
    ##################################################################################

    # merge in CBSA info and group by
    county_cbsa_map = get_county_cbsa_map()
    case_df = case_df.merge(
        county_cbsa_map, left_on="FIPS", right_on="COUNTY", how="inner"
    )
    death_df = death_df.merge(
        county_cbsa_map, left_on="FIPS", right_on="COUNTY", how="inner"
    )

    case_df = case_df.groupby(["CBSA", "date"])["CASE_COUNT"].sum().reset_index()
    death_df = death_df.groupby(["CBSA", "date"])["DEATH_COUNT"].sum().reset_index()

    death_df = death_df.loc[death_df["CBSA"] != 99999]
    case_df = case_df.loc[case_df["CBSA"] != 99999]

    # group by week and aggregate
    death_df["week"] = death_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    case_df["week"] = case_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    death_df = death_df.groupby(["CBSA", "week"], as_index=False).agg(
        {"DEATH_COUNT": "sum"}
    )
    death_df = death_df.rename(columns={"week": "date"})

    case_df = case_df.groupby(["CBSA", "week"], as_index=False).agg(
        {"CASE_COUNT": "sum"}
    )
    case_df = case_df.rename(columns={"week": "date"})

    death_df["node_key"] = (
        death_df["CBSA"].astype(str) + "-" + death_df["date"].astype("str")
    )
    case_df["node_key"] = (
        case_df["CBSA"].astype(str) + "-" + case_df["date"].astype("str")
    )

    # # fix since the counts are reported as cummulative
    death_df.sort_values(by=["CBSA", "date"], inplace=True)
    case_df.sort_values(by=["CBSA", "date"], inplace=True)

    death_df["DEATH_COUNT"] = death_df.groupby("CBSA")["DEATH_COUNT"].diff().fillna(0)
    case_df["CASE_COUNT"] = case_df.groupby("CBSA")["CASE_COUNT"].diff().fillna(0)

    # clip at 0 since there are negative cases and deaths
    print(
        "rows with neg cases:",
        sum(case_df["CASE_COUNT"] < 0),
        "out of",
        case_df.shape[0],
    )
    print(
        "rows with neg deaths:",
        sum(death_df["DEATH_COUNT"] < 0),
        "out of",
        death_df.shape[0],
    )
    case_df["CASE_COUNT"] = case_df["CASE_COUNT"].clip(lower=0)
    death_df["DEATH_COUNT"] = death_df["DEATH_COUNT"].clip(lower=0)

    lagged_case_cols = []
    lagged_death_cols = []
    for dd in range(TIME_WINDOW_SIZE - 1):
        lagged_case_col = case_df.groupby("CBSA")["CASE_COUNT"].shift(dd + 1).fillna(0)
        lagged_case_cols.append(lagged_case_col.rename(f"CASE_COUNT_PREV_{dd}"))
        lagged_death_col = (
            death_df.groupby("CBSA")["DEATH_COUNT"].shift(dd + 1).fillna(0)
        )
        lagged_death_cols.append(lagged_death_col.rename(f"DEATH_COUNT_PREV_{dd}"))
    case_df = pd.concat([case_df] + lagged_case_cols, axis=1)
    death_df = pd.concat([death_df] + lagged_death_cols, axis=1)

    # correct the 7 day average so that it's not rounding to integers
    case_df[f"CASE_COUNT_{TIME_WINDOW_SIZE}DAY_AVG"] = case_df[
        [f"CASE_COUNT_PREV_{i}" for i in range(TIME_WINDOW_SIZE - 1)] + ["CASE_COUNT"]
    ].mean(axis=1)
    death_df[f"DEATH_COUNT_{TIME_WINDOW_SIZE}DAY_AVG"] = death_df[
        [f"DEATH_COUNT_PREV_{i}" for i in range(TIME_WINDOW_SIZE - 1)] + ["DEATH_COUNT"]
    ].mean(axis=1)

    # compute deltas
    case_df = case_df.sort_values(by=["date", "CBSA"])
    case_df["CASE_DELTA"] = (
        case_df.groupby(["CBSA"])[f"CASE_COUNT_{TIME_WINDOW_SIZE}DAY_AVG"]
        .diff(-1)
        .fillna(0)
    )
    case_df["CASE_DELTA"] = case_df["CASE_DELTA"] * -1

    death_df = death_df.sort_values(by=["date", "CBSA"])
    death_df["DEATH_DELTA"] = (
        death_df.groupby(["CBSA"])[f"DEATH_COUNT_{TIME_WINDOW_SIZE}DAY_AVG"]
        .diff(-1)
        .fillna(0)
    )
    death_df["DEATH_DELTA"] = death_df["DEATH_DELTA"] * -1

    death_subset_cols = [
        "date",
        "CBSA",
        "node_key",
        "DEATH_DELTA",
        f"DEATH_COUNT_{TIME_WINDOW_SIZE}DAY_AVG",
        "DEATH_COUNT",
        "DEATH_COUNT_PREV_0",
        "DEATH_COUNT_PREV_1",
        "DEATH_COUNT_PREV_2",
        "DEATH_COUNT_PREV_3",
        "DEATH_COUNT_PREV_4",
        "DEATH_COUNT_PREV_5",
    ]
    case_subset_cols = [
        "date",
        "CBSA",
        "node_key",
        "CASE_DELTA",
        f"CASE_COUNT_{TIME_WINDOW_SIZE}DAY_AVG",
        "CASE_COUNT",
        "CASE_COUNT_PREV_0",
        "CASE_COUNT_PREV_1",
        "CASE_COUNT_PREV_2",
        "CASE_COUNT_PREV_3",
        "CASE_COUNT_PREV_4",
        "CASE_COUNT_PREV_5",
    ]
    death_df = death_df.loc[
        (START_DATE <= death_df["date"]) & (death_df["date"] <= END_DATE),
        death_subset_cols,
    ]
    case_df = case_df.loc[
        (START_DATE <= case_df["date"]) & (case_df["date"] <= END_DATE),
        case_subset_cols,
    ]

    death_df.reset_index(inplace=True, drop=True)
    case_df.reset_index(inplace=True, drop=True)

    return death_df, case_df
