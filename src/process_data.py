import pathlib
import pandas as pd
import torch
from utils import BOROUGH_FIPS_MAP, getDateRange
from torch_geometric.data import Data

START_DATE = "02/29/2020"
END_DATE = "05/30/2020"
TRAIN_SPLIT_IDX = 120
TIME_WINDOW_SIZE = 7
DS_LABEL = "test_gen"


def create_node_key():
    dates = getDateRange(START_DATE, END_DATE)
    fips_list = list(BOROUGH_FIPS_MAP.values())

    node_dict = dict()

    curr_idx = 0
    for f in fips_list:
        for d in dates:
            key_str = f"{f}-{d.strftime('%Y-%m-%d')}"
            node_dict[key_str] = curr_idx
            curr_idx += 1

    return node_dict


def process_mobility_report():
    DTYPE = {
        "census_fips_code": "Int64",
        "date": "str",
    }
    mobility_report_df = pd.read_csv(
        "../data/raw/2020_US_Region_Mobility_Report.csv", dtype=DTYPE
    )
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


def process_case_death_data():
    death_df = pd.read_csv(
        "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/deaths-by-day.csv"
    )
    case_df = pd.read_csv(
        "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/cases-by-day.csv"
    )

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
        death_subset_df["date_of_interest"].astype("str")
        + "-"
        + death_subset_df["FIPS"].astype(str)
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
        death_subset_df.groupby(["FIPS"])["DEATH_COUNT"].diff().fillna(0)
    )

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
        case_subset_df["date_of_interest"].astype("str")
        + "-"
        + case_subset_df["FIPS"].astype(str)
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
        case_subset_df.groupby(["FIPS"])["CASE_COUNT"].diff().fillna(0)
    )

    deltaT = pd.Timedelta(value=1, unit="D")
    dates = getDateRange(START_DATE, END_DATE)

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
