"""
This script processes SafeGraph data for the years 2020 and 2021. It reads in CSV files containing
ZIP to FIPS mappings and SafeGraph data, merges the data, aggregates it by date range and FIPS codes,
and saves the aggregated data to CSV files.

Functions:
    process_year(path, fname): Processes SafeGraph data for a given year, merges it with ZIP to FIPS
    mappings, aggregates it, and saves the result to a CSV file.
"""

import polars as pl
import polars.selectors as cs
import pandas as pd

# safragraph data processed in batches using `process_safegraph_harddrive.py`
path_2020 = f"/burg/apam/users/nhw2114/safegraph/processed/harddrive2/2020/batch*.csv"
path_2021 = f"/burg/apam/users/nhw2114/safegraph/processed/harddrive2/2021/batch*.csv"
path_2022 = f"/burg/apam/users/nhw2114/safegraph/processed/harddrive2/2022/batch*.csv"

OUT_DIR = "/burg/apam/users/nhw2114/repos/cgnn/data/raw/mobility/"


def process_year(path, fname):
    """
    Processes SafeGraph data for a given year, merges it with ZIP to FIPS mappings, aggregates it by
    date range and FIPS codes, and saves the result to a CSV file.

    Args:
        path (str): The file path pattern to the SafeGraph CSV files for the year.
        fname (str): The output file name (without extension) for the aggregated data.
    """
    print(f"Reading in CSVs from {path}")
    df = pl.scan_csv(
        path, schema_overrides={"cbsa_orig": pl.String, "cbsa_dest": pl.String}
    )

    # parse datetimes and extract dates
    df = df.with_columns(
        [
            pl.col("date_range_start")
            .str.to_datetime()
            .dt.date()
            .alias("date_range_start"),
            pl.col("date_range_end")
            .str.to_datetime()
            .dt.date()
            .alias("date_range_end"),
        ]
    )

    df_group = df.group_by(
        ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
    ).agg(
        [
            pl.sum("visitor_home_aggregation"),
            pl.sum("VISITS_DAY_0"),
            pl.sum("VISITS_DAY_1"),
            pl.sum("VISITS_DAY_2"),
            pl.sum("VISITS_DAY_3"),
            pl.sum("VISITS_DAY_4"),
            pl.sum("VISITS_DAY_5"),
            pl.sum("VISITS_DAY_6"),
        ]
    )
    df_group.sink_csv(f"{OUT_DIR}/{fname}.csv")

    return df_group


def process_all_years(df_list, fname):
    """
    Processes SafeGraph data for a all years, merges it with ZIP to FIPS mappings, aggregates it by
    date range and FIPS codes, and saves the result to a CSV file.

    Args:
        df_list (list): List of data frames to concat.
        fname (str): The output file name (without extension) for the aggregated data.
    """
    df_combined = pl.concat(df_list)

    df_combined.group_by(
        ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
    ).agg(
        [
            pl.sum("visitor_home_aggregation"),
            pl.sum("VISITS_DAY_0"),
            pl.sum("VISITS_DAY_1"),
            pl.sum("VISITS_DAY_2"),
            pl.sum("VISITS_DAY_3"),
            pl.sum("VISITS_DAY_4"),
            pl.sum("VISITS_DAY_5"),
            pl.sum("VISITS_DAY_6"),
        ]
    )
    df_combined = df_combined.filter(
        pl.col("cbsa_orig") != "99999",
        pl.col("cbsa_dest") != "99999",
    )

    df_combined.sink_csv(f"{OUT_DIR}/{fname}.csv")

    return df_combined


print("Processing 2020 data")
df_2020 = process_year(path_2020, "2020_harddrive_us")

print("Processing 2021 data")
df_2021 = process_year(path_2021, "2021_harddrive_us")

print("Processing 2022 data")
df_2022 = process_year(path_2022, "2022_harddrive_us")

print("Combining all years")
df_combined = process_all_years([df_2020, df_2021, df_2022], "all_harddrive_us")


print("Done")
