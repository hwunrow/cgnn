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
path_2020 = f"/burg/apam/users/nhw2114/safegraph/processed/harddrive/2020/*.csv"
path_2021 = f"/burg/apam/users/nhw2114/safegraph/processed/harddrive/2021/*.csv"

OUT_DIR = "/burg/apam/users/nhw2114/repos/cgnn/data/raw/"
ZIP_CBSA_PATH = "/burg/apam/users/nhw2114/repos/cgnn/data/raw/ZIP_CBSA_122024.csv"


def get_zip_cbsa_map(path):
    """
    Returns a lazy DataFrame mapping ZIP codes to CBSA codes.

    Since ZIP-CBSA is one to many, keep the CBSA with the largest RES_RATIO for each zip.

    Args:
        path (str): Path to the CSV file (currently hardcoded).

    Returns:
        polars.LazyFrame: Lazy DataFrame mapping ZIP codes to CBSA codes.
    """
    zip_cbsa_map = pl.read_csv(
        path,
        schema_overrides={"CBSA": pl.Int64},
    )
    zip_cbsa_map = zip_cbsa_map.sort(["ZIP", "RES_RATIO"], descending=[False, True])
    zip_cbsa_map = zip_cbsa_map.unique(subset="ZIP", keep="first")
    zip_cbsa_map = zip_cbsa_map.select(
        [pl.col("ZIP").cast(pl.String).alias("zip"), pl.col("CBSA")]
    )
    zip_cbsa_map = zip_cbsa_map.lazy()

    return zip_cbsa_map


zip_cbsa_map = get_zip_cbsa_map(ZIP_CBSA_PATH)


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
        path, schema_overrides={"zip_orig": pl.String, "zip_dest": pl.String}
    )

    unique_zips = df.select(pl.col("zip_orig")).unique().collect().to_series().to_list()
    missing_zips = [
        zip_code
        for zip_code in unique_zips
        if zip_code not in zip_cbsa_map.collect().to_series().to_list()
    ]
    print(f"Missing ZIPs: {missing_zips}")

    # merge cbsa codes
    df = df.join(zip_cbsa_map, left_on="zip_orig", right_on="zip", how="inner").rename(
        {"CBSA": "cbsa_orig"}
    )
    df = df.join(zip_cbsa_map, left_on="zip_dest", right_on="zip", how="inner").rename(
        {"CBSA": "cbsa_dest"}
    )
    df = df.drop(["zip_orig", "zip_dest"])

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


print("Processing 2020 data")
process_year(path_2020, "2020_harddrive_us")

print("Processing 2021 data")
process_year(path_2021, "2021_harddrive_us")
print("Done")
