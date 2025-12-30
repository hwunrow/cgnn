"""
This script processes advan data for the years 2020 through 2025. It reads in batch CSV files,
aggregates it by date range and CBSA codes, and saves the aggregated data to CSV files.
"""

import pandas as pd
import glob

# advan data processed in batches using `process_advan_api.py`
path = (
    lambda year: f"/Users/hwunrow/Documents/GitHub/cgnn/data/raw/advan_plus/{year}/batch*.csv"
)
OUT_DIR = "/Users/hwunrow/Documents/GitHub/cgnn/data/raw/advan_plus/"


def process_year(path, fname):
    """
    Processes advan data for a given year, merges it with ZIP to CBSA mappings, aggregates it by
    date range and FIPS codes, and saves the result to a CSV file.

    Args:
        path (str): The file path pattern to the advan CSV files for the year.
        fname (str): The output file name (without extension) for the aggregated data.
    """
    print(f"Reading in CSVs from {path}")
    files = glob.glob(path)

    df = pd.concat([pd.read_csv(file, dtype={"cbsa_orig": str, "cbsa_dest": str}) for file in files])
    df = df.groupby(
        ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
    )[["visitor_home_aggregation"]].sum().reset_index()
    print(df.shape)

    df.to_csv(f"{OUT_DIR}/{fname}.csv", index=False)

    return df


def process_all_years(df_list, fname):
    """
    Processes advan data for a all years, merges it with ZIP to FIPS mappings, aggregates it by
    date range and FIPS codes, and saves the result to a CSV file.

    Args:
        df_list (list): List of data frames to concat.
        fname (str): The output file name (without extension) for the aggregated data.
    """
    df_combined = pd.concat(df_list)

    df = df.groupby(
        ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
    )[["visitor_home_aggregation"]].sum().reset_index()

    df.to_csv(f"{OUT_DIR}/{fname}.csv", index=False)


if __name__ == "__main__":
    df_list = []
    for year in range(2020, 2026):
        print(f"Processing {year} data")
        df = process_year(path(year), f"{year}_api_us")
        df_list.append(df)

    process_all_years(df_list, "all_api_us")

    print("Done")
