"""
This script processes advan data for the years 2020 through 2025. It reads in CSV files containing
ZIP to FIPS mappings and advan data, merges the data, aggregates it by date range and FIPS codes,
and saves the aggregated data to CSV files.

Functions:
    process_year(path, fname): Processes advan data for a given year, merges it with ZIP to FIPS
    mappings, aggregates it, and saves the result to a CSV file.
"""

import polars as pl
import polars.selectors as cs

from cgnn.process_xwalk import get_zip_cbsa_map

# advan data processed in batches using `process_advan_api.py`
path = (
    lambda year: f"/Users/hwunrow/Documents/GitHub/cgnn/data/raw/advan/{year}/batch*.csv"
)
OUT_DIR = "/Users/hwunrow/Documents/GitHub/cgnn/data/raw/advan/"

zip_cbsa_map = get_zip_cbsa_map()
# Convert to Polars and ensure string dtypes; make Lazy for joins
zip_cbsa_pl = (
    pl.from_pandas(zip_cbsa_map)
    .select([pl.col("ZIP").cast(pl.String), pl.col("CBSA").cast(pl.String)])
    .lazy()
)


def process_year(path, fname):
    """
    Processes advan data for a given year, merges it with ZIP to CBSA mappings, aggregates it by
    date range and FIPS codes, and saves the result to a CSV file.

    Args:
        path (str): The file path pattern to the advan CSV files for the year.
        fname (str): The output file name (without extension) for the aggregated data.
    """
    print(f"Reading in CSVs from {path}")
    df = pl.scan_csv(
        path, schema_overrides={"zip_orig": pl.String, "zip_dest": pl.String}
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

    # Map ZIP -> CBSA for origin and destination using left joins
    df = df.join(
        zip_cbsa_pl.rename({"ZIP": "zip_orig", "CBSA": "cbsa_orig"}),
        on="zip_orig",
        how="left",
    )
    df = df.join(
        zip_cbsa_pl.rename({"ZIP": "zip_dest", "CBSA": "cbsa_dest"}),
        on="zip_dest",
        how="left",
    )

    # Aggregate by CBSA pairs (keeping strings)
    df_cbsa = df.group_by(
        ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
    ).agg([pl.sum("visitor_home_aggregation")])

    df_cbsa = df_cbsa.filter(
        pl.col("cbsa_orig") != "99999",
        pl.col("cbsa_dest") != "99999",
    )

    df_cbsa.sink_csv(f"{OUT_DIR}/{fname}.csv")

    return df_cbsa


def process_all_years(df_list, fname):
    """
    Processes advan data for a all years, merges it with ZIP to FIPS mappings, aggregates it by
    date range and FIPS codes, and saves the result to a CSV file.

    Args:
        df_list (list): List of data frames to concat.
        fname (str): The output file name (without extension) for the aggregated data.
    """
    df_combined = pl.concat(df_list)

    df_out = df_combined.group_by(
        ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
    ).agg([pl.sum("visitor_home_aggregation")])
    df_out = df_out.filter(
        pl.col("cbsa_orig") != "99999",
        pl.col("cbsa_dest") != "99999",
    )

    df_out.sink_csv(f"{OUT_DIR}/{fname}.csv")

    return df_out


df_list = []
for year in range(2020, 2021):
    print(f"Processing {year} data")
    df = process_year(path(year), f"{year}_api_us")
    df_list.append(df)

print("Done")
