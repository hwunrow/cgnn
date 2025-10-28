import argparse
import pandas as pd
import json
from tqdm import tqdm
import time
import glob
import sys
import os
import csv

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process a batch of files.")
parser.add_argument(
    "--source",
    type=str,
    default="/Users/hwunrow/Documents/GitHub/cgnn/data/raw/advan/dewey-downloads/weekly-patterns/",
    required=False,
    help="Source directory",
)
parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Output directory",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=500,
    required=False,
    help="Number of files to process in each batch",
)
parser.add_argument(
    "--batch-index",
    type=int,
    default=0,
    required=False,
    help="Index of the batch to process",
)
parser.add_argument(
    "--file-extension",
    type=str,
    default="*.gz",
    required=False,
    help="File extension for glob pattern",
)
args = parser.parse_args()

source = args.source
output_dir = args.out
batchsize = args.batchsize
batch_index = args.batch_index

log_dir = os.path.join(output_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Load file paths
files = sorted(glob.glob(f"{source}/{args.file_extension}", recursive=True))

fields = [
    "date_range_start",
    "date_range_end",
    "postal_code",
    "iso_country_code",
    "visitor_home_aggregation",
]

# Split files into batches
batch_list = [files[i : i + batchsize] for i in range(0, len(files), batchsize)]

# Load mappings - read ZIP and TRACT as strings to preserve leading zeros
tract_zip_map = pd.read_csv(
    "/Users/hwunrow/Documents/GitHub/cgnn/data/raw/ZIP_TRACT_032020.csv",
    dtype={"ZIP": str, "TRACT": str},
)

# Ensure the batch index is within range
if batch_index >= len(batch_list):
    print(f"Batch index {batch_index} is out of range.")
    exit(1)


def process_visitor_data_vectorized(batch_df):
    """
    Process visitor home aggregation data using itertuples for better performance.

    Args:
        batch_df: DataFrame with visitor_home_aggregation column

    Returns:
        DataFrame with tract-level visitor aggregation data
    """
    if len(batch_df) == 0:
        return pd.DataFrame()

    # Parse JSON data using itertuples for better performance
    all_data = []

    for row in tqdm(
        batch_df.itertuples(index=False),
        total=len(batch_df),
        desc="Processing visitor data",
    ):
        try:
            # Parse visitor home aggregation
            visitor_dict = json.loads(row.visitor_home_aggregation)

            # Create complete records for each tract using extend for better performance
            records = [
                {
                    "tract": tract,
                    "visitor_home_aggregation": visitor_count,
                    "postal_code": row.postal_code,
                    "date_range_start": row.date_range_start,
                    "date_range_end": row.date_range_end,
                }
                for tract, visitor_count in visitor_dict.items()
            ]
            all_data.extend(records)

        except (json.JSONDecodeError, KeyError) as e:
            # Only print errors occasionally to avoid spam
            if len(all_data) % 1000 == 0:
                print(f"Error processing row: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    # Create DataFrame efficiently - no merge needed
    result_df = pd.DataFrame(all_data)

    return result_df


def process_batch_optimized(i):
    """
    Process a batch of files using optimized vectorized JSON processing.

    Args:
        i: Batch index to process

    Returns:
        DataFrame with aggregated visitor data by zip codes
    """
    write_head = True
    # Select the batch of files to process
    files_to_process = batch_list[i]
    li = []
    print("reading files")
    for file in tqdm(files_to_process, file=sys.stdout):
        print(file)
        df = pd.read_csv(file, usecols=fields, dtype={"postal_code": str})
        # drop na rows
        na_rows = sum(df.isna().sum(axis=1) > 0)
        nrows = df.shape[0]
        print(f"{na_rows} ({na_rows / nrows * 100 :.2f}%) rows out of {nrows} have NAs")
        df = df.dropna()
        with open(
            f"{output_dir}/logs/nan_log_batch{i}.csv", "a", newline=""
        ) as csvfile:
            fieldnames = ["file", "na_rows", "nrows"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if i == 0 and write_head:
                writer.writeheader()
                write_head = False
            writer.writerow({"file": file, "na_rows": na_rows, "nrows": nrows})
        # remove canada rows
        df = df.loc[df["iso_country_code"] != "CA"]
        # drop empty strings
        df = df.loc[df["visitor_home_aggregation"] != "{}"]
        df = df.loc[df["visitor_home_aggregation"] != ""]
        li.append(df)
    batch_df = pd.concat(li, axis=0, ignore_index=True)

    print("unloading json visitor_home_aggregation")
    start_time = time.time()
    # Use vectorized approach instead of progress_apply
    sum_df = process_visitor_data_vectorized(batch_df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Vectorized JSON processing took {elapsed_time:.4f} seconds")

    print("merging zip code info")
    start_time = time.time()
    sum_df = sum_df.merge(
        tract_zip_map[["TRACT", "ZIP"]], left_on="tract", right_on="TRACT", how="left"
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.merge took {elapsed_time:.4f} seconds")
    print("percent rows with no tract", sum(sum_df.TRACT.isna()) / sum_df.shape[0])

    sum_df.drop(["tract", "TRACT"], axis=1, inplace=True)
    sum_df.rename(columns={"postal_code": "zip_dest", "ZIP": "zip_orig"}, inplace=True)

    print("pd.groupby")
    start_time = time.time()
    sum_df = (
        sum_df.groupby(["date_range_start", "date_range_end", "zip_orig", "zip_dest"])
        .sum()
        .reset_index()
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.groupby took {elapsed_time:.4f} seconds")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"batch_{i}.csv")
    print(f"saving batch {i} (optimized)")
    sum_df.to_csv(output_file, index=False)

    return sum_df


# Process the batch
process_batch_optimized(batch_index)
