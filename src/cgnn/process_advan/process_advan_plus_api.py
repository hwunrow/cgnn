import argparse
import pandas as pd
import json
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import time
import glob
import sys
import os
import csv

from cgnn.process_xwalk import get_zip_cbsa_map

import logging

logger = logging.getLogger(__name__)

class TqdmToLogFile:
    """File-like object that writes tqdm output to both stdout and log file"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.stdout = sys.stdout
        
    def write(self, s):
        # Write to stdout (for console display)
        self.stdout.write(s)
        # Also append to log file
        with open(self.log_file_path, 'a') as f:
            f.write(s)
    
    def flush(self):
        self.stdout.flush()

def process_visitor_data_vectorized(batch_df, log_file=None, chunk_size=100000):
    """
    Process visitor home aggregation data using itertuples for better performance.
    Processes in chunks to avoid memory issues with very large lists.

    Args:
        batch_df: DataFrame with visitor_home_aggregation column
        log_file: Optional path to log file for tqdm output
        chunk_size: Number of records to accumulate before converting to DataFrame

    Returns:
        DataFrame with tract-level visitor aggregation data
    """
    if len(batch_df) == 0:
        return pd.DataFrame()

    # Parse JSON data using itertuples for better performance
    all_data = []
    chunk_dfs = []  # Store DataFrames for each chunk

    # Create file-like object for tqdm if log_file is provided
    tqdm_file = TqdmToLogFile(log_file) if log_file else sys.stdout
    
    with logging_redirect_tqdm():
        for row in tqdm(
            batch_df.itertuples(index=False),
            total=len(batch_df),
            desc="Processing visitor data",
            file=tqdm_file,
        ):
            try:
                # Parse visitor home aggregation
                visitor_dict = json.loads(row.VISITOR_HOME_AGGREGATION)

                # Create complete records for each tract using extend for better performance
                records = [
                    {
                        "tract": tract,
                        "visitor_home_aggregation": visitor_count,
                        "postal_code": row.POSTAL_CODE,
                        "date_range_start": row.DATE_RANGE_START,
                        "date_range_end": row.DATE_RANGE_END,
                    }
                    for tract, visitor_count in visitor_dict.items()
                ]
                all_data.extend(records)
                
                # Process chunk when it reaches chunk_size
                if len(all_data) >= chunk_size:
                    chunk_df = pd.DataFrame(all_data)
                    chunk_dfs.append(chunk_df)
                    all_data = []  # Reset for next chunk
                    del chunk_df  # Free memory

            except (json.JSONDecodeError, KeyError) as e:
                # Only log errors occasionally to avoid spam
                if len(all_data) % 1000 == 0:
                    logger.warning(f"Error processing row: {e}")
                continue

    # Process remaining data
    if all_data:
        chunk_df = pd.DataFrame(all_data)
        chunk_dfs.append(chunk_df)
        del all_data, chunk_df

    # Concatenate all chunks
    if not chunk_dfs:
        return pd.DataFrame()
    
    result_df = pd.concat(chunk_dfs, axis=0, ignore_index=True)
    del chunk_dfs  # Free memory

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
        logger.info(f"{file}")
        df = pd.read_parquet(file, columns=fields)
        # drop na rows
        na_rows = sum(df.isna().sum(axis=1) > 0)
        nrows = df.shape[0]
        logger.info(f"{na_rows} ({na_rows / nrows * 100 :.2f}%) rows out of {nrows} have NAs")
        df = df.dropna()
        with open(
            f"{log_dir}/nan_log_batch{i}.csv", "a", newline=""
        ) as csvfile:
            fieldnames = ["file", "na_rows", "nrows"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if i == 0 and write_head:
                writer.writeheader()
                write_head = False
            writer.writerow({"file": file, "na_rows": na_rows, "nrows": nrows})
        # remove canada rows
        df = df.loc[df["ISO_COUNTRY_CODE"] != "CA"]
        # drop empty strings
        df = df.loc[df["VISITOR_HOME_AGGREGATION"] != "{}"]
        df = df.loc[df["VISITOR_HOME_AGGREGATION"] != ""]
        li.append(df)
    batch_df = pd.concat(li, axis=0, ignore_index=True)
    del li
    
    # print("filter out duplicates")
    # rows_before_filter = len(batch_df)
    # batch_df = batch_df.drop_duplicates()
    # rows_after_filter = len(batch_df)
    # rows_filtered = rows_before_filter - rows_after_filter
    # with open(
    #     f"{log_dir}/duplicate_log_batch{i}.csv", "a", newline=""
    # ) as csvfile:
    #     fieldnames = ["batch_index", "rows_before_filter", "rows_after_filter", "rows_filtered", "filter_percentage"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     if i == 0:
    #         writer.writeheader()
    #     writer.writerow({
    #         "batch_index": i,
    #         "rows_before_filter": rows_before_filter,
    #         "rows_after_filter": rows_after_filter,
    #         "rows_filtered": rows_filtered,
    #         "filter_percentage": (rows_filtered / rows_before_filter * 100) if rows_before_filter > 0 else 0
    #     })
    # print(f"after dropping duplicates: {batch_df.shape[0]} rows")

    logger.info("unloading json visitor_home_aggregation")
    start_time = time.time()
    # Reconstruct log_file path for tqdm output
    log_file = os.path.join(log_dir, f"batch_{i}.log")
    # Use vectorized approach instead of progress_apply
    sum_df = process_visitor_data_vectorized(batch_df, log_file=log_file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Vectorized JSON processing took {elapsed_time:.4f} seconds")
    del batch_df

    logger.info("merging zip code info")
    start_time = time.time()
    sum_df = sum_df.merge(
        tract_zip_map[["TRACT", "ZIP"]], left_on="tract", right_on="TRACT", how="left"
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"pd.merge took {elapsed_time:.4f} seconds")
    pct_no_tract = sum(sum_df.TRACT.isna()) / sum_df.shape[0]
    logger.info(f"percent rows with no tract: {pct_no_tract:.4f}")

    sum_df.drop(["tract", "TRACT"], axis=1, inplace=True)
    sum_df.rename(columns={"postal_code": "zip_dest", "ZIP": "zip_orig"}, inplace=True)

    logger.info("merging CBSA info")
    start_time = time.time()
    sum_df = sum_df.merge(
        zip_cbsa_map.rename(columns={"ZIP": "zip_orig", "CBSA": "cbsa_orig"}),
        on="zip_orig",
        how="left",
    )
    sum_df = sum_df.merge(
        zip_cbsa_map.rename(columns={"ZIP": "zip_dest", "CBSA": "cbsa_dest"}),
        on="zip_dest",
        how="left",
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"pd.merge took {elapsed_time:.4f} seconds")

    logger.info("convert dates to Y-m-d format")
    start_time = time.time()
    sum_df["date_range_start"] = sum_df["date_range_start"].dt.date
    sum_df["date_range_end"] = sum_df["date_range_end"].dt.date
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"convert dates to Y-m-d format took {elapsed_time:.4f} seconds")

    logger.info("pd.groupby")
    start_time = time.time()
    sum_df = (
        sum_df.groupby(
            ["date_range_start", "date_range_end", "cbsa_orig", "cbsa_dest"]
        )[["visitor_home_aggregation"]]
        .sum()
        .reset_index()
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"pd.groupby took {elapsed_time:.4f} seconds")

    # drop 99999 cbsa codes
    sum_df = sum_df.loc[(sum_df["cbsa_orig"] != "99999") & (sum_df["cbsa_dest"] != "99999")]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"batch_{i}.csv")
    logger.info(f"saving batch {i} (optimized)")
    sum_df.to_csv(output_file, index=False)

    return sum_df


# Process the batch
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a batch of files.")
    parser.add_argument(
        "--source",
        type=str,
        default="/Users/hwunrow/Documents/GitHub/cgnn/data/raw/advan_plus/dewey-downloads/weekly-patterns-plus/",
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
        default="*.snappy.parquet",
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

    # Configure logging to write to both file and console
    log_file = os.path.join(log_dir, f"batch_{batch_index}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ],
        force=True  # Override any existing configuration
    )
    logger = logging.getLogger(__name__)

    # Load file paths
    files = sorted(glob.glob(f"{source}/{args.file_extension}", recursive=True))

    fields = [
        "PERSISTENT_ID",
        "DATE_RANGE_START",
        "DATE_RANGE_END",
        "POSTAL_CODE",
        "ISO_COUNTRY_CODE",
        "VISITOR_HOME_AGGREGATION",
    ]

    # Split files into batches
    batch_list = [files[i : i + batchsize] for i in range(0, len(files), batchsize)]

    # Load mappings - read ZIP and TRACT as strings to preserve leading zeros
    tract_zip_map = pd.read_csv(
        "/Users/hwunrow/Documents/GitHub/cgnn/data/raw/ZIP_TRACT_032020.csv",
        dtype={"ZIP": str, "TRACT": str},
    )

    zip_cbsa_map = get_zip_cbsa_map()

    # Ensure the batch index is within range
    if batch_index >= len(batch_list):
        print(f"Batch index {batch_index} is out of range.")
        exit(1)

    process_batch_optimized(batch_index)
