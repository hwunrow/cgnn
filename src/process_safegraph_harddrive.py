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
parser = argparse.ArgumentParser(description='Process a batch of files.')
parser.add_argument('--source', type=str, default="/burg/apam/users/nhw2114/safegraph/raw/harddrive/", required=False, help='Source directory')
parser.add_argument('--out', type=str, default="/burg/apam/users/nhw2114/safegraph/processed/harddrive/", required=False, help='Output directory')
parser.add_argument('--batchsize', type=int, default=5, required=False, help='Number of files to process in each batch')
parser.add_argument('--batch-index', type=int, default=0, required=False, help='Index of the batch to process')
parser.add_argument('--file-extension', type=str, default="**/*.gz", required=False, help='File extension for glob pattern')
args = parser.parse_args()

source = args.source
output_dir = args.out
batchsize = args.batchsize
batch_index = args.batch_index

# Load file paths
files = glob.glob(f"{source}/{args.file_extension}", recursive=True)

# Split files into batches
batch_list = [files[i:i+batchsize] for i in range(0, len(files), batchsize)]

# Load mappings
tract_zip_map = pd.read_csv('/burg/home/nhw2114/repos/cgnn/data/raw/ZIP_TRACT_032020.csv')
tract_zip_map['ZIP'] = tract_zip_map['ZIP'].astype(str)
tract_zip_map['TRACT'] = tract_zip_map['TRACT'].astype(str)

# Ensure the batch index is within range
if batch_index >= len(batch_list):
    print(f'Batch index {batch_index} is out of range.')
    exit(1)

# Function to process each row
def pandas_sum_values_or_zero(x):
    if pd.isna(x["visitor_home_aggregation"]):
        pass
    else:
        row_dict = json.loads(x["visitor_home_aggregation"])
        df = pd.DataFrame(list(row_dict.items()), columns=['tract', 'visitor_home_aggregation'])
        df['postal_code'] = x['postal_code']
        df['date_range_start'] = x['date_range_start']
        df['date_range_end'] = x['date_range_end']
        days = json.loads(x['visits_by_day'])
        for i in range(len(days)):
            df[f'VISITS_DAY_{i}'] = days[i]
        return df

fields = ['date_range_start', 'date_range_end', 'postal_code', 'visitor_home_aggregation', 'visits_by_day']

def process_batch(i):
    write_head = True
    # Select the batch of files to process
    files_to_process = batch_list[i]
    li = []
    print('reading files')
    for file in tqdm(files_to_process, file=sys.stdout):
        print(file)
        df = pd.read_csv(file, usecols=fields)
        # drop na rows
        na_rows = sum(df.isna().sum(axis=1) > 0)
        nrows = df.shape[0]
        print(f"{na_rows} ({na_rows / nrows * 100 :.2f}%) rows out of {nrows} have NAs")
        df = df.dropna()
        with open(f'{output_dir}/logs/nan_log.csv', 'a', newline='') as csvfile:
            fieldnames = ['file', 'na_rows', 'nrows']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if i == 0 and write_head:
                writer.writeheader()
                write_head = False
            writer.writerow({'file': file, 'na_rows': na_rows, 'nrows': nrows})
        # drop empty strings
        df = df.loc[df["visitor_home_aggregation"] != '{}']
        df = df.loc[df["visits_by_day"] != ""]
        df = df.loc[df["visitor_home_aggregation"] != ""]
        li.append(df)
    batch_df = pd.concat(li, axis=0, ignore_index=True)

    print('unloading json visitor_home_aggregation')
    tqdm.pandas(desc="json unloaded", file=sys.stdout)
    df_list = batch_df.progress_apply(pandas_sum_values_or_zero, axis=1, raw=False, result_type="reduce")
    print('concatentating data frames')
    start_time = time.time()
    sum_df = pd.concat(df_list.tolist())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.concat took {elapsed_time:.4f} seconds")

    print('merging zip code info')
    start_time = time.time()
    sum_df = sum_df.merge(tract_zip_map[['TRACT', 'ZIP']], left_on='tract', right_on='TRACT', how='left')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.merge took {elapsed_time:.4f} seconds")
    print('percent rows with no tract', sum(sum_df.TRACT.isna()) / sum_df.shape[0])

    sum_df.drop(['tract', 'TRACT'], axis=1, inplace=True)
    sum_df.rename(columns={'postal_code': 'zip_dest', 'ZIP': 'zip_orig'}, inplace=True)

    print('pd.groupby')
    start_time = time.time()
    sum_df = sum_df.groupby(['date_range_start', 'date_range_end', 'zip_orig', 'zip_dest']).sum().reset_index()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.groupby took {elapsed_time:.4f} seconds")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'batch_{i}.csv')
    print(f'saving batch {i}')
    sum_df.to_csv(output_file, index=False)

    return sum_df

# Process the batch
process_batch(batch_index)
