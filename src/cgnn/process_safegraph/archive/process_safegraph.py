import argparse
import pandas as pd
import json
from tqdm import tqdm
import time
import glob
import sys
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process a batch of files.')
parser.add_argument('--source', type=str, required=True, help='Source directory')
parser.add_argument('--out', type=str, required=True, help='Output directory')
parser.add_argument('--batchsize', type=int, required=True, help='Number of files to process in each batch')
parser.add_argument('--batch-index', type=int, required=True, help='Index of the batch to process')
parser.add_argument('--file-extension', type=str, default="*.gz", required=False, help='File extension for glob pattern')
args = parser.parse_args()

source = args.source
output_dir = args.out
batchsize = args.batchsize
batch_index = args.batch_index

# Load file paths
files = glob.glob(f"{source}/{args.file_extension}")

# Split files into batches
batch_list = [files[i:i+batchsize] for i in range(0, len(files), batchsize)]

# Ensure the batch index is within range
if batch_index >= len(batch_list):
    print(f'Batch index {batch_index} is out of range.')
    exit(1)

# Select the batch of files to process
files_to_process = batch_list[batch_index]

# Load mappings
zip_county_map = pd.read_csv('/burg/home/nhw2114/repos/cgnn/data/raw/ZIP_COUNTY_CROSSWALK.csv')
tract_zip_map = pd.read_csv('/burg/home/nhw2114/repos/cgnn/data/raw/ZIP_TRACT_032020.csv')

# Process the mappings
zip_county_map = zip_county_map.sort_values(by=['ZIP', 'TOT_RATIO'], ascending=False)
zip_county_map = zip_county_map.drop_duplicates(subset='ZIP', keep='first')
zip_county_map['ZIP'] = zip_county_map['ZIP'].astype(str)
zip_county_map['COUNTY'] = zip_county_map['COUNTY'].astype(str)
tract_zip_map['ZIP'] = tract_zip_map['ZIP'].astype(str)
tract_zip_map['TRACT'] = tract_zip_map['TRACT'].astype(str)
tract_county_map = zip_county_map.merge(tract_zip_map[['ZIP', 'TRACT']], left_on='ZIP', right_on='ZIP', how='right')

# Function to process each row
def pandas_sum_values_or_zero(x):
    if pd.isna(x["VISITOR_HOME_AGGREGATION"]):
        pass
    else:
        row_dict = json.loads(x["VISITOR_HOME_AGGREGATION"])
        df = pd.DataFrame(list(row_dict.items()), columns=['tract', 'visitor_home_aggregation'])
        df['COUNTY_DEST'] = x['COUNTY']
        df['DATE_RANGE_START'] = x['DATE_RANGE_START']
        df['DATE_RANGE_END'] = x['DATE_RANGE_END']
        days = json.loads(x['VISITS_BY_DAY'])
        for i in range(len(days)):
            df[f'VISITS_DAY_{i}'] = days[i]
        return df

def process_batch(files, i):
    li = []
    print('reading files')
    for file in tqdm(files, file=sys.stdout):
        df = pd.read_csv(file)
        df = df.loc[df["VISITS_BY_DAY"] != ""]
        df = df.loc[df["VISITOR_HOME_AGGREGATION"] != ""]
        df = df.merge(zip_county_map, left_on='POSTAL_CODE', right_on='ZIP', how='left')
        li.append(df)
    batch_df = pd.concat(li, axis=0, ignore_index=True)

    print('unloading json VISITOR_HOME_AGGREGATION')
    tqdm.pandas(desc="json unloaded", file=sys.stdout)
    df_list = batch_df.progress_apply(pandas_sum_values_or_zero, axis=1, raw=False, result_type="reduce")
    
    print('concatentating data frames')
    start_time = time.time()
    sum_df = pd.concat(df_list.tolist())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.concat took {elapsed_time:.4f} seconds")

    print('merging county info')
    start_time = time.time()
    sum_df = sum_df.merge(tract_county_map[['TRACT', 'COUNTY']], left_on='tract', right_on='TRACT', how='left')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.merge took {elapsed_time:.4f} seconds")
    print('percent rows with no tract', sum(sum_df.TRACT.isna()) / sum_df.shape[0])

    sum_df.drop(['tract', 'TRACT'], axis=1, inplace=True)
    sum_df.rename(columns={'COUNTY' : 'COUNTY_ORIG'}, inplace=True)
    print('pd.groupby')
    start_time = time.time()
    sum_df = sum_df.groupby(['DATE_RANGE_START', 'DATE_RANGE_END', 'COUNTY_ORIG', 'COUNTY_DEST']).sum().reset_index()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"pd.groupby took {elapsed_time:.4f} seconds")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'batch_{i}.csv')
    print(f'saving batch {i}')
    sum_df.to_csv(output_file, index=False)

    return sum_df

# Process the batch
process_batch(files_to_process, batch_index)
