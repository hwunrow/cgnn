# Process Advan Data

## Download from API
```
uvx --from deweypy dewey --api-key akv1_acTbB7zhgC3evEOaq0U7N0_oeUVhXFIHH3w speedy-download fldr_bpyousrmfggrfubk --partition-key-before 2020-12-31 --partition-key-after 2020-01-06 --skip-existing
```

## Process bathces in parallel

```
src/cgnn/process_advan/process_advan_api.py
```

Bash script. Adjust home directory as necessary.

```
#!/bin/bash


# Configuration
SOURCE_DIR="~/cgnn/data/raw/advan/dewey-downloads/weekly-patterns/"
OUT_DIR="~/cgnn/data/raw/advan/2020"
BATCH_START=0
BATCH_SIZE=100
SCRIPT_PATH="src/cgnn/process_advan/process_advan_api.py"
NUM_WORKERS=2 # Number of parallel workers - adjust based on your CPU

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/logs"

# Calculate number of batches using find to handle large file counts
echo "Counting files in source directory..."
NUM_FILES=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.gz" -type f | wc -l | tr -d ' ')
NUM_BATCHES=$(( (NUM_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))

# Process all batches in parallel
seq $BATCH_START $((NUM_BATCHES-1)) | parallel -j "$NUM_WORKERS" --progress \
    python "$SCRIPT_PATH" \
    --out "$OUT_DIR" \
    --batchsize $BATCH_SIZE \
    --batch-index {}
fi
```

### Crosswalk from TRACT TO ZIP
[ZIP_TRACT_032020](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)

### Crosswalk from ZIP TO CBSA
[ZIP_CBSA_122024](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
See
```
src/cgnn/process_xwalk.py
```

### Concat batches
```
src/cgnn/process_advan/process_advan_api.ipynb
```
