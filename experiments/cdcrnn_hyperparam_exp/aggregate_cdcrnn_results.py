import argparse
import json
import os

import pandas as pd


def load_metrics_from_dir(root_dir: str):
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith("_metrics.json"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    rows.append(data)
                except Exception as e:
                    print(f"Failed to read {fpath}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CDCRNN sweep metrics into one CSV and print best configs per objective.")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing *_metrics.json files (e.g. nb/experiment_plots_20260210_135402).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Path to write aggregated CSV (default: <root-dir>/cdcrnn_metrics_aggregated.csv).",
    )
    args = parser.parse_args()

    df = load_metrics_from_dir(args.root_dir)
    if df.empty:
        print(f"No metrics found under {args.root_dir}")
        return

    if args.out_csv is None:
        args.out_csv = os.path.join(args.root_dir, "cdcrnn_metrics_aggregated.csv")

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote aggregated metrics to {args.out_csv}")

    if "objective" not in df.columns or "test_loss" not in df.columns:
        print("Columns 'objective' or 'test_loss' missing; cannot compute best per objective.")
        return

    print("\nBest configs per objective (lowest test_loss):")
    for obj, group in df.groupby("objective"):
        valid = group[pd.notnull(group["test_loss"])]
        if valid.empty:
            print(f"  {obj}: no valid rows")
            continue
        best = valid.loc[valid["test_loss"].idxmin()]
        print(
            f"  {obj}: "
            f"transformation={best.get('transformation')}, "
            f"target={best.get('target')}, "
            f"horizon={best.get('horizon')}, "
            f"test_loss={best.get('test_loss')}"
        )


if __name__ == "__main__":
    main()

