"""Dataset exploration helper script for SCSE3040 MLOps Mini Project (Group G184).

Loads dataset paths from params.yaml and prints raw summary statistics strictly
without subjective interpretation or pre-judging data problems.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_config(params_path: str = "params.yaml") -> dict:
    """Load configuration dictionary safely from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_exploration():
    """Run all 9 raw summary checks on train.csv."""
    config = load_config("params.yaml")
    train_path = config["data"]["raw_train_path"]
    holdout_path = config["data"]["raw_holdout_path"]

    df = pd.read_csv(train_path)

    print("=" * 70)
    print("SUMMARY 1: Shape, Dtypes, and Null Counts")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    summary_1 = pd.DataFrame({
        "dtype": df.dtypes,
        "null_count": df.isnull().sum()
    })
    print(summary_1)

    print("\n" + "=" * 70)
    print("SUMMARY 2: df.describe() for Numeric Columns")
    print("=" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe().T)

    print("\n" + "=" * 70)
    print("SUMMARY 3: Describe of Each Numeric Column Grouped by Tehsil")
    print("=" * 70)
    if "tehsil" in df.columns:
        for col in numeric_cols:
            print(f"\n--- Feature: {col} (Grouped by tehsil) ---")
            print(df.groupby("tehsil")[col].describe())

    print("\n" + "=" * 70)
    print("SUMMARY 4: value_counts().head(10) of Each Numeric Column")
    print("=" * 70)
    for col in numeric_cols:
        print(f"\n--- Feature: {col} ---")
        print(df[col].value_counts().head(10))

    print("\n" + "=" * 70)
    print("SUMMARY 5: value_counts() of Each Text Column (repr-printed)")
    print("=" * 70)
    text_cols = [c for c in ["seed_variety", "tehsil"] if c in df.columns]
    for col in text_cols:
        print(f"\n--- Text Column: {col} ---")
        vc = df[col].value_counts(dropna=False)
        for val, count in vc.items():
            print(f"{repr(val):<25} : {count}")

    print("\n" + "=" * 70)
    print("SUMMARY 6: Number of Fully Duplicated Rows")
    print("=" * 70)
    dup_count = df.duplicated().sum()
    print(f"Fully duplicated rows: {dup_count}")

    print("\n" + "=" * 70)
    print("SUMMARY 7: Columns in train.csv vs holdout_features.csv")
    print("=" * 70)
    if Path(holdout_path).exists():
        df_holdout = pd.read_csv(holdout_path)
        train_cols = set(df.columns)
        holdout_cols = set(df_holdout.columns)
        in_train_not_holdout = sorted(list(train_cols - holdout_cols))
        in_holdout_not_train = sorted(list(holdout_cols - train_cols))
        print(f"Train columns count:   {len(train_cols)}")
        print(f"Holdout columns count: {len(holdout_cols)}")
        print(f"Columns in train but NOT holdout: {in_train_not_holdout}")
        print(f"Columns in holdout but NOT train: {in_holdout_not_train}")
    else:
        print(f"Holdout file not found at: {holdout_path}")

    print("\n" + "=" * 70)
    print("SUMMARY 8: Highest 30 Values of Target, Sorted")
    print("=" * 70)
    target_col = config["data"].get("target_col", "yield_qtl_per_acre")
    if target_col in df.columns:
        top_30 = df[target_col].sort_values(ascending=False).head(30)
        print(top_30.reset_index(drop=True))
    else:
        print(f"Target column '{target_col}' not found in train.csv")

    print("\n" + "=" * 70)
    print("SUMMARY 9: Longest Run of Consecutive Identical Values (Sorted by Date)")
    print("=" * 70)
    if "date" in df.columns:
        df_sorted = df.sort_values("date").reset_index(drop=True)
        for col in numeric_cols:
            # Group consecutive identical values
            is_new_run = (df_sorted[col] != df_sorted[col].shift()).cumsum()
            run_sizes = df_sorted.groupby(is_new_run).size()
            max_len = run_sizes.max()
            max_run_id = run_sizes.idxmax()
            run_subset = df_sorted[is_new_run == max_run_id]

            run_val = run_subset[col].iloc[0]
            start_d = run_subset["date"].iloc[0]
            end_d = run_subset["date"].iloc[-1]
            print(f"Feature: {col:<20} | Max run: {max_len:<4} | Value: {run_val:<8} | Start: {start_d} | End: {end_d}")
    else:
        print("Column 'date' not found for chronological sorting")
    print("=" * 70)


if __name__ == "__main__":
    run_exploration()
