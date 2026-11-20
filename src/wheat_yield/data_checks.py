"""Data checks skeleton for SCSE3040 MLOps Mini Project (Group G184, Problem MP099).

Contains helper function skeletons for investigating the seven known data quality issues.
Each function raises NotImplementedError and will be implemented by the student.
"""

from typing import Any, Dict, Tuple, Union
import pandas as pd


def find_leaking_column(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target_col: str = "yield_qtl_per_acre",
) -> Dict[str, Any]:
    """Find any feature column that leaks target information or is unavailable at inference time.

    Expected findings.yaml keys:
      present: bool (True if a leaking column exists, False otherwise)
      column: str or None (the leaking column name)
    """
    # Check for features present during training but absent during test/inference time
    train_features = set(df_train.columns) - {target_col, "record_id"}
    holdout_features = set(df_holdout.columns) - {"record_id"}

    extra_features = sorted(list(train_features - holdout_features))
    if extra_features:
        return {
            "present": True,
            "column": extra_features[0],
        }

    # Check for unnatural proxy features with near-perfect correlation to the target
    if target_col in df_train.columns:
        numeric_features = [
            c for c in df_train.select_dtypes(include=["number"]).columns
            if c != target_col
        ]
        for col in numeric_features:
            corr = df_train[col].corr(df_train[target_col])
            if abs(corr) >= 0.98:
                return {
                    "present": True,
                    "column": col,
                }

    return {
        "present": False,
        "column": None,
    }


def find_unit_mixup(df: pd.DataFrame, group_col: str = "tehsil") -> Dict[str, Any]:
    """Detect if a feature column was recorded in conflicting units by a specific subgroup.

    Expected findings.yaml keys:
      present: bool (True if a unit mixup exists, False otherwise)
      column: str or None (the affected feature column)
      group_value: str or None (the tehsil value affected)
      factor: float or None (conversion factor, wrong value = right value * factor)
    """
    if group_col not in df.columns:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    # Inspect numerical feature columns (excluding target)
    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c != "yield_qtl_per_acre"
    ]

    for col in numeric_cols:
        group_means = df.groupby(group_col)[col].mean()
        if len(group_means) < 2:
            continue

        baseline_mean = group_means.median()
        if baseline_mean <= 0:
            continue

        for grp_val, grp_mean in group_means.items():
            ratio = grp_mean / baseline_mean
            # Common unit mixup scaling factors (e.g. cm vs mm = 0.1, or grams vs kg = 1000)
            for factor in [0.01, 0.1, 10.0, 100.0]:
                if 0.8 * factor <= ratio <= 1.2 * factor:
                    return {
                        "present": True,
                        "column": col,
                        "group_value": grp_val,
                        "factor": factor,
                    }

    return {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }


def find_sentinel_value(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify if impossible sentinel values (-1, -999, 999, 9999) were used for missing entries.

    Expected findings.yaml keys:
      present: bool (True if sentinel values exist, False otherwise)
      column: str or None (the column containing sentinel values)
      value: float/int or None (the sentinel number)
      count: int or None (number of rows with the sentinel)
    """
    # Common numeric codes entered in place of missing/blank measurements
    common_sentinels = [-999, -999.0, -99, -99.0, -1, -1.0, 999, 999.0, 9999, 9999.0]
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        for sent_val in common_sentinels:
            matches = int((df[col] == sent_val).sum())
            if matches > 0:
                return {
                    "present": True,
                    "column": col,
                    "value": sent_val,
                    "count": matches,
                }

    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }


def find_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """Count duplicate records in the training dataset.

    Expected findings.yaml keys:
      present: bool (True if duplicate rows exist, False otherwise)
      count: int or None (number of extra duplicate rows)
    """
    # Count rows that are exact duplicates of earlier rows (extra copies to delete)
    extra_count = int(df.duplicated().sum())
    is_present = extra_count > 0

    return {
        "present": is_present,
        "count": extra_count if is_present else None,
    }


def find_stuck_sensor(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    """Detect sensor freeze where a numeric column maintains an identical value for consecutive dates.

    Expected findings.yaml keys:
      present: bool (True if a stuck sensor issue exists, False otherwise)
      column: str or None (the stuck sensor column name)
      start_date: str or None (first date with stuck reading, YYYY-MM-DD)
      end_date: str or None (last date with stuck reading, YYYY-MM-DD)
    """
    # Sort chronologically by date
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    numeric_cols = df_sorted.select_dtypes(include=["number"]).columns

    stuck_col = None
    longest_streak = 0
    stuck_start = None
    stuck_end = None

    for col in numeric_cols:
        # Create a new group ID each time the value changes
        is_change = (df_sorted[col] != df_sorted[col].shift()).cumsum()
        run_sizes = df_sorted.groupby(is_change).size()
        max_run = int(run_sizes.max())

        # An unusually long streak (e.g. 10 or more consecutive rows) indicates a freeze
        if max_run >= 10 and max_run > longest_streak:
            longest_streak = max_run
            stuck_col = col
            run_id = run_sizes.idxmax()
            run_rows = df_sorted[is_change == run_id]
            stuck_start = str(run_rows[date_col].iloc[0])
            stuck_end = str(run_rows[date_col].iloc[-1])

    is_present = stuck_col is not None

    return {
        "present": is_present,
        "column": stuck_col if is_present else None,
        "start_date": stuck_start if is_present else None,
        "end_date": stuck_end if is_present else None,
    }


def find_target_typos(df: pd.DataFrame, target_col: str = "yield_qtl_per_acre") -> Dict[str, Any]:
    """Detect decimal or typographical errors in the target column.

    Expected findings.yaml keys:
      present: bool (True if target typos exist, False otherwise)
      count: int or None (number of rows with typed errors)
    """
    if target_col not in df.columns:
        return {"present": False, "count": None}

    # In wheat farming, typical yield is ~10-35 qtl/acre.
    # Typos entered 10x too large would appear as extreme outliers far above normal range (> 3x median).
    median_val = df[target_col].median()
    typos = df[df[target_col] > 3 * median_val]
    typo_count = int(len(typos))
    is_present = typo_count > 0

    return {
        "present": is_present,
        "count": typo_count if is_present else None,
    }


def find_spelling_variants(df: pd.DataFrame, text_cols=("seed_variety", "tehsil")) -> Dict[str, Any]:
    """Identify categorical columns with inconsistent casing, whitespace, or abbreviations.

    Expected findings.yaml keys:
      present: bool (True if spelling variants exist, False otherwise)
      column: str or None (the text column with variants)
      true_levels: int or None (count of canonical categories after cleaning)
    """
    for col in text_cols:
        if col not in df.columns:
            continue

        raw_levels = int(df[col].nunique())

        # Normalize whitespace and casing, and collapse abbreviation variants (e.g. 'PBW.' -> 'PBW-343')
        normalized = df[col].astype(str).str.strip().str.upper()
        canonical = normalized.replace({
            r"^PBW.*": "PBW-343",
            r"^DBW.*": "DBW-187",
            r"^HD.*": "HD-2967",
        }, regex=True)

        canonical_levels = int(canonical.nunique())

        # If raw variants collapse into fewer true canonical categories, spelling variants are present
        if raw_levels > canonical_levels:
            return {
                "present": True,
                "column": col,
                "true_levels": canonical_levels,
            }

    return {
        "present": False,
        "column": None,
        "true_levels": None,
    }
