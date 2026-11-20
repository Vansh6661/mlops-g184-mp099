"""Data checks skeleton for SCSE3040 MLOps Mini Project (Group G184, Problem MP099).

Contains helper function skeletons for investigating the seven known data quality issues.
Each function raises NotImplementedError and will be implemented by the student.
"""

from typing import Any, Dict, Tuple, Union
import pandas as pd


def find_leaking_column(df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> Dict[str, Any]:
    """Find any feature column that leaks target information or is unavailable at inference time.

    Expected findings.yaml keys:
      present: bool (True if a leaking column exists, False otherwise)
      column: str or None (the leaking column name)
    """
    raise NotImplementedError


def find_unit_mixup(df: pd.DataFrame, group_col: str = "tehsil") -> Dict[str, Any]:
    """Detect if a feature column was recorded in conflicting units by a specific subgroup.

    Expected findings.yaml keys:
      present: bool (True if a unit mixup exists, False otherwise)
      column: str or None (the affected feature column)
      group_value: str or None (the tehsil value affected)
      factor: float or None (conversion factor, wrong value = right value * factor)
    """
    raise NotImplementedError


def find_sentinel_value(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify if impossible sentinel values (-1, -999, 999, 9999) were used for missing entries.

    Expected findings.yaml keys:
      present: bool (True if sentinel values exist, False otherwise)
      column: str or None (the column containing sentinel values)
      value: float/int or None (the sentinel number)
      count: int or None (number of rows with the sentinel)
    """
    raise NotImplementedError


def find_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """Count duplicate records in the training dataset.

    Expected findings.yaml keys:
      present: bool (True if duplicate rows exist, False otherwise)
      count: int or None (number of extra duplicate rows)
    """
    raise NotImplementedError


def find_stuck_sensor(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    """Detect sensor freeze where a numeric column maintains an identical value for consecutive dates.

    Expected findings.yaml keys:
      present: bool (True if a stuck sensor issue exists, False otherwise)
      column: str or None (the stuck sensor column name)
      start_date: str or None (first date with stuck reading, YYYY-MM-DD)
      end_date: str or None (last date with stuck reading, YYYY-MM-DD)
    """
    raise NotImplementedError


def find_target_typos(df: pd.DataFrame, target_col: str = "yield_qtl_per_acre") -> Dict[str, Any]:
    """Detect decimal or typographical errors in the target column.

    Expected findings.yaml keys:
      present: bool (True if target typos exist, False otherwise)
      count: int or None (number of rows with typed errors)
    """
    raise NotImplementedError


def find_spelling_variants(df: pd.DataFrame, text_cols=("seed_variety", "tehsil")) -> Dict[str, Any]:
    """Identify categorical columns with inconsistent casing, whitespace, or abbreviations.

    Expected findings.yaml keys:
      present: bool (True if spelling variants exist, False otherwise)
      column: str or None (the text column with variants)
      true_levels: int or None (count of canonical categories after cleaning)
    """
    raise NotImplementedError
