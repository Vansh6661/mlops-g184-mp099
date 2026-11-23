"""Data cleaning pipeline for SCSE3040 MLOps Mini Project (Group G184, Problem MP099)."""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw wheat yield training data based on verified data issues.

    Fixes applied:
    1. Drop duplicate rows (37 extra copies).
    2. Fix unit mixup: multiply Tehsil South rainfall by 10 (convert cm to mm).
    3. Standardize seed_variety text variants into 3 canonical varieties.
    4. Drop stuck-sensor rows for irrigation_count (2026-03-25 to 2026-04-14).
    """
    df = df.copy()

    # 1. Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Fix unit mixup: Tehsil South was recorded in cm, multiply by 10 for mm
    south_mask = df["tehsil"] == "Tehsil South"
    df.loc[south_mask, "rainfall_mm"] = df.loc[south_mask, "rainfall_mm"] * 10.0

    # 3. Normalize seed_variety spelling, casing, whitespace, and abbreviations
    cleaned_variety = df["seed_variety"].astype(str).str.strip().str.upper()
    df["seed_variety"] = cleaned_variety.replace({
        r"^PBW.*": "PBW-343",
        r"^DBW.*": "DBW-187",
        r"^HD.*": "HD-2967",
    }, regex=True)

    # 4. Remove stuck-sensor frozen rows (frozen at 7 from 2026-03-25 to 2026-04-14)
    stuck_mask = (
        (df["date"] >= "2026-03-25")
        & (df["date"] <= "2026-04-14")
        & (df["irrigation_count"] == 7)
    )
    df = df[~stuck_mask].reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/train.csv")
    cleaned_df = clean_data(raw_df)

    print(f"Raw shape:     {raw_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Duplicates:    {cleaned_df.duplicated().sum()}")
    print(f"Seed varieties:{cleaned_df['seed_variety'].nunique()} -> {list(cleaned_df['seed_variety'].unique())}")