"""Unit tests for wheat yield data cleaning pipeline (Group G184, Problem MP099)."""

import pandas as pd
from src.wheat_yield.cleaning import clean_data


def test_drop_duplicates():
    """Verify that exact duplicate rows are properly removed."""
    df_raw = pd.DataFrame({
        "record_id": ["REC_1", "REC_1"],
        "date": ["2026-01-10", "2026-01-10"],
        "rainfall_mm": [600.0, 600.0],
        "fertilizer_kg": [80.0, 80.0],
        "sowing_delay_days": [10, 10],
        "irrigation_count": [4, 4],
        "soil_ph": [6.5, 6.5],
        "farm_acres": [5.0, 5.0],
        "seed_variety": ["PBW-343", "PBW-343"],
        "tehsil": ["Tehsil North", "Tehsil North"],
        "yield_qtl_per_acre": [18.0, 18.0],
    })

    cleaned = clean_data(df_raw)
    assert len(cleaned) == 1


def test_unit_mixup_correction():
    """Verify that Tehsil South rainfall is multiplied by 10 (cm to mm)."""
    df_raw = pd.DataFrame({
        "record_id": ["REC_1", "REC_2"],
        "date": ["2026-01-10", "2026-01-11"],
        "rainfall_mm": [70.0, 700.0],
        "fertilizer_kg": [80.0, 80.0],
        "sowing_delay_days": [10, 10],
        "irrigation_count": [4, 4],
        "soil_ph": [6.5, 6.5],
        "farm_acres": [5.0, 5.0],
        "seed_variety": ["PBW-343", "PBW-343"],
        "tehsil": ["Tehsil South", "Tehsil North"],
        "yield_qtl_per_acre": [18.0, 18.0],
    })

    cleaned = clean_data(df_raw)
    # Tehsil South should be 70.0 * 10 = 700.0
    assert cleaned.loc[cleaned["tehsil"] == "Tehsil South", "rainfall_mm"].iloc[0] == 700.0
    # Tehsil North should stay unchanged at 700.0
    assert cleaned.loc[cleaned["tehsil"] == "Tehsil North", "rainfall_mm"].iloc[0] == 700.0


def test_seed_variety_normalization():
    """Verify that noisy seed variety representations collapse into canonical names."""
    df_raw = pd.DataFrame({
        "record_id": ["REC_1", "REC_2", "REC_3", "REC_4"],
        "date": ["2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13"],
        "rainfall_mm": [500.0, 500.0, 500.0, 500.0],
        "fertilizer_kg": [80.0, 80.0, 80.0, 80.0],
        "sowing_delay_days": [5, 5, 5, 5],
        "irrigation_count": [3, 3, 3, 3],
        "soil_ph": [7.0, 7.0, 7.0, 7.0],
        "farm_acres": [4.0, 4.0, 4.0, 4.0],
        "seed_variety": ["PBW.", " HD-2967", "Dbw-187", "PBW-343 "],
        "tehsil": ["Tehsil North", "Tehsil North", "Tehsil West", "Tehsil West"],
        "yield_qtl_per_acre": [15.0, 16.0, 17.0, 18.0],
    })

    cleaned = clean_data(df_raw)
    assert set(cleaned["seed_variety"].unique()) == {"PBW-343", "HD-2967", "DBW-187"}


def test_stuck_sensor_removal():
    """Verify that frozen irrigation sensor rows between 2026-03-25 and 2026-04-14 are dropped."""
    df_raw = pd.DataFrame({
        "record_id": ["REC_1", "REC_2"],
        "date": ["2026-03-28", "2026-05-01"],
        "rainfall_mm": [600.0, 600.0],
        "fertilizer_kg": [80.0, 80.0],
        "sowing_delay_days": [10, 10],
        "irrigation_count": [7, 7],
        "soil_ph": [6.5, 6.5],
        "farm_acres": [5.0, 5.0],
        "seed_variety": ["PBW-343", "PBW-343"],
        "tehsil": ["Tehsil North", "Tehsil North"],
        "yield_qtl_per_acre": [18.0, 18.0],
    })

    cleaned = clean_data(df_raw)
    # The record from 2026-03-28 should be dropped; only 2026-05-01 remains
    assert len(cleaned) == 1
    assert cleaned["date"].iloc[0] == "2026-05-01"