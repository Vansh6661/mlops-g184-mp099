"""Unit tests for wheat yield model artifact and prediction pipeline (Group G184, Problem MP099)."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_synthetic_data():
    """Create small synthetic feature DataFrame matching the required input schema."""
    return pd.DataFrame({
        "rainfall_mm": [650.0, 450.0, 800.0],
        "fertilizer_kg": [80.0, 95.0, 70.0],
        "sowing_delay_days": [10, 5, 20],
        "irrigation_count": [4, 6, 3],
        "soil_ph": [6.8, 7.2, 6.4],
        "farm_acres": [8.5, 12.0, 5.0],
        "seed_variety": ["PBW-343", "DBW-187", "HD-2967"],
        "tehsil": ["Tehsil North", "Tehsil South", "Tehsil West"],
    })


def test_saved_model_artifact_exists():
    """Verify that models/model.joblib exists and can be loaded."""
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "models/model.joblib artifact is missing!"
    pipeline = joblib.load(model_path)
    assert hasattr(pipeline, "predict"), "Loaded model must have a 'predict' method."


def test_model_prediction_shape_and_validity(sample_synthetic_data):
    """Verify that model predictions match input length and are positive realistic yields."""
    pipeline = joblib.load("models/model.joblib")
    predictions = pipeline.predict(sample_synthetic_data)

    assert len(predictions) == len(sample_synthetic_data)
    assert isinstance(predictions, np.ndarray)
    assert np.all(predictions > 0), "Predicted wheat yield per acre must be positive."


def test_model_handles_unseen_categories():
    """Verify that OneHotEncoder with handle_unknown='ignore' does not crash on unseen values."""
    pipeline = joblib.load("models/model.joblib")
    unseen_data = pd.DataFrame({
        "rainfall_mm": [500.0],
        "fertilizer_kg": [60.0],
        "sowing_delay_days": [12],
        "irrigation_count": [4],
        "soil_ph": [6.5],
        "farm_acres": [7.0],
        "seed_variety": ["UNSEEN_VARIETY"],
        "tehsil": ["Tehsil Unknown"],
    })
    preds = pipeline.predict(unseen_data)
    assert len(preds) == 1
    assert preds[0] > 0
