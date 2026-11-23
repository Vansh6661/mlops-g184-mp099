"""Model training and MLflow tracking script for SCSE3040 MLOps Mini Project."""

import hashlib
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.wheat_yield.cleaning import clean_data


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of raw data file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_models():
    # 1. Set MLflow tracking URI to SQLite database in repo root
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "wheat_yield_mp099"
    mlflow.set_experiment(experiment_name)

    # 2. Load and clean training data
    raw_path = "data/raw/train.csv"
    data_sha256 = compute_file_sha256(raw_path)
    df_raw = pd.read_csv(raw_path)
    df_clean = clean_data(df_raw)

    target_col = "yield_qtl_per_acre"
    feature_cols = [
        "rainfall_mm",
        "fertilizer_kg",
        "sowing_delay_days",
        "irrigation_count",
        "soil_ph",
        "farm_acres",
        "seed_variety",
        "tehsil",
    ]

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    numeric_features = [
        "rainfall_mm",
        "fertilizer_kg",
        "sowing_delay_days",
        "irrigation_count",
        "soil_ph",
        "farm_acres",
    ]
    categorical_features = ["seed_variety", "tehsil"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # 3. Try 3 different model configurations
    configs = [
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 5},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 150, "max_depth": 15, "min_samples_split": 2},
    ]

    best_mae = float("inf")
    best_run_id = None
    registered_model_name = "mp099_wheat_yield"

    for i, cfg in enumerate(configs, start=1):
        with mlflow.start_run(run_name=f"rf_config_{i}") as run:
            # Build full pipeline
            regressor = RandomForestRegressor(random_state=42, **cfg)
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("regressor", regressor)]
            )

            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            mae_val = float(mean_absolute_error(y_val, val_preds))

            # Mandatory logging rules from handbook:
            # 1. Log data_sha256 param
            mlflow.log_param("data_sha256", data_sha256)
            # 2. Log hyperparameters
            mlflow.log_params(cfg)
            # 3. Log metric containing 'mae'
            mlflow.log_metric("mae_val", mae_val)

            # Log model artifact with cloudpickle to avoid untrusted types warning
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"Run {i}: config={cfg} -> mae_val={mae_val:.4f} (run_id: {run.info.run_id})")

            if mae_val < best_mae:
                best_mae = mae_val
                best_run_id = run.info.run_id

    print("\n" + "=" * 60)
    print(f"Best Run ID: {best_run_id} with MAE: {best_mae:.4f}")

    # 4. Register the best model as Version 1 and assign alias 'champion'
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri, name=registered_model_name
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=model_version.version,
    )

    print(f"Registered model '{registered_model_name}' version {model_version.version} as '@champion'")
    print("=" * 60)


if __name__ == "__main__":
    train_models()