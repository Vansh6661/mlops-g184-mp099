"""Export champion model from MLflow registry to standalone joblib artifact."""

from pathlib import Path
import joblib
import mlflow.sklearn


def export_champion_model():
    # 1. Connect to local MLflow SQLite registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # 2. Load the champion model pipeline
    model_name = "mp099_wheat_yield"
    model_uri = f"models:/{model_name}@champion"
    print(f"Loading champion model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # 3. Ensure models/ directory exists and dump joblib
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.joblib"

    joblib.dump(model, output_path)
    print(f"Model exported successfully to: {output_path}")


if __name__ == "__main__":
    export_champion_model()