# SCSE3040 MLOps Mini Project --- Group G184

## Project Details
- **Problem:** MP099 --- Wheat Yield per Acre (Bhopal)
- **Target:** `yield_qtl_per_acre` (Regression, evaluated using MAE)
- **Faculty Mentor:** Dr. Richa Sharma
- **Batch:** 86

## Team Members
- `S24CSEU2266` ---> Vansh Kaushik
- `S24CSEU2252` ---> Shreya Srivastava
- `S24CSEU2279` ---> Vedant Singh
- `S24CSEU2263` ---> Sanyam Singhroha

---

## Project Description-
This project develops an end-to-end MLOps pipeline for the **Bhopal Kisan Producer Company**. The client procures wheat from member farms across Bhopal and needs reliable pre-harvest yield predictions (`yield_qtl_per_acre`) to plan logistics, book transport trucks, and reserve warehouse storage.

By replacing manual, guesswork-based harvest estimates with automated, drift-monitored predictions, this system minimizes post-harvest spoilage, prevents transport bottlenecks, and optimizes capital allocation for warehouse leasing across the Bhopal region.

The solution covers:
- Versioned data and pipeline management via **DVC** and **Git**.
- Rigorous exploratory data analysis and data cleaning.
- Experiment tracking and model registry using **MLflow** (Champion-Challenger setup).
- Serving via a **FastAPI** web service with strict validation schemas.
- Containerization with **Docker** and deployment manifests for **Kubernetes**.
- Continuous integration with **GitHub Actions**.
- Continuous data monitoring, drift analysis (PSI), and automated retraining triggers.

---

## Repository Structure Map
```text
mlops-g184-mp099/
├── .github/
│   └── workflows/              # GitHub Actions CI pipelines (e.g. ci.yml)
├── .gitignore                  # Git ignore rules (.venv, caches, etc.)
├── app/                        # FastAPI prediction application
├── data/
│   └── raw/                    # Raw datasets (.dvc tracked for large files)
│       ├── train.csv.dvc       # DVC pointer for messy training data
│       ├── later.csv.dvc       # DVC pointer for operational stream data
│       └── holdout_features.csv # Benchmark holdout set (committed to Git)
├── k8s/                        # Kubernetes deployment and service manifests
├── models/                     # Saved model artifacts (.joblib)
├── predictions/                # Predictions on holdout data
├── reports/                    # findings.yaml and project audit reports
├── src/
│   └── wheat_yield/            # Core package source code
│       ├── __init__.py
│       ├── data_checks.py      # Investigation functions for data issues
│       └── explore.py          # Summary and dataset exploration helper
├── tests/                      # Unit and integration test suites
├── params.yaml                 # Centralized configuration parameters
├── README.md                   # Project overview and instructions
└── requirements.txt            # Pinned Python package dependencies.
```
