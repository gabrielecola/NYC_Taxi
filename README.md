# NYC Taxi Fare Prediction

End-to-end machine learning project that predicts NYC yellow taxi **fare amounts** using a 100k-row sample from the BigQuery public dataset (`tlc_yellow_trips_2022`).

**Authors:** Gabriele Cola · Swann Etro

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Google Cloud Authentication](#google-cloud-authentication)
4. [Environment Setup](#environment-setup)
5. [Running the Notebook](#running-the-notebook)
6. [MLflow Tracking](#mlflow-tracking)
7. [Models](#models)

---

## Project Structure

```
NYC_Taxi/
├── NYC_Taxi2.ipynb          # Main notebook (orchestrator)
├── Functions/
│   ├── config.py            # All constants: BQ project, columns, hyperparameters
│   ├── data_loader.py       # BigQuery loading logic
│   ├── pre_processing.py    # Type conversion, NA imputation, outlier removal
│   ├── feature_engineering.py  # Temporal features, label encoding, train/test split
│   ├── modelling.py         # Model building, training, evaluation
│   └── mlflow_tracking.py   # MLflow logging helpers
├── Data/                    # Local data files (if any)
├── mlruns/                  # MLflow run artifacts (auto-generated)
├── requirements.txt
└── README.md
```

---

## Prerequisites

- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with the **BigQuery API enabled**

---

## Google Cloud Authentication

The notebook reads data directly from BigQuery. You need to authenticate your local machine before running it.

### 1. Install the Google Cloud SDK

Follow the official guide for your OS: https://cloud.google.com/sdk/docs/install

Verify the installation:

```bash
gcloud --version
```

### 2. Log in to your Google account

```bash
gcloud auth login
```

A browser window will open. Sign in with the Google account that has access to the BigQuery project.

### 3. Set up Application Default Credentials (ADC)

This is what the Python BigQuery client uses internally:

```bash
gcloud auth application-default login
```

Again, a browser window will open. After completing this step, credentials are saved locally and the notebook will pick them up automatically — no service account key needed.

### 4. Set your default project

```bash
gcloud config set project nyc-taxi-project-455608
```

### 5. Verify access

```bash
bq ls bigquery-public-data:new_york_taxi_trips
```

You should see a list of tables including `tlc_yellow_trips_2022`.

> **Common error:** `google.auth.exceptions.DefaultCredentialsError`
> → You skipped step 3. Run `gcloud auth application-default login` and retry.

---

## Environment Setup

### 1. Create the conda environment

```bash
conda create -n NYC_Taxi python=3.11 -y
conda activate NYC_Taxi
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install packages not in requirements.txt

```bash
pip install db_dtypes mlflow xgboost lightgbm
```

> `db_dtypes` is required by the BigQuery client to parse certain column types into pandas.

### 4. Register the kernel with Jupyter

```bash
python -m ipykernel install --user --name NYC_Taxi --display-name "Python (NYC_Taxi)"
```

### 5. Select the kernel in VS Code

`Cmd+Shift+P` → **Python: Select Interpreter** → pick `NYC_Taxi`

---

## Running the Notebook

Open `NYC_Taxi2.ipynb` and run all cells top to bottom. The notebook is organised into six sections:

| Section | What happens |
|---------|-------------|
| **0. Libraries** | Imports and loads all custom modules from `Functions/` |
| **1. Importing** | Pulls 100k random rows from BigQuery |
| **2. Pre-Processing** | Type conversion → NA imputation → outlier removal |
| **3. EDA** | Distributions, peak hours, seasonality, anomaly detection, correlation heatmap |
| **4. Feature Engineering** | Temporal features, label encoding, train/test split, StandardScaler |
| **5. Model Training** | Trains 7 models + VotingRegressor, prints RMSE / MAE / R² summary |
| **6. Model Tracking** | Logs all runs, metrics, plots and serialised models to MLflow |

---

## MLflow Tracking

After running section 6, launch the MLflow UI to compare runs:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

Each run logs:
- **Metrics:** RMSE, MAE, R²
- **Parameters:** model hyperparameters
- **Artifacts:** True-vs-Predicted scatter plots, serialised model files
- **Parent run:** surfaces the best model name and its RMSE at a glance

To clean up old MLflow runs:

```bash
bash cleanup_mlruns.sh
```

---

## Models

| Model | Library |
|-------|---------|
| Linear Regression | scikit-learn |
| Ridge Regression | scikit-learn |
| Lasso Regression | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| MLP Neural Network | scikit-learn |
| XGBoost | xgboost |
| **VotingRegressor** (ensemble) | scikit-learn |

The best single model from the initial runs is **Gradient Boosting**, with the **VotingRegressor** ensemble close behind.
Section 5.2 also performs a `RandomizedSearchCV` hyperparameter search on Ridge Regression specifically, and saves the best estimator to `ridge_best_model.pkl`.
