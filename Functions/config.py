# ─── BigQuery ────────────────────────────────────────────────────────────────
BQ_PROJECT  = 'nyc-taxi-project-455608'
BQ_TABLE    = 'bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022'
SAMPLE_SIZE = 100_000

# ─── Column groups ────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'trip_distance', 'fare_amount', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'imp_surcharge',
    'airport_fee', 'total_amount',
]

IMPUTE_COLS = ['passenger_count', 'rate_code', 'store_and_fwd_flag', 'airport_fee']

# ─── Outlier removal ─────────────────────────────────────────────────────────
OUTLIER_COL    = 'trip_distance'
OUTLIER_Q_LOW  = 0.01
OUTLIER_Q_HIGH = 0.99

# ─── Modelling ────────────────────────────────────────────────────────────────
TARGET    = 'fare_amount'
DROP_COLS = ['fare_amount', 'pickup_datetime', 'dropoff_datetime', 'total_amount']

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = 'VotingRegressor_Comparison'

# ─── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS = {
    'ridge': {'alpha': 1.0},
    'lasso': {'alpha': 0.1},
    'rf':    {'n_estimators': 100, 'random_state': 42},
    'gb':    {'n_estimators': 100, 'random_state': 42},
    'mlp':   {'hidden_layer_sizes': (64, 32), 'max_iter': 2000,
               'early_stopping': True, 'random_state': 42},
    'xgb':   {'n_estimators': 100, 'random_state': 42},
    'lgbm':  {'n_estimators': 100, 'random_state': 42},
}
