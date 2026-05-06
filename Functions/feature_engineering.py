import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from config import TARGET, DROP_COLS, TEST_SIZE, RANDOM_STATE


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trip_duration_minutes and pickup_hour derived from datetime columns."""
    df = df.copy()
    df['trip_duration_minutes'] = (
        (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
    )
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    return df


def label_encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every remaining object (categorical) column in-place."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pipeline: temporal features → label encoding."""
    df = add_temporal_features(df)
    df = label_encode_object_columns(df)
    return df


def split_and_scale(
    df: pd.DataFrame,
    drop_cols: list = DROP_COLS,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split into train/test and apply StandardScaler.

    Returns
    -------
    X_train_sc, X_test_sc, y_train, y_test, scaler
    """
    X = df.drop(drop_cols, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler
