import pandas as pd

from config import NUMERIC_COLS, IMPUTE_COLS, OUTLIER_COL, OUTLIER_Q_LOW, OUTLIER_Q_HIGH


def convert_numeric_columns(df: pd.DataFrame, cols: list = NUMERIC_COLS) -> pd.DataFrame:
    """Cast object columns that should be numeric to float."""
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def mean_impute_columns(df: pd.DataFrame, columns: list = IMPUTE_COLS) -> pd.DataFrame:
    """
    Impute missing values:
    - categorical columns → most frequent value
    - numeric columns     → rounded mean (0 if no valid mean exists)
    """
    df = df.copy()
    for col in columns:
        if df[col].dtype == 'object':
            fill = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(fill)
        elif pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].dropna().mean()
            df[col] = df[col].fillna(round(mean_val) if pd.notna(mean_val) else 0)
    return df


def remove_outliers(
    df: pd.DataFrame,
    col: str = OUTLIER_COL,
    q_low: float = OUTLIER_Q_LOW,
    q_high: float = OUTLIER_Q_HIGH,
) -> pd.DataFrame:
    """Remove rows where `col` falls outside [Q_low - 1.5*IQR, Q_high + 1.5*IQR]."""
    Q1 = df[col].quantile(q_low)
    Q3 = df[col].quantile(q_high)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


def get_season(week: int) -> str:
    """Map an ISO week number to a season name."""
    if week <= 8 or week >= 49:
        return 'Winter'
    elif week <= 22:
        return 'Spring'
    elif week <= 35:
        return 'Summer'
    else:
        return 'Autumn'


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full pre-processing pipeline: convert → impute → remove outliers."""
    df = convert_numeric_columns(df)
    df = mean_impute_columns(df)
    df = remove_outliers(df)
    return df
