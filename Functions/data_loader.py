import warnings
import pandas as pd
import db_dtypes  # required by BigQuery's .to_dataframe()
import google.cloud.bigquery
from google.cloud import bigquery

from config import BQ_PROJECT, BQ_TABLE, SAMPLE_SIZE


def load_from_bigquery(
    project: str = BQ_PROJECT,
    table: str = BQ_TABLE,
    sample_size: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Load a random sample from a BigQuery table."""
    client = bigquery.Client(project=project)
    query = f"""
        SELECT *
        FROM `{table}`
        ORDER BY RAND()
        LIMIT {sample_size}
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return client.query(query).to_dataframe()
