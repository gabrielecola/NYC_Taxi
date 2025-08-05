from sklearn.preprocessing import LabelEncoder

def label_encode_object_columns(df):
    """
    Encodes all object (categorical) columns in the DataFrame using LabelEncoder.
    Returns a new DataFrame with encoded values.
    """
    df_encoded = df.copy()
    le = LabelEncoder()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded