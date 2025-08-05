
import pandas as pd


def mean_impute_columns(df, columns):
    for col in columns:
        if df[col].dtype == 'object':
            # Per colonne categoriche: riempio con la modalità più frequente
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Per colonne numeriche
            mean_val = df[col].dropna().mean()
            if pd.notna(mean_val):
                df[col] = df[col].fillna(round(mean_val))
            else:
                df[col] = df[col].fillna(0)  # Se non esiste una media, riempio con 0 o un valore predefinito
    return df



def get_season(week):
    if week <= 8 or week >= 49:
        return 'Winter'
    elif week <= 22:
        return 'Spring'
    elif week <= 35:
        return 'Summer'
    else:
        return 'Autumn'