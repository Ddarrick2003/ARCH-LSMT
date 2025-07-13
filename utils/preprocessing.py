import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df):
    import streamlit as st

    df = df.copy()

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Returns']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input CSV: {missing_cols}")

    # Ensure all required columns are numeric
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')

    # Add Log_Volume feature
    df['Log_Volume'] = np.log(df['Volume'] + 1e-6)

    # Drop any rows with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'RSI', 'Log_Volume', 'MACD', 'Returns'])

    if df.empty:
        raise ValueError("After cleaning, dataframe is empty. Check your CSV content.")

    # Apply hybrid normalization
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    price_features = ['Open', 'High', 'Low', 'Close', 'RSI', 'Log_Volume']
    df[price_features] = minmax_scaler.fit_transform(df[price_features])

    zscore_features = ['MACD', 'Returns']
    df[zscore_features] = standard_scaler.fit_transform(df[zscore_features])

    st.write("✅ Data successfully preprocessed. Final shape:", df.shape)
    return df, minmax_scaler, standard_scaler


