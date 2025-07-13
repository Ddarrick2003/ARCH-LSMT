import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df):
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Returns']
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if df.empty:
        raise ValueError("After cleaning, dataframe is empty.")

    df['Log_Volume'] = np.log(df['Volume'] + 1e-6)
    price_features = ['Open', 'High', 'Low', 'Close', 'RSI', 'Log_Volume']
    zscore_features = ['MACD', 'Returns']

    df[price_features] = MinMaxScaler().fit_transform(df[price_features])
    df[zscore_features] = StandardScaler().fit_transform(df[zscore_features])
    return df, MinMaxScaler(), StandardScaler()
