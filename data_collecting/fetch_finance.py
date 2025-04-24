"""
Fetches Tesla stock data from Yahoo Finance and enriches it with additional metadata.
"""

import yfinance as yf
import pandas as pd
import numpy as np

def load_tesla_data():
    df = yf.download('TSLA', start='2022-01-01', end='2025-04-19', interval='1d')
    df = df.dropna()
    df.reset_index(inplace=True)

    ticker = yf.Ticker('TSLA')
    info = ticker.info

    numeric_info = {
        key.replace(" ", "_").replace("/", "_").replace("%", "pct"): value
        for key, value in info.items()
        if isinstance(value, (int, float))
    }

    meta_df = pd.DataFrame([numeric_info] * len(df), index=df.index)

    df_combined = pd.concat([df, meta_df], axis=1)


    date_col = df_combined.iloc[:, 0]  # First column is 'Date'
    data_only = df_combined.iloc[:, 1:]
    numeric_data = data_only.select_dtypes(include='number')
    final_df = pd.concat([date_col, numeric_data], axis=1)

    return final_df


df = load_tesla_data()
desired_col_indices = [0, 1, 2, 3, 4, 5, 16, 19, 25, 26, 27, 37, 32, 71, 60, 90, 64]
df = df.iloc[:, desired_col_indices]
df = df.loc[:, df.nunique() > 1]
print(df.head())


csv_path = '../data/tesla_enriched_data.csv'
df.to_csv(csv_path, index=False)
print(f"Data saved to '{csv_path}'")