"""
Combine all sentiment data with financial data into data/final_dataset.csv file
"""

import pandas as pd
from pathlib import Path

def add_extra_features(df):
    """
    Given a DataFrame with a 'Sentiment Score' column,
    add six new features:
      - score_sq             = (Sentiment Score)^2
      - score_cu             = (Sentiment Score)^3
      - score_dev            = |Sentiment Score − 0.5|
      - inverted_score       = 1 − (Sentiment Score)
      - skewed_score         = (Sentiment Score)^2
      - inverted_skewed_score = (1 − Sentiment Score)^2
    """
    s = df['Sentiment Score']
    df['score_sq']             = s.pow(2)
    df['score_cu']             = s.pow(3)
    df['score_dev']            = (s - 0.5).abs()
    df['inverted_score']       = 1 - s
    df['inverted_skewed_score'] = (1 - s).pow(2)
    return df

# ─── Load data ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
df_sentiment = pd.read_csv(DATA_DIR / 'daily_average_sentiment.csv')
df_tsla = pd.read_csv(
    DATA_DIR / 'tesla_enriched_data.csv',
    parse_dates=["('Date', '')"]
)

# ─── Rename TSLA columns ──────────────────────────────────────────────────────────
df_tsla = df_tsla.rename(columns={
    "('Date', '')": 'Date',
    "('Close', 'TSLA')": 'Close',
    "('High', 'TSLA')": 'High',
    "('Low', 'TSLA')": 'Low',
    "('Open', 'TSLA')": 'Open',
    "('Volume', 'TSLA')": 'Volume',
})

# ─── Ensure datetime types ───────────────────────────────────────────────────────
df_sentiment['Published'] = pd.to_datetime(df_sentiment['Published'])
df_tsla            = df_tsla.dropna(subset=['Date'])

# ─── Sort for asof merge ─────────────────────────────────────────────────────────
df_sentiment.sort_values('Published', inplace=True)
df_tsla.sort_values('Date', inplace=True)

# ─── Filter sentiment to TSLA date range ─────────────────────────────────────────
min_pub = df_tsla['Date'].min()
max_pub = df_tsla['Date'].max()
df_sentiment = df_sentiment[df_sentiment['Published'].between(min_pub, max_pub)]

# ─── Merge as‑of on nearest date (±120 days tolerance) ─────────────────────────
combined = pd.merge_asof(
    df_sentiment,
    df_tsla,
    left_on  = 'Published',
    right_on = 'Date',
    direction = 'nearest',
    tolerance = pd.Timedelta(days=120)
)

# ─── Drop duplicate 'Date' column from TSLA ─────────────────────────────────────
combined = combined.drop(columns=['Date'])

# ─── Add all six extra features ────────────────────────────────────────────────
combined = add_extra_features(combined)

# ─── Write out and report ───────────────────────────────────────────────────────
combined.to_csv('../data/final_dataset.csv', index=False)
print(f"Done! {len(combined)} rows written to data/final_dataset.csv with all extra features.")
