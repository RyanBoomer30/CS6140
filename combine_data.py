import pandas as pd

# Load the sentiment data and TSLA data
df_sentiment = pd.read_csv('data/daily_average_sentiment.csv')
df_tsla = pd.read_csv(
    'data/tesla_enriched_data.csv',
    parse_dates=["('Date', '')"]
)

# rename columns in df_tsla
df_tsla = df_tsla.rename(columns={
    "('Date', '')": 'Date',
    "('Close', 'TSLA')": 'Close',
    "('High', 'TSLA')": 'High',
    "('Low', 'TSLA')": 'Low',
    "('Open', 'TSLA')": 'Open',
    "('Volume', 'TSLA')": 'Volume',
})

# make sure that the 'Published' column in df_sentiment is in datetime format
df_sentiment['Published'] = pd.to_datetime(df_sentiment['Published'])

# make sure the 'Date' column in df_tsla is in datetime format
df_tsla = df_tsla.dropna(subset=['Date'])

# sort both DataFrames by date
df_sentiment.sort_values('Published', inplace=True)
df_tsla.sort_values('Date', inplace=True)

# filter sentiment data to match TSLA date range
min_pub = df_tsla['Date'].min()
max_pub = df_tsla['Date'].max()
df_sentiment = df_sentiment[df_sentiment['Published'].between(min_pub, max_pub)]

# combine data
combined = pd.merge_asof(
    df_sentiment,
    df_tsla,
    left_on='Published',
    right_on='Date',
    direction='nearest',
    tolerance=pd.Timedelta(days=120)
)

# drop duplicate columns
combined = combined.drop(columns=['Date'])

combined.to_csv('data/final_dataset.csv', index=False)
print(f"Done! {len(combined)} news rows matched to their closest TSLA dates.")