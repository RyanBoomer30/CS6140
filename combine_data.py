import pandas as pd

df_news = pd.read_csv('data/merged_dataset.csv')
df_tsla = pd.read_csv(
    'data/tesla_enriched_data.csv',
    parse_dates=["('Date', '')"]
)

# Parse "Published" with a fixed format
df_news['Published'] = pd.to_datetime(
    df_news['Published'],
    format='%m/%d/%Y, %I:%M %p, %z UTC',
    errors='coerce',
    utc=True
).dt.tz_convert(None)

# 3) Drop any news rows that failed to parse
df_news = df_news.dropna(subset=['Published'])

# 4) Clean up / rename the TSLA columns
df_tsla = df_tsla.rename(columns={
    "('Date', '')":      'Date',
    "('Close', 'TSLA')": 'Close',
    "('High', 'TSLA')":  'High',
    "('Low', 'TSLA')":   'Low',
    "('Open', 'TSLA')":  'Open',
    "('Volume', 'TSLA')":'Volume',
})
df_tsla = df_tsla.dropna(subset=['Date'])

# 5) Sort both DataFrames by their datetime keys
df_news.sort_values('Published', inplace=True)
df_tsla.sort_values('Date', inplace=True)

# 6) restrict TSLA to the span of your NEWS dates
min_pub = df_news['Published'].min()
max_pub = df_news['Published'].max()
df_tsla = df_tsla[df_tsla['Date'].between(min_pub, max_pub)]

combined = pd.merge_asof(
    df_news,
    df_tsla,
    left_on='Published',
    right_on='Date',
    direction='nearest',
    tolerance=pd.Timedelta(days=120) # Closest of 4 months
)

# 8) Save the result
combined.to_csv('data/final_dataset.csv', index=False)
print(f"Done! {len(combined)} news rows matched to their closest TSLA dates.")
