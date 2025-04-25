import pandas as pd
from pathlib import Path

def add_extra_features(df):
    s = df['Sentiment Score']
    df['score_sq'] = s.pow(2)
    df['score_cu'] = s.pow(3)
    df['score_dev'] = (s - 0.5).abs()
    df['inverted_score'] = 1 - s
    df['inverted_skewed_score'] = (1 - s).pow(2)
    return df

def main():
    data_dir = Path(__file__).resolve().parent.parent / 'data'

    # Merge headlines with sentiment scores and compute daily averages
    headlines = pd.read_csv(data_dir / 'news_headlines_2022_sorted.csv')
    scores = pd.read_csv(data_dir / 'sentimental_score.csv')
    merged = pd.merge(
        headlines[['Title', 'Published']],
        scores[['Title', 'Sentiment Score']],
        on='Title'
    )
    merged['Published'] = pd.to_datetime(
        merged['Published'].str.split(',').str[0],
        format='%m/%d/%Y'
    )
    daily_sentiment = (
        merged
        .groupby('Published', as_index=False)['Sentiment Score']
        .mean()
        .sort_values('Published')
    )

    # Load daily sentiment and TSLA data
    df_sent = daily_sentiment
    df_tsla = pd.read_csv(
        data_dir / 'tesla_enriched_data.csv',
        parse_dates=["('Date', '')"]
    ).rename(columns={
        "('Date', '')": 'Date',
        "('Close', 'TSLA')": 'Close',
        "('High', 'TSLA')": 'High',
        "('Low', 'TSLA')": 'Low',
        "('Open', 'TSLA')": 'Open',
        "('Volume', 'TSLA')": 'Volume',
    }).dropna(subset=['Date'])

    # Align and merge on nearest date within 120 days
    df_sent = df_sent.sort_values('Published')
    df_tsla = df_tsla.sort_values('Date')
    combined = pd.merge_asof(
        df_sent,
        df_tsla,
        left_on='Published',
        right_on='Date',
        direction='nearest',
        tolerance=pd.Timedelta(days=120)
    ).drop(columns=['Date'])

    combined = add_extra_features(combined)
    combined.to_csv(data_dir / 'final_dataset.csv', index=False)
    print(f"Done! {len(combined)} rows written to {data_dir/'final_dataset.csv'}")

if __name__ == '__main__':
    main()