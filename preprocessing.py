import re
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader.data as web
import yfinance as yf
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

cnbc_link = 'data/cnbc_headlines.csv'
guardian_link = 'data/guardian_headlines.csv' 
reuter_link = 'data/reuters_headlines.csv'
lm_dictionary = 'data/LM_dictionary.csv'

# Helpers
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns.values]
    df.columns = df.columns.astype(str)
    return df

def clean_date_string(date_str):
    if not isinstance(date_str, str):
        date_str = str(date_str)
    cleaned = re.sub(r'\s*ET', '', date_str)
    return cleaned.strip()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    return [word for word in tokens if word not in stop_words]

def compute_sentiment(tokens, pos, neg):
    return sum(1 for t in tokens if t in pos) - sum(1 for t in tokens if t in neg)

# Load financial headline
def load_news_data():
    cnbc_df = pd.read_csv(cnbc_link)
    guardian_df = pd.read_csv(guardian_link)
    reuters_df = pd.read_csv(reuter_link)
    cnbc_df = cnbc_df.rename(columns={'Time': 'Date', 'Headlines': 'Headline'})
    guardian_df = guardian_df.rename(columns={'Time': 'Date', 'Headlines': 'Headline'})
    reuters_df = reuters_df.rename(columns={'Time': 'Date', 'Headlines': 'Headline'})
    news_df = pd.concat([cnbc_df[['Date', 'Headline']],
                         guardian_df[['Date', 'Headline']],
                         reuters_df[['Date', 'Headline']]],
                        ignore_index=True)
    news_df['Date'] = news_df['Date'].apply(clean_date_string)
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')
    news_df = news_df.dropna(subset=['Date'])
    news_df['Date'] = news_df['Date'].dt.date
    news_df.sort_values('Date', inplace=True)
    return news_df

stop_words = set(stopwords.words('english'))

# Load LM sentimental dictionary
def load_lm_lexicon():
    lm_df = pd.read_csv(lm_dictionary)
    lm_df['Word'] = lm_df['Word'].str.lower()
    pos = set(lm_df.loc[lm_df['Positive'] > 0, 'Word'])
    neg = set(lm_df.loc[lm_df['Negative'] > 0, 'Word'])
    return pos, neg

# Load VIX, S&P500 and Yield spread
def load_market_data(start_date, end_date):
    vix = web.DataReader("VIXCLS", "fred", start_date, end_date).rename(columns={"VIXCLS": "VIX"})
    vix.index = pd.to_datetime(vix.index).normalize()
    vix.index.name = "Date"

    spread = web.DataReader("T10Y2Y", "fred", start_date, end_date).rename(columns={"T10Y2Y": "YieldSpread"})
    spread.index = pd.to_datetime(spread.index).normalize()
    spread.index.name = "Date"

    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    col = 'Adj Close' if 'Adj Close' in sp500.columns else 'Close'
    sp500['SP500_Return'] = sp500[col].pct_change() * 100
    sp500 = sp500[['SP500_Return']]
    sp500.index = pd.to_datetime(sp500.index).normalize()
    sp500.index.name = "Date"

    return ensure_flat_columns(vix), ensure_flat_columns(spread), ensure_flat_columns(sp500)

def main():
    # Loading and preprocessing
    news_df = load_news_data()
    print("News data loaded:", news_df.shape)

    news_df['Tokens'] = news_df['Headline'].apply(preprocess_text)
    pos, neg = load_lm_lexicon()
    print(f"Loaded LM lexicon: {len(pos)} positive, {len(neg)} negative words")

    news_df['SentimentScore'] = news_df['Tokens'].apply(lambda t: compute_sentiment(t, pos, neg))
    daily_sentiment = news_df.groupby('Date')['SentimentScore'].mean().to_frame()
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index).normalize()
    daily_sentiment.index.name = "Date"


    # Timeline since the first financial new article to the last in the dataset
    start = pd.to_datetime(news_df['Date'].min()).normalize()
    end = pd.to_datetime(news_df['Date'].max()).normalize()
    vix, spread, sp500 = load_market_data(start, end)
    print("Market data loaded.")

    # Merging all the datasets
    ds = ensure_flat_columns(daily_sentiment.reset_index())
    vix = ensure_flat_columns(vix.reset_index())
    spread = ensure_flat_columns(spread.reset_index())
    sp500 = ensure_flat_columns(sp500.reset_index())

    df = ds.merge(vix, on="Date").merge(spread, on="Date").merge(sp500, on="Date")
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)

    print("Final merged dataset shape:", df.shape)
    df.to_csv("final_dataset.csv", index=False)

if __name__ == "__main__":
    main()