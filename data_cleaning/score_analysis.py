"""
Analysis of the sentiment scores, positive, negative and neutral
"""

import pandas as pd

def add_basis_features(df):
    # original column is assumed to be 'Sentiment Score'
    s = df['Sentiment Score']
    # 1) square
    df['score_sq']   = s ** 2
    # 2) cube
    df['score_cu']   = s ** 3
    # 3) distance from neutral (0.5)
    df['score_dev']  = (s - 0.5).abs()
    return df

def count_sentiments(df):
    pos = ((df['Sentiment Score'] > 0.5)).sum()
    neg = ((df['Sentiment Score'] < 0.5)).sum()
    neu = ((df['Sentiment Score'] == 0.5)).sum()
    return pos, neg, neu

# ─── News headlines sentiment ──────────────────────────────────────────────────
print("Calculating the number of positive and negative sentiment scores for news headlines...")
df_news = pd.read_csv('data/sentimental_score.csv')
df_news = add_basis_features(df_news)

pos, neg, neu = count_sentiments(df_news)
print(f"Positive sentiment scores: {pos}")
print(f"Negative sentiment scores: {neg}")
print(f"Neutral sentiment scores: {neu}")

print("----------------------------------------------------------------------------")

# ─── Daily average sentiment ───────────────────────────────────────────────────
print("Calculating the number of positive and negative sentiment scores for daily average sentiment...")
df_daily = pd.read_csv('data/final_dataset.csv')
df_daily = add_basis_features(df_daily)

pos, neg, neu = count_sentiments(df_daily)
print(f"Positive sentiment scores: {pos}")
print(f"Negative sentiment scores: {neg}")
print(f"Neutral sentiment scores: {neu}")