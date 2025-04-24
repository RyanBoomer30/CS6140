"""
Merge data from headlines and its sentimental score
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv('../data/news_headlines_2022_sorted.csv')
df2 = pd.read_csv('../data/sentimental_score.csv')

df1_selected = df1[['Title', 'Published']]
df2_selected = df2[['Title', 'Sentiment Score']]


merged_df = pd.merge(df1_selected, df2_selected, on='Title', how='inner')

merged_df.to_csv('../data/merged_dataset.csv', index=False)

merged_df['Published'] = pd.to_datetime(merged_df['Published'].str.split(',').str[0], format='%m/%d/%Y')

daily_sentiment = merged_df.groupby('Published')['Sentiment Score'].mean().reset_index()

daily_sentiment_sorted = daily_sentiment.sort_values('Published')
print(daily_sentiment_sorted)

daily_sentiment_sorted.to_csv('../data/daily_average_sentiment.csv', index=False)
