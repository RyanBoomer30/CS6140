import pandas as pd
import numpy as np

df1 = pd.read_csv('data/news_headlines_2022_sorted.csv')
df2 = pd.read_csv('data/different_sentimental_scores.csv')

df1_selected = df1[['Title', 'Published']]
df2_selected = df2[['Title', 'Sentiment Score', 'inverted_score', 'skewed_score', 'inverted_skewed_score']]

merged_df = pd.merge(df1_selected, df2_selected, on='Title', how='inner')

merged_df.to_csv('data/merged_dataset_with_sent_scores.csv', index=False)

merged_df['Published'] = pd.to_datetime(merged_df['Published'].str.split(',').str[0], format='%m/%d/%Y')

# Group by 'Published' and calculate the mean for all score columns
daily_sentiment = merged_df.groupby('Published')[['Sentiment Score', 'inverted_score', 'skewed_score', 'inverted_skewed_score']].mean().reset_index()

daily_sentiment_sorted = daily_sentiment.sort_values('Published')
print(daily_sentiment_sorted)

daily_sentiment_sorted.to_csv('data/dataset_with_more_sent_scores.csv', index=False)
