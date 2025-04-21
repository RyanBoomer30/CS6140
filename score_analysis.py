import pandas as pd

# calculate the number of positive and negative sentiment scores

print("Calculating the number of positive and negative sentiment scores for news headlines...")
df = pd.read_csv('data/sentimental_score.csv')

positive = 0
negative = 0
neural = 0

for index, row in df.iterrows():
    score = row.get("Sentiment Score")

    if score > 0.5:
        positive += 1
    elif score < 0.5:
        negative += 1
    else:
        neural += 1

print(f"Positive sentiment scores: {positive}")
print(f"Negative sentiment scores: {negative}")
print(f"Neutral sentiment scores: {neural}")

print("----------------------------------------------------------------------------")

print("Calculating the number of positive and negative sentiment scores for daily average sentiment...")
df = pd.read_csv('data/final_dataset.csv')

positive = 0
negative = 0
neural = 0

for index, row in df.iterrows():
    score = row.get("Sentiment Score")

    if score > 0.5:
        positive += 1
    elif score < 0.5:
        negative += 1
    else:
        neural += 1

print(f"Positive sentiment scores: {positive}")
print(f"Negative sentiment scores: {negative}")
print(f"Neutral sentiment scores: {neural}")