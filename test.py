import pandas as pd

# Read sentimental_score.csv and prints out the total number of positive and negative
df = pd.read_csv('data/final_dataset.csv')

positive = 0
negative = 0

for index, row in df.iterrows():
    score = row.get("Sentiment Score")

    if score >= 0.5:
        positive += 1
    else:
        negative += 1

print(positive)
print(negative)