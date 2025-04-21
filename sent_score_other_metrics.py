import pandas as pd

# Load the CSV file
input_file = 'data/sentimental_score.csv'
output_file = 'data/different_sentimental_scores.csv'

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Create a new column with the transformed scores
df['inverted_score'] = 1 - df['Sentiment Score']
df['skewed_score'] = df['Sentiment Score'] * df['Sentiment Score']
df['inverted_skewed_score'] = df['inverted_score'] * df['inverted_score']

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}")