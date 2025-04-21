import pandas as pd

# Load the CSV file
df = pd.read_csv('data/news_headlines_2022.csv')

# Display the first few rows to inspect the file structure
print("Before sorting:")
print(df.head())

# Convert the 'Published' column to datetime:
# Remove the " UTC" text to enable proper parsing with timezone offset.
df['Published_parsed'] = pd.to_datetime(
    df['Published'].str.replace(" UTC", ""), 
    format="%m/%d/%Y, %I:%M %p, %z", 
    errors='coerce'
)

# Check for any conversion issues
if df['Published_parsed'].isnull().any():
    print("Warning: Some dates could not be parsed correctly.")
else:
    print("All dates parsed successfully.")

# Sort the dataframe by the parsed Published date in ascending order
df_sorted = df.sort_values('Published_parsed')

df_sorted = df_sorted.drop_duplicates(subset=['Title'], keep='first')  # Keeps the first occurrence

# Optionally, drop the helper 'Published_parsed' column if you only want the original columns
df_sorted = df_sorted.drop(columns=['Published_parsed'])

# Save the sorted data to a new CSV file
sorted_file_path = 'data/news_headlines_2022_sorted.csv'
df_sorted.to_csv(sorted_file_path, index=False)
print(f"\nCSV file has been successfully sorted by date and saved to: {sorted_file_path}")

# Display the first few rows of the sorted dataframe to verify
print("\nAfter sorting:")
print(df_sorted.head())
