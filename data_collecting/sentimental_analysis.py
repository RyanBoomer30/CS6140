"""
Using Groq API for Sentiment Analysis on News Headlines
"""

import os
import re
import argparse
import pandas as pd
from groq import Groq

# Using tenacity for better retry handling
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)  

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_groq_api(client, prompt, model="llama3-70b-8192"):
    """
    Calls the Groq API with the provided prompt.
    Uses tenacity for automatic retries with exponential backoff.
    
    Args:
        client: The Groq client instance
        prompt: The prompt to send to the API
        model: The model to use (default: llama3-70b-8192)
        
    Returns:
        The stripped text response from the API
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        temperature=0,  # Deterministic output
        max_tokens=10,  # Adjust if necessary
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
        
    # Extract the response text from the completion
    result = chat_completion.choices[0].message.content.strip()
    print(f"API response: {result}") 
    return result

def main():
    csv_file = "../data/news_headlines_2022.csv"
    output_file = "../data/sentimental_score.csv"

    # Check for Groq API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Please set the GROQ_API_KEY environment variable.")
        return

    # Initialize the Groq client with the official SDK
    client = Groq(api_key=groq_api_key)

    # Load the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    results = []
    total_headlines = len(df)

    # Process each headline, call the Groq API, and extract a sentiment score
    for index, row in df.iterrows():
        headline = row.get("Title")
        if not headline:
            print(f"Skipping row {index} because it does not contain a 'Title' field.")
            continue

        print(f"\nProcessing headline {index + 1}/{total_headlines}: {headline}")
        
        # Build the prompt according to the API requirements
        prompt = f"""
            You are given a news headline: "{headline}"
            Perform sentiment analysis on the headline.
            Using a scale where 0 represents a very negative sentiment and 1 represents a very positive sentiment,
            compute and output only a single float value between 0 and 1 representing the overall sentiment.
            Return only the float value with no additional text.
        """

        try:
            api_result = call_groq_api(client, prompt, model=args.model)
        except RetryError as e:
            print(f"Error after maximum retries for headline at row {index}: {e}")
            api_result = None
        except Exception as e:
            print(f"Unexpected error processing headline at row {index}: {e}")
            api_result = None

        # Attempt to convert the result to a float
        sentiment_score = None
        if api_result is not None:
            try:
                # First try direct conversion
                sentiment_score = float(api_result)
            except ValueError:
                # If direct conversion fails, try regex to extract a float
                match = re.search(r"([0-1](?:\.\d+)?)", api_result)
                if match:
                    sentiment_score = float(match.group(1))
                else:
                    print(f"Failed to parse sentiment score from: '{api_result}'")

        results.append({"Title": headline, "Sentiment Score": sentiment_score})
        print(f"Result: {headline} -> {sentiment_score}")

    # Save the results into an output CSV file
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_file, index=False)
        print(f"\nSentiment analysis results saved to {output_file}")
        print(f"Successfully processed {len(results)} headlines.")
        
        # Report on any missing scores
        missing_scores = results_df["Sentiment Score"].isna().sum()
        if missing_scores > 0:
            print(f"Warning: {missing_scores} headlines ({missing_scores/len(results)*100:.1f}%) could not be analyzed.")
    except Exception as e:
        print(f"Error writing output CSV: {e}")

if __name__ == "__main__":
    main()