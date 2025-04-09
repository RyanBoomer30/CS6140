from langchain.llms import OpenAI

from langchain import Chain

# Initialize OpenAI LLM
llm = OpenAI(model="text-davinci-003")

# Define a function for sentiment analysis
def analyze_sentiment(text):
    response = llm.completion(
        prompt=f"Analyze the sentiment of the following text: {text}",

        max_tokens=60
    )

    return response.choices[0].text.strip()

if __name__ == "__main__":
    # Example usage
    text = "I love the new design of the website!"

    sentiment = analyze_sentiment(text)

    print(f"Sentiment: {sentiment}")