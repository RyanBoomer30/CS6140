# 1. Data collecting

## Requirement: Save the api keys as the following environment variables
- Serp_API_KEY: needed to perform google scraping
- Groq_API_KEY: needed to perform sentimental analysis

## a. fetch_finance.py -> data/tesla_enriched_data.csv
This is used to create all the dataset that includes all TESLA financial data from beginning of 2022 to April 2025

## b. serp.py -> data/news_headlines_2022.csv
This is used to create a financial headlines dataset for our sentiment evaluation. The queries are all included in the QUERIES variable (Note that each free account can only generates about 1000 unique data indexes since there are many duplicated results)

## c. sentimental_analysis.py -> sentimental_score.csv
Given the financial headlines dataset generated above, run an LLM prompt through all the indexes and produce a new dataset that includes the sentimental score. Feel free to change the prompt or model

# 2. Data cleaning
## a. sort_headline.py -> news_headlines_2022_sorted.csv
Sort the new headlines by date and removed any duplicated titles

## b. merge_data.py -> final_dataset.csv
Merge all the headlines and sentimental datasets together by matching the headline dates with the corresponding average sentimental score of that date, removing the headlines, and adding basis function expansion for the features

# 3. Model
Run any of the following scripts below once the final_dataset has been created to train the models
## a. news_for_LSTM.py
## b. news_for_RNN.py
## c. random_forest.py