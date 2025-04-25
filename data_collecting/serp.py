import requests
import time
import pandas as pd
import os

def main():
  API_KEY = os.environ.get("SERP_API_KEY")

  if not API_KEY:
      print("Please set the SERP_API_KEY environment variable.")
      return

  QUERIES = [
    "Autonomous driving technology breakthrough",
    "EV competition",
    "Electric car profit",
    "EV sales",
    "Tesla earnings",
    "Tesla market share",
    "Tesla FSD",
    "Lithium supply chain",
    "Tesla import export",
    "Tesla production efficiency",
    "Elon Musk politics",
    "Battery Technology",
    "Autonomous Vehicles",
    "Cybertruck",
    "BYD",
    "Full-self driving Cars",
    "Elon on X",
    "Doge",
    "TESLA earning report",
    "Tesla Stock Slides Amid New U.S.-China Tariff Talks",
    "Elon Musk's Political Stunts Stir Investor Uncertainty Again",
    "EV Competition Heats Up, Tesla Market Share Dips",
    "Trump's Return Sparks Debate on EV Subsidy Cuts",
    "Tesla Production Delays Blamed on Rising Tariff Costs",
    "Cybertruck Launch Disappoints, Tesla Shares Take a Hit",
    "Elon Musk Tweets Trigger Volatility in Tesla Valuation",
    "Global EV Demand Grows, Tesla Faces Supply Chain Strain",
    "Tesla Earnings Beat Expectations Despite Trade War Pressure",
    "Musk's China Visit Calms Fears Over Expansion Slowdown",
    "Tesla recall",
    "Tesla short",
    "Tesla stock crash",
    "EV market downturn",
    "Tesla autopilot accidents",
    "Self-driving car failures",
    "Tesla losing market share",
    "Tesla lawsuit",
    "EV competition intensifies",
    "Tesla regulatory challenges"
  ]

  NUM_RESULTS = 200
  RESULTS_PER_PAGE = 10

  csv_filename = "../data/news_headlines_2022.csv"

  # If the CSV already exists, load its Titles into a set to avoid re-appending
  if os.path.exists(csv_filename):
      existing_df = pd.read_csv(csv_filename, usecols=["Title"])
      seen_titles = set(existing_df["Title"].dropna().astype(str).tolist())
      write_header = False
  else:
      seen_titles = set()
      write_header = True

  for query in QUERIES:
      for page in range(0, NUM_RESULTS // RESULTS_PER_PAGE):
          params = {
              "q": query,
              "engine": "google_news",
              "api_key": API_KEY,
              "hl": "en",
              "gl": "us",
              "num": RESULTS_PER_PAGE,
              "start": page * RESULTS_PER_PAGE
          }

          response = requests.get("https://serpapi.com/search", params=params)
          data = response.json()

          news_results = data.get("news_results", [])
          new_rows = []
          for item in news_results:
              title = item.get("title")
              if title and title not in seen_titles:
                  seen_titles.add(title)
                  new_rows.append({
                      "Title": title,
                      "Source": item.get("source"),
                      "Published": item.get("date"),
                      "Link": item.get("link")
                  })

          if new_rows:
              df_new = pd.DataFrame(new_rows)
              # append new uniques only
              df_new.to_csv(csv_filename, mode="a", index=False, header=write_header)
              write_header = False  # only write header once
              print(f"Appended {len(new_rows)} new headlines (total seen: {len(seen_titles)})")
          else:
              print("No new headlines on this page.")

          time.sleep(1.5)

      print(f"\n✅ Done processing query '{query}'.")

if __name__ == "__main__":
    main()