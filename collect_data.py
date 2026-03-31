"""
Data collection pipeline for sentiment-return analysis.
Pulls stock prices and news from Financial Modeling Prep (FMP) API.
Scores sentiment with VADER and saves structured output.

Usage: python collect_data.py [--extended]
  --extended: Use a larger date range and more tickers for IEEE TAC expansion
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_STABLE = "https://financialmodelingprep.com/stable"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Original study tickers and window
ORIGINAL_TICKERS = ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA"]
ORIGINAL_START = "2025-03-01"
ORIGINAL_END = "2025-05-04"

# Extended tickers: add sector diversity (finance, healthcare, energy, consumer, industrials)
EXTENDED_TICKERS = ORIGINAL_TICKERS + [
    "SPY",   # market proxy (always needed)
    "MSFT",  # tech
    "META",  # tech/social
    "JPM",   # financials
    "JNJ",   # healthcare
    "XOM",   # energy
    "WMT",   # consumer staples
    "BA",    # industrials
    "DIS",   # media/entertainment
    "AMD",   # semiconductors
]
EXTENDED_START = "2024-10-01"
EXTENDED_END = "2025-05-31"

# SPY is always included for market-adjusted analysis
MARKET_PROXY = "SPY"


def fetch_daily_prices(ticker, start_date, end_date):
    """Fetch daily historical prices from FMP stable API."""
    url = f"{FMP_BASE_STABLE}/historical-price-eod/full"
    params = {"symbol": ticker, "from": start_date, "to": end_date, "apikey": FMP_API_KEY}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data or not isinstance(data, list):
        print(f"  WARNING: No price data returned for {ticker}")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]


def fetch_finnhub_news(ticker, start_date, end_date):
    """Fetch company news from Finnhub API.
    Finnhub returns up to ~1000 articles per request window.
    We chunk by week to maximize coverage.
    """
    all_articles = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=7), end)
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": current.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "token": FINNHUB_API_KEY,
        }
        for attempt in range(5):
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            print(f"    WARNING: Skipping chunk {current.date()} - {chunk_end.date()} after 5 retries")
            current = chunk_end + timedelta(days=1)
            continue

        articles = resp.json()
        if isinstance(articles, list):
            # Normalize to common format
            for art in articles:
                all_articles.append({
                    "publishedDate": datetime.fromtimestamp(art.get("datetime", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                    "title": art.get("headline", ""),
                    "text": art.get("summary", ""),
                    "site": art.get("source", ""),
                    "url": art.get("url", ""),
                })
        current = chunk_end + timedelta(days=1)
        time.sleep(1.2)  # more conservative rate limiting

    # Deduplicate by title
    seen_titles = set()
    deduped = []
    for art in all_articles:
        title = art["title"].strip().lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            deduped.append(art)

    return deduped


def score_sentiment(articles):
    """Score each article with VADER sentiment."""
    analyzer = SentimentIntensityAnalyzer()
    scored = []
    for art in articles:
        text = ""
        if art.get("title"):
            text += art["title"]
        if art.get("text"):
            text += " " + art["text"]
        if not text.strip():
            continue
        scores = analyzer.polarity_scores(text)
        scored.append({
            "publishedDate": art.get("publishedDate", ""),
            "title": art.get("title", ""),
            "source": art.get("site", ""),
            "url": art.get("url", ""),
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"],
        })
    return scored


def align_trading_day(dt, trading_dates):
    """Align a datetime to the appropriate trading day.
    Articles published after 16:00 ET are aligned to the next trading day.
    """
    if pd.isna(dt):
        return pd.NaT
    # Simple ET approximation (UTC-4 during EDT)
    pub_date = dt.date()
    pub_hour = dt.hour
    # If after 4 PM or if published on a non-trading day, assign to next trading day
    target = pub_date
    if pub_hour >= 16:
        target = pub_date + timedelta(days=1)
    # Find the next trading date >= target
    future = trading_dates[trading_dates >= pd.Timestamp(target)]
    if len(future) > 0:
        return future.iloc[0]
    return pd.NaT


def run_collection(tickers, start_date, end_date, output_dir):
    """Main collection loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Always include SPY
    all_tickers = list(dict.fromkeys(tickers + [MARKET_PROXY]))

    for ticker in all_tickers:
        print(f"\n{'='*60}")
        print(f"Processing {ticker} ({start_date} to {end_date})")
        print(f"{'='*60}")

        ticker_dir = output_path / ticker
        ticker_dir.mkdir(exist_ok=True)

        # Skip if already collected
        if (ticker_dir / "daily_sentiment.csv").exists() or (ticker == MARKET_PROXY and (ticker_dir / "prices.csv").exists()):
            print(f"  Already collected, skipping (delete {ticker_dir} to re-collect)")
            continue

        # 1. Fetch prices
        print(f"  Fetching prices...")
        prices = fetch_daily_prices(ticker, start_date, end_date)
        if prices.empty:
            print(f"  SKIPPING {ticker}: no price data")
            continue
        prices.to_csv(ticker_dir / "prices.csv", index=False)
        print(f"  Got {len(prices)} trading days")

        # Compute log returns
        prices["log_return"] = np.log(prices["close"] / prices["close"].shift(1))
        prices.to_csv(ticker_dir / "prices.csv", index=False)

        # 2. Fetch news (skip for SPY - it's just a market proxy)
        if ticker == MARKET_PROXY:
            print(f"  SPY is market proxy only - skipping news collection")
            continue

        print(f"  Fetching news articles...")
        articles = fetch_finnhub_news(ticker, start_date, end_date)
        print(f"  Got {len(articles)} raw articles")

        # Save raw articles
        with open(ticker_dir / "raw_articles.json", "w") as f:
            json.dump(articles, f, indent=2, default=str)

        # 3. Score sentiment
        print(f"  Scoring sentiment...")
        scored = score_sentiment(articles)
        print(f"  Scored {len(scored)} articles")

        scored_df = pd.DataFrame(scored)
        if scored_df.empty:
            print(f"  WARNING: No scored articles for {ticker}")
            continue

        scored_df.to_csv(ticker_dir / "scored_articles.csv", index=False)

        # 4. Align to trading days and aggregate
        print(f"  Aggregating daily sentiment...")
        scored_df["publishedDate"] = pd.to_datetime(scored_df["publishedDate"], errors="coerce")
        trading_dates = prices["date"]

        scored_df["trading_date"] = scored_df["publishedDate"].apply(
            lambda x: align_trading_day(x, trading_dates)
        )
        scored_df = scored_df.dropna(subset=["trading_date"])

        # Daily aggregation using 10% trimmed mean
        from scipy.stats import trim_mean
        daily_sent = scored_df.groupby("trading_date").agg(
            sentiment_trimmed_mean=("compound", lambda x: trim_mean(x, 0.1)),
            sentiment_mean=("compound", "mean"),
            sentiment_median=("compound", "median"),
            article_count=("compound", "count"),
        ).reset_index()
        daily_sent.rename(columns={"trading_date": "date"}, inplace=True)
        daily_sent.to_csv(ticker_dir / "daily_sentiment.csv", index=False)
        print(f"  Generated {len(daily_sent)} daily sentiment observations")

        # Save scored articles with trading date alignment
        scored_df.to_csv(ticker_dir / "scored_articles_aligned.csv", index=False)

    print(f"\nData collection complete. Output in: {output_path}")


if __name__ == "__main__":
    extended = "--extended" in sys.argv

    if extended:
        print("EXTENDED MODE: Larger date range + more tickers for IEEE TAC")
        run_collection(EXTENDED_TICKERS, EXTENDED_START, EXTENDED_END, "data/extended")
    else:
        print("ORIGINAL MODE: Replicating original study window")
        run_collection(ORIGINAL_TICKERS, ORIGINAL_START, ORIGINAL_END, "data/original")
