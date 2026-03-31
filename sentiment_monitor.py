"""
Sentiment Anomaly Detection System
-----------------------------------
Implements the monitoring tool described in Section 4.10 of the paper:

    Z_{k,t} = (S_{k,t} - mu_{k,t}^{(w)}) / sigma_{k,t}^{(w)}

If |Z_{k,t}| > threshold and no corroborating event is found, flag for review.

Modes:
  --backtest    Run on historical data to show what would have been flagged
  --live        Poll Finnhub every N minutes for new articles, score, and alert

Usage:
  python sentiment_monitor.py --backtest
  python sentiment_monitor.py --live --interval 15
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

TICKERS = ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA", "MSFT",
           "META", "JPM", "JNJ", "XOM", "WMT", "BA", "DIS", "AMD"]

# Rolling window size (trading days)
WINDOW = 20

# Z-score threshold for flagging anomalies
Z_THRESHOLD = 2.0

SECTOR = dict(AAPL="Tech", GOOGL="Tech", AMZN="Tech", MSFT="Tech", META="Tech",
              NVDA="Semiconductors", AMD="Semiconductors",
              TSLA="Automotive", JPM="Finance", JNJ="Healthcare",
              XOM="Energy", WMT="Consumer", BA="Industrials", DIS="Media")

analyzer = SentimentIntensityAnalyzer()


# =============================================================================
# Core: compute Z-score from a sentiment series
# =============================================================================
def compute_z_scores(daily_sentiment, window=WINDOW):
    """
    Given a Series of daily sentiment values, compute the rolling Z-score:
        Z_t = (S_t - rolling_mean) / rolling_std
    """
    rolling_mean = daily_sentiment.rolling(window=window, min_periods=5).mean()
    rolling_std = daily_sentiment.rolling(window=window, min_periods=5).std()
    z = (daily_sentiment - rolling_mean) / rolling_std.replace(0, np.nan)
    return z, rolling_mean, rolling_std


def flag_anomalies(dates, z_scores, threshold=Z_THRESHOLD):
    """Return list of (date, z_score) for anomaly days."""
    flags = []
    for date, z in zip(dates, z_scores):
        if pd.notna(z) and abs(z) > threshold:
            flags.append((date, z))
    return flags


# =============================================================================
# Backtest mode: run on historical data
# =============================================================================
def run_backtest():
    """Run anomaly detection on all collected historical data."""
    data_dir = Path("data/extended")
    if not data_dir.exists():
        print("ERROR: No extended data found. Run collect_data.py --extended first.")
        return

    print("=" * 70)
    print("SENTIMENT ANOMALY DETECTION — BACKTEST")
    print(f"Rolling window: {WINDOW} days | Z-threshold: {Z_THRESHOLD}")
    print("=" * 70)

    all_flags = []

    for ticker in TICKERS:
        sent_path = data_dir / ticker / "daily_sentiment.csv"
        price_path = data_dir / ticker / "prices.csv"
        if not sent_path.exists():
            continue

        sent_df = pd.read_csv(sent_path, parse_dates=["date"]).sort_values("date")
        price_df = pd.read_csv(price_path, parse_dates=["date"]).sort_values("date")

        # Merge to get returns alongside sentiment
        merged = sent_df.merge(price_df[["date", "log_return"]], on="date", how="inner")

        z_scores, roll_mean, roll_std = compute_z_scores(
            merged["sentiment_trimmed_mean"], window=WINDOW
        )
        merged["z_score"] = z_scores
        merged["rolling_mean"] = roll_mean
        merged["rolling_std"] = roll_std

        flags = flag_anomalies(merged["date"], merged["z_score"])

        if flags:
            print(f"\n--- {ticker} ({SECTOR[ticker]}) ---")
            print(f"  Total trading days: {len(merged)}")
            print(f"  Anomalies detected: {len(flags)}")

            for date, z in flags:
                row = merged[merged["date"] == date].iloc[0]
                direction = "POSITIVE" if z > 0 else "NEGATIVE"
                next_day = merged[merged["date"] > date]
                next_return = next_day.iloc[0]["log_return"] if len(next_day) > 0 else float("nan")

                flag_entry = {
                    "ticker": ticker,
                    "sector": SECTOR[ticker],
                    "date": date.strftime("%Y-%m-%d"),
                    "sentiment": round(row["sentiment_trimmed_mean"], 4),
                    "rolling_mean": round(row["rolling_mean"], 4),
                    "z_score": round(z, 3),
                    "direction": direction,
                    "articles_that_day": int(row["article_count"]),
                    "same_day_return": round(row["log_return"], 6) if pd.notna(row["log_return"]) else None,
                    "next_day_return": round(next_return, 6) if pd.notna(next_return) else None,
                }
                all_flags.append(flag_entry)

                print(f"\n  [{direction}] {date.strftime('%Y-%m-%d')}  Z = {z:+.3f}")
                print(f"    Sentiment: {row['sentiment_trimmed_mean']:.4f}  "
                      f"(rolling mean: {row['rolling_mean']:.4f})")
                print(f"    Articles: {int(row['article_count'])}")
                print(f"    Same-day return: {row['log_return']:+.4%}" if pd.notna(row["log_return"]) else "    Same-day return: N/A")
                print(f"    Next-day return: {next_return:+.4%}" if pd.notna(next_return) else "    Next-day return: N/A")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_flags)} anomalies across {len(set(f['ticker'] for f in all_flags))} tickers")
    print("=" * 70)

    if all_flags:
        # Analyze: do anomaly days predict next-day returns?
        pos_flags = [f for f in all_flags if f["direction"] == "POSITIVE" and f["next_day_return"] is not None]
        neg_flags = [f for f in all_flags if f["direction"] == "NEGATIVE" and f["next_day_return"] is not None]

        if pos_flags:
            avg_next = np.mean([f["next_day_return"] for f in pos_flags])
            print(f"\n  Positive sentiment spikes ({len(pos_flags)} events):")
            print(f"    Avg next-day return: {avg_next:+.4%}")

        if neg_flags:
            avg_next = np.mean([f["next_day_return"] for f in neg_flags])
            print(f"\n  Negative sentiment spikes ({len(neg_flags)} events):")
            print(f"    Avg next-day return: {avg_next:+.4%}")

        # Save detailed log
        out_path = Path("output")
        out_path.mkdir(exist_ok=True)
        with open(out_path / "anomaly_flags.json", "w") as f:
            json.dump(all_flags, f, indent=2)
        print(f"\n  Detailed log saved to: output/anomaly_flags.json")

    return all_flags


# =============================================================================
# Live mode: poll for new articles and alert in real-time
# =============================================================================
def fetch_today_news(ticker):
    """Fetch today's news for a ticker from Finnhub."""
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": yesterday,
        "to": today,
        "token": FINNHUB_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        time.sleep(5)
        resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def score_articles(articles):
    """Score a list of Finnhub articles with VADER, return mean compound."""
    if not articles:
        return None, 0
    compounds = []
    for art in articles:
        text = (art.get("headline", "") + " " + art.get("summary", "")).strip()
        if text:
            score = analyzer.polarity_scores(text)
            compounds.append(score["compound"])
    if not compounds:
        return None, 0
    # Trimmed mean (10%)
    from scipy.stats import trim_mean
    return trim_mean(compounds, 0.1), len(compounds)


def run_live(interval_minutes=15):
    """Poll for new articles and check for sentiment anomalies."""
    data_dir = Path("data/extended")
    print("=" * 70)
    print("SENTIMENT ANOMALY DETECTION — LIVE MONITOR")
    print(f"Polling every {interval_minutes} minutes")
    print(f"Rolling window: {WINDOW} days | Z-threshold: {Z_THRESHOLD}")
    print(f"Monitoring: {', '.join(TICKERS)}")
    print("=" * 70)

    # Load historical baselines
    baselines = {}
    for ticker in TICKERS:
        sent_path = data_dir / ticker / "daily_sentiment.csv"
        if sent_path.exists():
            df = pd.read_csv(sent_path).sort_values("date")
            sent = df["sentiment_trimmed_mean"]
            if len(sent) >= 5:
                baselines[ticker] = {
                    "mean": sent.tail(WINDOW).mean(),
                    "std": sent.tail(WINDOW).std(),
                    "last_values": sent.tail(WINDOW).tolist(),
                }

    if not baselines:
        print("ERROR: No baseline data. Run collect_data.py --extended first.")
        return

    print(f"\nBaselines loaded for {len(baselines)} tickers. Starting monitor...\n")

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Cycle {cycle}] {now}")
        print("-" * 50)

        alerts = []
        for ticker in TICKERS:
            if ticker not in baselines:
                continue
            try:
                articles = fetch_today_news(ticker)
                sentiment, n_articles = score_articles(articles)
                if sentiment is None:
                    continue

                bl = baselines[ticker]
                if bl["std"] > 0:
                    z = (sentiment - bl["mean"]) / bl["std"]
                else:
                    z = 0.0

                status = "OK"
                if abs(z) > Z_THRESHOLD:
                    direction = "POS SPIKE" if z > 0 else "NEG SPIKE"
                    status = f"*** ALERT: {direction} ***"
                    alerts.append({
                        "ticker": ticker,
                        "sector": SECTOR[ticker],
                        "sentiment": sentiment,
                        "z_score": z,
                        "articles": n_articles,
                        "direction": direction,
                    })

                print(f"  {ticker:5s} | sent={sentiment:+.3f} | "
                      f"Z={z:+.2f} | articles={n_articles:3d} | {status}")

                time.sleep(0.5)  # rate limiting

            except Exception as e:
                print(f"  {ticker:5s} | ERROR: {e}")

        if alerts:
            print(f"\n  *** {len(alerts)} ANOMALIES DETECTED ***")
            for a in alerts:
                print(f"    {a['ticker']} ({a['sector']}): Z={a['z_score']:+.3f}, "
                      f"{a['direction']}, {a['articles']} articles")
            print("  Action: Review flagged tickers for corroborating disclosures")

        print(f"\n  Next check in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    if "--live" in sys.argv:
        interval = 15
        if "--interval" in sys.argv:
            idx = sys.argv.index("--interval")
            interval = int(sys.argv[idx + 1])
        run_live(interval)
    else:
        run_backtest()
