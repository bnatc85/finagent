"""
Sentiment Anomaly Detection Dashboard
--------------------------------------
Live monitoring tool described in the paper:
    Z_{k,t} = (S_{k,t} - mu^(w)) / sigma^(w)

Launch: streamlit run app.py
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import trim_mean

load_dotenv()

# Support both .env (local) and Streamlit secrets (cloud deployment)
def get_secret(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None

FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
_FINNHUB_MISSING = not FINNHUB_API_KEY

TICKERS = ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA", "MSFT",
           "META", "JPM", "JNJ", "XOM", "WMT", "BA", "DIS", "AMD"]

SECTOR = dict(AAPL="Tech", GOOGL="Tech", AMZN="Tech", MSFT="Tech", META="Tech",
              NVDA="Semi", AMD="Semi", TSLA="Auto", JPM="Finance",
              JNJ="Health", XOM="Energy", WMT="Consumer", BA="Industrial", DIS="Media")

SECTOR_COLORS = {
    "Tech": "#1f77b4", "Semi": "#9467bd", "Auto": "#ff7f0e",
    "Finance": "#2ca02c", "Health": "#d62728", "Energy": "#e68a00",
    "Consumer": "#8c564b", "Industrial": "#7f7f7f", "Media": "#e377c2",
}

DATA_DIR = Path("data/extended")

analyzer = SentimentIntensityAnalyzer()


# =============================================================================
# Data helpers
# =============================================================================
@st.cache_data(ttl=3600)
def load_baselines():
    """Load historical sentiment baselines from collected data."""
    baselines = {}
    for ticker in TICKERS:
        sent_path = DATA_DIR / ticker / "daily_sentiment.csv"
        if not sent_path.exists():
            continue
        df = pd.read_csv(sent_path, parse_dates=["date"]).sort_values("date")
        sent = df["sentiment_trimmed_mean"]
        if len(sent) >= 5:
            baselines[ticker] = {
                "history": df,
                "mean": sent.tail(20).mean(),
                "std": sent.tail(20).std(),
            }
    return baselines


@st.cache_data(ttl=3600)
def _load_anomaly_flags():
    """Load backtest anomaly flags if available."""
    path = Path("output/anomaly_flags.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def fetch_news(ticker, days_back=1):
    """Fetch recent news from Finnhub."""
    if not FINNHUB_API_KEY:
        return [], "no_api_key"
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": start, "to": today, "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            return [], "rate_limited"
        if resp.status_code == 401 or resp.status_code == 403:
            return [], f"auth_error_{resp.status_code}"
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return [], f"unexpected_response: {str(data)[:100]}"
        return data, "ok"
    except Exception as e:
        return [], str(e)


def score_articles(articles):
    """Score articles with VADER, return trimmed mean and details."""
    if not articles:
        return None, []
    scored = []
    for art in articles:
        text = (art.get("headline", "") + " " + art.get("summary", "")).strip()
        if not text:
            continue
        scores = analyzer.polarity_scores(text)
        scored.append({
            "headline": art.get("headline", ""),
            "source": art.get("source", ""),
            "datetime": datetime.fromtimestamp(art.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
            "compound": scores["compound"],
            "url": art.get("url", ""),
        })
    if not scored:
        return None, []
    compounds = [s["compound"] for s in scored]
    mean_sent = trim_mean(compounds, 0.1)
    return mean_sent, scored


def compute_z(sentiment, baseline):
    """Compute Z-score against baseline."""
    if baseline["std"] > 0 and sentiment is not None:
        return (sentiment - baseline["mean"]) / baseline["std"]
    return 0.0


# =============================================================================
# Streamlit App
# =============================================================================
st.set_page_config(
    page_title="Sentiment Anomaly Monitor",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Sentiment Anomaly Detection System")
st.caption("Real-time monitoring tool — flags unusual sentiment deviations for analyst review")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.25,
                            help="Flag when |Z| exceeds this value")
    days_back = st.selectbox("News Lookback", [1, 2, 3, 5], index=0,
                             help="How many days of news to fetch")
    selected_tickers = st.multiselect("Tickers", TICKERS, default=TICKERS)

    st.divider()
    st.markdown("**Formula:**")
    st.latex(r"Z_{k,t} = \frac{S_{k,t} - \mu_{k,t}^{(w)}}{\sigma_{k,t}^{(w)}}")
    with st.expander("What does this mean?"):
        st.markdown("""
        **Z** = the anomaly score (how unusual today's sentiment is)

        | Variable | Meaning |
        |----------|---------|
        | **Z_{k,t}** | The anomaly score for stock *k* on day *t*. Values above +2 or below -2 are flagged. |
        | **S_{k,t}** | Today's sentiment score — the average emotional tone of all news articles about this stock today, measured on a scale from -1 (very negative) to +1 (very positive). |
        | **μ_{k,t}^(w)** | The rolling average sentiment over the past *w* trading days (default: 20). This is what "normal" looks like for this stock. |
        | **σ_{k,t}^(w)** | The rolling standard deviation — how much sentiment typically varies day-to-day. A stock with wild media swings has a high σ. |
        | **w** | The lookback window (20 trading days ≈ 1 month). |

        **In plain English:** Take today's sentiment, subtract the recent average,
        and divide by how much sentiment normally varies. If the result is bigger
        than 2 (or smaller than -2), the news tone is far outside the normal
        range and worth investigating.

        **Example:** If a stock normally has sentiment around +0.20 (mildly positive)
        with day-to-day swings of about 0.10, and today's sentiment is -0.15,
        the Z-score would be (-0.15 - 0.20) / 0.10 = **-3.5** — a major negative
        anomaly.
        """)

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    scan_button = st.button("🔍 Scan Now", type="primary", use_container_width=True)

# Load baselines
baselines = load_baselines()

# API key warning
if _FINNHUB_MISSING:
    st.error("**Finnhub API key not found.** Live scanning will not work. "
             "Add `FINNHUB_API_KEY` to your Streamlit secrets "
             "(Settings > Secrets) or `.env` file.")

if not baselines:
    st.warning("No baseline data found. The monitor will operate without historical baselines "
               "and use the current scan window to establish them.")
    st.info("For full functionality, run `python collect_data.py --extended` locally first, "
            "then commit the `data/extended/` folder to the repo.")
    # Set empty baselines — live scan will still work but Z-scores won't be meaningful
    # until enough data accumulates

# Main scan logic
if scan_button or auto_refresh:
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Status bar
    progress = st.progress(0, text="Scanning...")
    alerts = []
    results = []

    for i, ticker in enumerate(selected_tickers):
        progress.progress((i + 1) / len(selected_tickers), text=f"Scanning {ticker}...")

        if ticker not in baselines:
            results.append({"Ticker": ticker, "Sector": SECTOR[ticker],
                           "Status": "⚠️ No baseline", "Sentiment": None,
                           "Z-Score": None, "Articles": 0})
            continue

        articles, status = fetch_news(ticker, days_back)
        if status == "no_api_key":
            results.append({"Ticker": ticker, "Sector": SECTOR[ticker],
                           "Status": "🔑 No API key", "Sentiment": None,
                           "Z-Score": None, "Articles": 0})
            continue
        if status == "rate_limited":
            results.append({"Ticker": ticker, "Sector": SECTOR[ticker],
                           "Status": "⏳ Rate limited", "Sentiment": None,
                           "Z-Score": None, "Articles": 0})
            time.sleep(2)
            continue
        if status not in ("ok",):
            results.append({"Ticker": ticker, "Sector": SECTOR[ticker],
                           "Status": f"❌ {status}", "Sentiment": None,
                           "Z-Score": None, "Articles": 0})
            continue

        sentiment, scored_details = score_articles(articles)
        if sentiment is None:
            results.append({"Ticker": ticker, "Sector": SECTOR[ticker],
                           "Status": f"— No articles ({len(articles)} raw)", "Sentiment": None,
                           "Z-Score": None, "Articles": 0})
            continue

        z = compute_z(sentiment, baselines[ticker])
        n_articles = len(scored_details)

        if abs(z) > z_threshold:
            direction = "🔴 POS SPIKE" if z > 0 else "🔵 NEG SPIKE"
            alerts.append({
                "ticker": ticker,
                "sector": SECTOR[ticker],
                "sentiment": sentiment,
                "z": z,
                "articles": n_articles,
                "direction": direction,
                "details": scored_details,
                "baseline_mean": baselines[ticker]["mean"],
            })
            status_str = direction
        elif abs(z) > z_threshold * 0.75:
            status_str = "⚠️ Elevated"
        else:
            status_str = "✅ Normal"

        results.append({
            "Ticker": ticker,
            "Sector": SECTOR[ticker],
            "Status": status_str,
            "Sentiment": round(sentiment, 4),
            "Z-Score": round(z, 2),
            "Articles": n_articles,
        })

        time.sleep(0.5)

    progress.empty()

    # Alert banner
    if alerts:
        st.error(f"🚨 **{len(alerts)} ANOMALIES DETECTED** — {scan_time}")
        st.caption("These tickers show sentiment deviations exceeding the threshold. "
                   "Review for corroborating disclosures before acting.")
    else:
        st.success(f"✅ No anomalies detected — {scan_time}")

    # Results table
    st.subheader("Scan Results")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True, hide_index=True,
                 column_config={
                     "Z-Score": st.column_config.NumberColumn(format="%.2f"),
                     "Sentiment": st.column_config.NumberColumn(format="%.4f"),
                 })

    # Z-score bar chart
    chart_data = df_results.dropna(subset=["Z-Score"]).copy()
    if not chart_data.empty:
        st.subheader("Z-Score Overview")

        chart_data = chart_data.set_index("Ticker")
        colors = ["#d62728" if abs(z) > z_threshold else "#ff9800" if abs(z) > z_threshold * 0.75
                  else "#2ca02c" for z in chart_data["Z-Score"]]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.bar(chart_data.index, chart_data["Z-Score"], color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(z_threshold, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7, label=f"+{z_threshold}")
        ax.axhline(-z_threshold, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7, label=f"-{z_threshold}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Z-Score")
        ax.set_title("Current Sentiment Z-Scores", fontweight="bold")
        ax.legend(fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Alert details
    if alerts:
        st.subheader("Anomaly Details")
        for alert in alerts:
            with st.expander(
                f"{alert['direction']}  **{alert['ticker']}** ({alert['sector']}) — "
                f"Z = {alert['z']:+.2f} | Sentiment = {alert['sentiment']:.4f} | "
                f"{alert['articles']} articles"
            ):
                st.markdown(f"**Baseline mean:** {alert['baseline_mean']:.4f}")
                st.markdown(f"**Current sentiment:** {alert['sentiment']:.4f}")
                st.markdown(f"**Deviation:** {abs(alert['z']):.2f} standard deviations from baseline")

                st.markdown("---")
                st.markdown("**Recent Articles (sorted by |sentiment|):**")
                details_df = pd.DataFrame(alert["details"])
                details_df = details_df.sort_values("compound", key=abs, ascending=False)
                for _, row in details_df.head(10).iterrows():
                    emoji = "🟢" if row["compound"] > 0.05 else "🔴" if row["compound"] < -0.05 else "⚪"
                    st.markdown(
                        f"{emoji} **{row['compound']:+.3f}** | {row['source']} | "
                        f"{row['datetime']}\n> {row['headline']}"
                    )

    # Historical context
    st.subheader("Historical Sentiment Baselines")
    tabs = st.tabs(selected_tickers)

    # Load historical anomaly flags for spike commentary
    anomaly_flags = _load_anomaly_flags()

    for tab, ticker in zip(tabs, selected_tickers):
        with tab:
            if ticker in baselines:
                hist = baselines[ticker]["history"].copy()
                hist = hist.sort_values("date")

                # Load price data for this ticker
                price_path = DATA_DIR / ticker / "prices.csv"
                prices = None
                if price_path.exists():
                    prices = pd.read_csv(price_path, parse_dates=["date"]).sort_values("date")
                    # Filter prices to sentiment date range
                    prices = prices[(prices["date"] >= hist["date"].min()) &
                                    (prices["date"] <= hist["date"].max())]

                import matplotlib.pyplot as plt

                # Combined chart: price + sentiment + anomaly bands
                if prices is not None and not prices.empty:
                    fig, (ax_price, ax_sent) = plt.subplots(2, 1, figsize=(8, 4.5),
                                                             sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})

                    # Top: Stock price
                    ax_price.plot(prices["date"], prices["close"], color="#1f77b4", linewidth=1.2)
                    ax_price.fill_between(prices["date"], prices["close"], alpha=0.08, color="#1f77b4")
                    ax_price.set_ylabel("Close Price ($)")
                    ax_price.set_title(f"{ticker} — Stock Price & Sentiment", fontsize=10, fontweight="bold")
                    ax_price.grid(axis="y", alpha=0.2)

                    # Mark spike days on price chart
                    ticker_flags_for_marks = [f for f in _load_anomaly_flags() if f["ticker"] == ticker]
                    for flag in ticker_flags_for_marks:
                        flag_date = pd.Timestamp(flag["date"])
                        price_on_day = prices[prices["date"] == flag_date]
                        if not price_on_day.empty:
                            color = "#d62728" if flag["direction"] == "NEGATIVE" else "#2ca02c"
                            marker = "v" if flag["direction"] == "NEGATIVE" else "^"
                            ax_price.scatter(flag_date, price_on_day["close"].iloc[0],
                                           color=color, marker=marker, s=60, zorder=5,
                                           edgecolors="white", linewidths=0.5)

                    # Bottom: Sentiment with anomaly bands
                    ax_sent.plot(hist["date"], hist["sentiment_trimmed_mean"],
                               color="#d4a017", linewidth=1)
                    ax_sent.axhline(baselines[ticker]["mean"], color="green",
                                  linestyle="--", linewidth=0.7, label=f"Mean ({baselines[ticker]['mean']:.3f})")
                    ax_sent.axhline(baselines[ticker]["mean"] + z_threshold * baselines[ticker]["std"],
                                  color="red", linestyle=":", linewidth=0.7, label=f"+{z_threshold}σ")
                    ax_sent.axhline(baselines[ticker]["mean"] - z_threshold * baselines[ticker]["std"],
                                  color="red", linestyle=":", linewidth=0.7, label=f"-{z_threshold}σ")
                    ax_sent.fill_between(hist["date"],
                                   baselines[ticker]["mean"] - z_threshold * baselines[ticker]["std"],
                                   baselines[ticker]["mean"] + z_threshold * baselines[ticker]["std"],
                                   alpha=0.1, color="green")
                    ax_sent.set_ylabel("Sentiment")
                    ax_sent.set_xlabel("Date")
                    ax_sent.legend(fontsize=7, loc="upper left")
                    ax_sent.grid(axis="y", alpha=0.2)

                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption("Top: closing price with spike markers (▲ positive spike, ▼ negative spike). "
                              "Bottom: daily sentiment with anomaly bands (green = normal range).")
                else:
                    # Fallback: sentiment only
                    fig, ax = plt.subplots(figsize=(8, 2.5))
                    ax.plot(hist["date"], hist["sentiment_trimmed_mean"],
                           color="#1f77b4", linewidth=1)
                    ax.axhline(baselines[ticker]["mean"], color="green",
                              linestyle="--", linewidth=0.7, label=f"Mean ({baselines[ticker]['mean']:.3f})")
                    ax.axhline(baselines[ticker]["mean"] + z_threshold * baselines[ticker]["std"],
                              color="red", linestyle=":", linewidth=0.7, label=f"+{z_threshold}σ")
                    ax.axhline(baselines[ticker]["mean"] - z_threshold * baselines[ticker]["std"],
                              color="red", linestyle=":", linewidth=0.7, label=f"-{z_threshold}σ")
                    ax.fill_between(hist["date"],
                                   baselines[ticker]["mean"] - z_threshold * baselines[ticker]["std"],
                                   baselines[ticker]["mean"] + z_threshold * baselines[ticker]["std"],
                                   alpha=0.1, color="green")
                    ax.set_ylabel("Sentiment")
                    ax.set_title(f"{ticker} — Historical Sentiment with Anomaly Bands", fontsize=10)
                    ax.legend(fontsize=7, loc="upper left")
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # --- Spike history commentary ---
                ticker_flags = [f for f in anomaly_flags if f["ticker"] == ticker]
                if ticker_flags:
                    st.markdown("---")
                    st.markdown(f"#### What historically happens when {ticker} has a sentiment spike?")

                    pos_flags = [f for f in ticker_flags if f["direction"] == "POSITIVE"
                                 and f.get("next_day_return") is not None]
                    neg_flags = [f for f in ticker_flags if f["direction"] == "NEGATIVE"
                                 and f.get("next_day_return") is not None]

                    # Positive spikes
                    if pos_flags:
                        avg_next = np.mean([f["next_day_return"] for f in pos_flags])
                        avg_same = np.mean([f["same_day_return"] for f in pos_flags if f.get("same_day_return") is not None])
                        direction_word = "rose" if avg_next > 0 else "fell"
                        st.markdown(
                            f"**Unusually positive news** ({len(pos_flags)} events observed): "
                            f"When media coverage turned sharply optimistic, the stock {direction_word} "
                            f"an average of **{abs(avg_next)*100:.2f}%** the next trading day. "
                            f"On the spike day itself, the stock moved **{avg_same*100:+.2f}%** on average."
                        )
                        for f in pos_flags:
                            ndr = f["next_day_return"]
                            sdr = f.get("same_day_return")
                            emoji = "📈" if ndr and ndr > 0 else "📉"
                            st.caption(
                                f"{emoji} **{f['date']}** — Sentiment surged to {f['sentiment']:.3f} "
                                f"(Z = {f['z_score']:+.2f}) on {f['articles_that_day']} articles. "
                                f"Same-day: {sdr*100:+.2f}%, Next-day: {ndr*100:+.2f}%"
                            )

                    # Negative spikes
                    if neg_flags:
                        avg_next = np.mean([f["next_day_return"] for f in neg_flags])
                        avg_same = np.mean([f["same_day_return"] for f in neg_flags if f.get("same_day_return") is not None])
                        direction_word = "bounced back" if avg_next > 0 else "continued falling"
                        st.markdown(
                            f"**Unusually negative news** ({len(neg_flags)} events observed): "
                            f"When media coverage turned sharply pessimistic, the stock {direction_word} "
                            f"an average of **{abs(avg_next)*100:.2f}%** the next trading day. "
                            f"On the spike day itself, the stock moved **{avg_same*100:+.2f}%** on average."
                        )
                        for f in neg_flags:
                            ndr = f["next_day_return"]
                            sdr = f.get("same_day_return")
                            emoji = "📈" if ndr and ndr > 0 else "📉"
                            st.caption(
                                f"{emoji} **{f['date']}** — Sentiment dropped to {f['sentiment']:.3f} "
                                f"(Z = {f['z_score']:+.2f}) on {f['articles_that_day']} articles. "
                                f"Same-day: {sdr*100:+.2f}%, Next-day: {ndr*100:+.2f}%"
                            )

                    # Overall takeaway
                    all_next = [f["next_day_return"] for f in ticker_flags if f.get("next_day_return") is not None]
                    if all_next:
                        avg_all = np.mean(all_next)
                        reversal_pct = sum(1 for f in ticker_flags
                                          if f.get("next_day_return") is not None
                                          and f["direction"] == "NEGATIVE" and f["next_day_return"] > 0
                                          or f["direction"] == "POSITIVE" and f["next_day_return"] < 0) / len(all_next) * 100

                        st.info(
                            f"**Bottom line for {ticker}:** Across {len(all_next)} historical spikes, "
                            f"the price reversed direction **{reversal_pct:.0f}%** of the time the next day. "
                            f"This {'suggests a reversal tendency — spikes may be overreactions.' if reversal_pct > 55 else 'suggests spikes sometimes carry momentum — exercise caution.' if reversal_pct < 45 else 'is roughly a coin flip — spikes alone are not reliable predictors.'}"
                        )

                else:
                    st.markdown("---")
                    st.info(f"No historical sentiment spikes detected for {ticker} at the current threshold. "
                            f"This stock's media coverage tends to stay within normal bounds.")

    if auto_refresh:
        time.sleep(60)
        st.rerun()

else:
    # Landing state — layman's explanation
    st.markdown("""
    ### What is this?

    This tool watches the **emotional tone of news articles** about major stocks and
    alerts you when that tone becomes unusually positive or negative compared to
    recent history.

    **Why does that matter?** Research shows that when media coverage suddenly shifts
    — for example, a flood of alarming headlines about a company — stock prices can
    move in the short term, even when nothing fundamental has changed about the
    business. This creates two risks:

    - **For investors:** You might make impulsive buy/sell decisions based on
      emotional headlines rather than real company performance.
    - **For markets:** Bad actors could deliberately flood media channels with
      extreme sentiment to manipulate stock prices.

    ### How it works

    1. **We track a baseline** — what "normal" news sentiment looks like for each
       stock over the past 20 trading days.
    2. **We measure deviations** — when today's sentiment is far outside that
       normal range, we calculate a Z-score (how many standard deviations away
       from normal).
    3. **We flag anomalies** — if the Z-score exceeds the threshold (default: 2.0),
       it means the emotional tone of coverage has shifted dramatically and may
       warrant closer attention.

    **A flag is not a trading signal.** It's a prompt to ask: *"Is there a real
    reason for this shift, or is the media just running hot?"*

    ---

    *Click **Scan Now** in the sidebar to check current conditions, or enable
    **Auto-refresh** for continuous monitoring.*
    """)

    # Show baseline summary
    st.subheader("Current Baselines")
    st.caption("These are the \"normal\" sentiment levels for each stock, based on "
               "recent history. The bands show where we'd expect sentiment to fall "
               "on a typical day.")
    baseline_rows = []
    for ticker in TICKERS:
        if ticker in baselines:
            bl = baselines[ticker]
            baseline_rows.append({
                "Ticker": ticker,
                "Sector": SECTOR[ticker],
                "Normal Sentiment": round(bl["mean"], 4),
                "Variability (Std)": round(bl["std"], 4),
                "Spike Threshold (+)": round(bl["mean"] + 2 * bl["std"], 4),
                "Spike Threshold (-)": round(bl["mean"] - 2 * bl["std"], 4),
                "Days Tracked": len(bl["history"]),
            })
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)

    # Historical spike summary
    anomaly_flags = _load_anomaly_flags()
    if anomaly_flags:
        st.subheader("What happens after a spike?")
        st.caption("Based on historical backtest data — when sentiment spiked in the past, "
                   "here's what the stock did the next day.")

        pos_flags = [f for f in anomaly_flags if f["direction"] == "POSITIVE"
                     and f.get("next_day_return") is not None]
        neg_flags = [f for f in anomaly_flags if f["direction"] == "NEGATIVE"
                     and f.get("next_day_return") is not None]

        col1, col2 = st.columns(2)
        with col1:
            if pos_flags:
                avg = np.mean([f["next_day_return"] for f in pos_flags]) * 100
                st.metric("After positive spikes", f"{avg:+.2f}% avg next-day",
                          delta="Tends to reverse" if avg < 0 else "Tends to continue",
                          delta_color="inverse" if avg < 0 else "normal")
        with col2:
            if neg_flags:
                avg = np.mean([f["next_day_return"] for f in neg_flags]) * 100
                st.metric("After negative spikes", f"{avg:+.2f}% avg next-day",
                          delta="Tends to bounce back" if avg > 0 else "Tends to continue down",
                          delta_color="normal" if avg > 0 else "inverse")

        st.caption(f"Based on {len(anomaly_flags)} historical anomaly events across "
                   f"{len(set(f['ticker'] for f in anomaly_flags))} stocks.")
