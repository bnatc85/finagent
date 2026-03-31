"""
Analysis pipeline for sentiment-return study.
Computes all TBD values from main2.tex:
  - Table 1: Lagged sentiment-return correlations
  - Table 2: Market-controlled regression (beta_k, gamma_k, R^2)
  - Table 3: Extreme-sentiment event analysis

Usage: python analyze.py [--extended] [--latex]
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm


def load_ticker_data(data_dir, ticker, market_proxy="SPY"):
    """Load prices and sentiment for a ticker, merge with market returns."""
    ticker_dir = Path(data_dir) / ticker
    spy_dir = Path(data_dir) / market_proxy

    # Load prices
    prices = pd.read_csv(ticker_dir / "prices.csv", parse_dates=["date"])
    prices = prices[["date", "close", "log_return"]].dropna()

    # Load daily sentiment
    sentiment = pd.read_csv(ticker_dir / "daily_sentiment.csv", parse_dates=["date"])

    # Load SPY returns
    spy_prices = pd.read_csv(spy_dir / "prices.csv", parse_dates=["date"])
    spy_prices = spy_prices[["date", "log_return"]].dropna()
    spy_prices.rename(columns={"log_return": "market_return"}, inplace=True)

    # Merge all
    merged = prices.merge(sentiment, on="date", how="inner")
    merged = merged.merge(spy_prices, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def compute_same_day_correlation(df):
    """Pearson correlation between same-day sentiment and return."""
    valid = df.dropna(subset=["sentiment_trimmed_mean", "log_return"])
    if len(valid) < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(valid["sentiment_trimmed_mean"], valid["log_return"])
    return r, p


def compute_lagged_correlations(df):
    """Compute lag-1 and lag-2 sentiment-to-return correlations."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Lag-1: sentiment at t-1 vs return at t
    df["sent_lag1"] = df["sentiment_trimmed_mean"].shift(1)
    df["sent_lag2"] = df["sentiment_trimmed_mean"].shift(2)

    valid1 = df.dropna(subset=["sent_lag1", "log_return"])
    valid2 = df.dropna(subset=["sent_lag2", "log_return"])

    lag1_r, lag1_p = (np.nan, np.nan)
    lag2_r, lag2_p = (np.nan, np.nan)

    if len(valid1) >= 5:
        lag1_r, lag1_p = stats.pearsonr(valid1["sent_lag1"], valid1["log_return"])
    if len(valid2) >= 5:
        lag2_r, lag2_p = stats.pearsonr(valid2["sent_lag2"], valid2["log_return"])

    return lag1_r, lag1_p, lag2_r, lag2_p


def compute_market_controlled_regression(df):
    """
    Estimate: r_{k,t} = alpha + beta * S_{k,t-1} + gamma * r_{m,t} + eps
    Returns beta, gamma, R^2, and their p-values.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["sent_lag1"] = df["sentiment_trimmed_mean"].shift(1)

    valid = df.dropna(subset=["sent_lag1", "log_return", "market_return"])
    if len(valid) < 10:
        return None

    X = valid[["sent_lag1", "market_return"]]
    X = sm.add_constant(X)
    y = valid["log_return"]

    model = sm.OLS(y, X).fit()

    return {
        "alpha": model.params.get("const", np.nan),
        "beta": model.params.get("sent_lag1", np.nan),
        "beta_p": model.pvalues.get("sent_lag1", np.nan),
        "gamma": model.params.get("market_return", np.nan),
        "gamma_p": model.pvalues.get("market_return", np.nan),
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    }


def compute_extreme_sentiment_events(df, tau=1.5):
    """
    Identify extreme sentiment days and compare next-day returns
    to non-event baseline.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    sent = df["sentiment_trimmed_mean"]
    mu = sent.mean()
    sigma = sent.std()

    if sigma == 0 or np.isnan(sigma):
        return None

    df["z_sent"] = (sent - mu) / sigma
    df["next_day_return"] = df["log_return"].shift(-1)

    valid = df.dropna(subset=["z_sent", "next_day_return"])

    events = valid[valid["z_sent"].abs() > tau]
    non_events = valid[valid["z_sent"].abs() <= tau]

    if len(events) < 2:
        # Try lower threshold
        tau = 1.0
        events = valid[valid["z_sent"].abs() > tau]
        non_events = valid[valid["z_sent"].abs() <= tau]

    if len(events) < 2:
        return None

    event_mean = events["next_day_return"].mean()
    baseline_mean = non_events["next_day_return"].mean()

    # t-test for difference (if enough observations)
    t_stat, t_p = np.nan, np.nan
    if len(events) >= 2 and len(non_events) >= 2:
        t_stat, t_p = stats.ttest_ind(events["next_day_return"], non_events["next_day_return"])

    return {
        "tau": tau,
        "n_events": len(events),
        "n_non_events": len(non_events),
        "event_mean_return": event_mean,
        "baseline_mean_return": baseline_mean,
        "t_stat": t_stat,
        "t_p": t_p,
    }


def compute_excess_return_correlation(df):
    """Correlation between sentiment and excess return (r_k - r_m)."""
    df = df.copy()
    df["excess_return"] = df["log_return"] - df["market_return"]
    valid = df.dropna(subset=["sentiment_trimmed_mean", "excess_return"])
    if len(valid) < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(valid["sentiment_trimmed_mean"], valid["excess_return"])
    return r, p


def run_analysis(data_dir, tickers):
    """Run full analysis and print results."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {data_dir}")
    print(f"{'='*80}")

    results = {}

    for ticker in tickers:
        ticker_dir = Path(data_dir) / ticker
        if not (ticker_dir / "daily_sentiment.csv").exists():
            print(f"\n  SKIPPING {ticker}: no sentiment data")
            continue

        print(f"\n--- {ticker} ---")
        df = load_ticker_data(data_dir, ticker)
        print(f"  Merged observations: {len(df)}")

        # Same-day correlation
        rho, rho_p = compute_same_day_correlation(df)
        print(f"  Same-day rho: {rho:.4f} (p={rho_p:.4f})")

        # Excess return correlation
        rho_ex, rho_ex_p = compute_excess_return_correlation(df)
        print(f"  Excess-return rho: {rho_ex:.4f} (p={rho_ex_p:.4f})")

        # Lagged correlations
        lag1_r, lag1_p, lag2_r, lag2_p = compute_lagged_correlations(df)
        print(f"  Lag-1 rho: {lag1_r:.4f} (p={lag1_p:.4f})")
        print(f"  Lag-2 rho: {lag2_r:.4f} (p={lag2_p:.4f})")

        # Market-controlled regression
        reg = compute_market_controlled_regression(df)
        if reg:
            print(f"  Regression: beta={reg['beta']:.6f} (p={reg['beta_p']:.4f}), "
                  f"gamma={reg['gamma']:.4f} (p={reg['gamma_p']:.4f}), "
                  f"R²={reg['r_squared']:.4f}, n={reg['n_obs']}")
        else:
            print(f"  Regression: insufficient data")

        # Extreme sentiment events
        event = compute_extreme_sentiment_events(df)
        if event:
            print(f"  Events: tau={event['tau']:.1f}, n_events={event['n_events']}, "
                  f"event_mean={event['event_mean_return']:.6f}, "
                  f"baseline_mean={event['baseline_mean_return']:.6f}, "
                  f"t_p={event['t_p']:.4f}")
        else:
            print(f"  Events: insufficient extreme days")

        # Article count
        scored_path = ticker_dir / "scored_articles.csv"
        article_count = 0
        if scored_path.exists():
            article_count = len(pd.read_csv(scored_path))

        results[ticker] = {
            "rho": rho, "rho_p": rho_p,
            "rho_ex": rho_ex, "rho_ex_p": rho_ex_p,
            "lag1_r": lag1_r, "lag1_p": lag1_p,
            "lag2_r": lag2_r, "lag2_p": lag2_p,
            "regression": reg,
            "events": event,
            "n_obs": len(df),
            "article_count": article_count,
        }

    # Print LaTeX tables
    if "--latex" in sys.argv:
        print_latex_tables(results, tickers)

    return results


def print_latex_tables(results, tickers):
    """Print LaTeX-formatted tables for the paper."""
    print(f"\n{'='*80}")
    print("LATEX TABLE OUTPUT")
    print(f"{'='*80}")

    # Table 1: Lagged Associations
    print("\n% Table 1: Lagged Sentiment-Return Associations")
    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\caption{Same-Day and Lagged Sentiment--Return Associations}")
    print("\\label{tab:lagged_results}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Stock} & \\textbf{Same-Day $\\rho$} & \\textbf{Lag-1 $\\rho^{(1)}$} & \\textbf{Lag-2 $\\rho^{(2)}$} \\\\")
    print("\\midrule")
    for t in tickers:
        if t not in results:
            continue
        r = results[t]
        rho_str = f"{r['rho']:.2f}"
        l1 = f"{r['lag1_r']:.2f}" if not np.isnan(r['lag1_r']) else "---"
        l2 = f"{r['lag2_r']:.2f}" if not np.isnan(r['lag2_r']) else "---"
        print(f"{t} & {rho_str} & {l1} & {l2} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Table 2: Market-Controlled Regression
    print("\n% Table 2: Lagged Market-Controlled Regression Results")
    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\caption{Lagged Market-Controlled Regression Results}")
    print("\\label{tab:regression_results}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Stock} & \\textbf{$\\beta_k$ on $S_{k,t-1}$} & \\textbf{$\\gamma_k$ on $r_{m,t}$} & \\textbf{$R^2$} \\\\")
    print("\\midrule")
    for t in tickers:
        if t not in results or results[t]["regression"] is None:
            continue
        reg = results[t]["regression"]
        sig_b = "*" if reg["beta_p"] < 0.05 else ""
        sig_g = "*" if reg["gamma_p"] < 0.05 else ""
        print(f"{t} & {reg['beta']:.4f}{sig_b} & {reg['gamma']:.4f}{sig_g} & {reg['r_squared']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Table 3: Extreme-Sentiment Event Analysis
    print("\n% Table 3: Extreme-Sentiment Event Analysis")
    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\caption{Extreme-Sentiment Event Analysis}")
    print("\\label{tab:event_results}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Stock} & \\textbf{Event Threshold} & \\textbf{Mean Next-Day Return (Event)} & \\textbf{Mean Next-Day Return (Baseline)} \\\\")
    print("\\midrule")
    for t in tickers:
        if t not in results or results[t]["events"] is None:
            continue
        ev = results[t]["events"]
        print(f"{t} & $|\\tilde{{S}}| > {ev['tau']:.1f}$ (n={ev['n_events']}) "
              f"& {ev['event_mean_return']:.6f} & {ev['baseline_mean_return']:.6f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    extended = "--extended" in sys.argv

    if extended:
        tickers = ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA",
                    "MSFT", "META", "JPM", "JNJ", "XOM", "WMT", "BA", "DIS", "AMD"]
        run_analysis("data/extended", tickers)
    else:
        tickers = ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA"]
        run_analysis("data/original", tickers)
