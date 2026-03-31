#!/usr/bin/env python3
"""
Generate publication-quality plots for the 14-ticker sentiment-return study.
Outputs saved to /workspaces/finagent/plots/ at 300 dpi.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/workspaces/finagent/data/extended"
OUT_DIR = "/workspaces/finagent/plots"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "GOOGL", "AMZN", "TSLA", "NVDA", "MSFT",
           "META", "JPM", "JNJ", "XOM", "WMT", "BA", "DIS", "AMD"]

# ── Pre-computed statistics ────────────────────────────────────────────────────
SAME_DAY_RHO = dict(AAPL=0.2176, GOOGL=-0.0457, AMZN=0.1022, TSLA=0.0503,
                     NVDA=0.3618, MSFT=0.1890, META=0.1072, JPM=-0.1865,
                     JNJ=-0.0829, XOM=0.0025, WMT=0.0456, BA=0.1700,
                     DIS=0.3030, AMD=0.3150)

LAG1_RHO = dict(AAPL=-0.1355, GOOGL=-0.1530, AMZN=-0.0339, TSLA=-0.0642,
                NVDA=-0.0507, MSFT=-0.3609, META=-0.1493, JPM=0.1864,
                JNJ=0.0137, XOM=0.2047, WMT=0.0412, BA=0.1024,
                DIS=-0.0250, AMD=-0.3836)

LAG2_RHO = dict(AAPL=-0.2321, GOOGL=0.1205, AMZN=-0.0934, TSLA=-0.2166,
                NVDA=0.0819, MSFT=0.2237, META=0.0706, JPM=-0.3560,
                JNJ=-0.2829, XOM=-0.0889, WMT=-0.0664, BA=-0.4553,
                DIS=-0.0038, AMD=0.1272)

BETA_K = dict(AAPL=0.0099, GOOGL=-0.0140, AMZN=0.0133, TSLA=-0.0290,
              NVDA=0.0166, MSFT=-0.0121, META=0.0191, JPM=-0.0065,
              JNJ=-0.0028, XOM=0.0093, WMT=-0.0047, BA=-0.0046,
              DIS=-0.0042, AMD=-0.0190)

GAMMA_K = dict(AAPL=1.3939, GOOGL=0.9083, AMZN=1.2732, TSLA=2.0447,
               NVDA=1.7180, MSFT=0.9046, META=1.4516, JPM=0.7918,
               JNJ=0.1208, XOM=0.6077, WMT=0.6830, BA=1.0823,
               DIS=1.1243, AMD=1.9503)

GAMMA_SIG = {t: True for t in TICKERS}
GAMMA_SIG["JNJ"] = False  # p = 0.2281

R2 = dict(AAPL=0.8111, GOOGL=0.5477, AMZN=0.8208, TSLA=0.7931,
          NVDA=0.8470, MSFT=0.7201, META=0.8111, JPM=0.8190,
          JNJ=0.0426, XOM=0.5355, WMT=0.5691, BA=0.6745,
          DIS=0.6314, AMD=0.8386)

EVENT_MEAN = dict(AAPL=-0.007306, GOOGL=-0.002278, AMZN=-0.000054,
                  TSLA=0.026449, NVDA=0.007166, MSFT=0.037175,
                  META=0.013893, JPM=-0.001173, JNJ=-0.006122,
                  XOM=-0.006767, WMT=-0.000358, BA=0.005767,
                  DIS=-0.008620, AMD=0.057255)

BASELINE_MEAN = dict(AAPL=0.004404, GOOGL=0.004850, AMZN=0.004908,
                     TSLA=0.006605, NVDA=0.009377, MSFT=0.000716,
                     META=0.005180, JPM=0.006681, JNJ=0.001895,
                     XOM=0.000862, WMT=0.005014, BA=0.011446,
                     DIS=0.009740, AMD=0.001562)

EVENT_SIG = {t: False for t in TICKERS}
EVENT_SIG["MSFT"] = True   # p = 0.0003
EVENT_SIG["AMD"] = True    # p = 0.0251

# ── Sector mapping & colours ──────────────────────────────────────────────────
SECTOR = dict(AAPL="Tech", GOOGL="Tech", AMZN="Tech", MSFT="Tech", META="Tech",
              NVDA="Semiconductors", AMD="Semiconductors",
              TSLA="Automotive", JPM="Finance", JNJ="Healthcare",
              XOM="Energy", WMT="Consumer", BA="Industrials", DIS="Media")

SECTOR_ORDER = ["Tech", "Semiconductors", "Automotive", "Finance",
                "Healthcare", "Energy", "Consumer", "Industrials", "Media"]

SECTOR_COLOR = dict(Tech="#1f77b4", Semiconductors="#9467bd",
                    Automotive="#ff7f0e", Finance="#2ca02c",
                    Healthcare="#d62728", Energy="#e68a00",
                    Consumer="#8c564b", Industrials="#7f7f7f",
                    Media="#e377c2")

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": True,   # needed for dual axis
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "axes.grid": False,
})


def load_data(ticker):
    """Load and merge price + sentiment data for a ticker."""
    prices = pd.read_csv(os.path.join(DATA_DIR, ticker, "prices.csv"), parse_dates=["date"])
    sent = pd.read_csv(os.path.join(DATA_DIR, ticker, "daily_sentiment.csv"), parse_dates=["date"])
    merged = prices.merge(sent, on="date", how="inner")
    return merged


def get_article_count(ticker):
    """Count articles from scored_articles.csv."""
    path = os.path.join(DATA_DIR, ticker, "scored_articles.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return len(df)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Individual sentiment-price overlays
# ══════════════════════════════════════════════════════════════════════════════
def plot_individual_overlays():
    print("Generating individual sentiment-price overlay plots...")
    for ticker in TICKERS:
        df = load_data(ticker)
        if df.empty:
            print(f"  WARNING: No merged data for {ticker}, skipping.")
            continue

        # Remove isolated early points that create long gaps
        # Find the main contiguous block: largest cluster of dates within 7 days of each other
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) > 3:
            gaps = df["date"].diff().dt.days
            big_gap_idx = gaps[gaps > 14].index
            if len(big_gap_idx) > 0:
                # Keep everything after the last big gap
                df = df.iloc[big_gap_idx[-1]:].reset_index(drop=True)

        # Also load full price series for the same date range (for smooth price line)
        prices_full = pd.read_csv(os.path.join(DATA_DIR, ticker, "prices.csv"), parse_dates=["date"])
        prices_full = prices_full[(prices_full["date"] >= df["date"].min()) &
                                  (prices_full["date"] <= df["date"].max())]
        prices_full = prices_full.sort_values("date")

        # Normalise to [0, 1]
        p_min, p_max = prices_full["close"].min(), prices_full["close"].max()
        price_norm = (prices_full["close"] - p_min) / (p_max - p_min + 1e-12)
        sent_norm = (df["sentiment_trimmed_mean"] - df["sentiment_trimmed_mean"].min()) / \
                    (df["sentiment_trimmed_mean"].max() - df["sentiment_trimmed_mean"].min() + 1e-12)

        rho = SAME_DAY_RHO[ticker]

        fig, ax1 = plt.subplots(figsize=(3.5, 2.2))
        ax2 = ax1.twinx()

        ax1.plot(prices_full["date"], price_norm, color="#1f77b4", linewidth=0.9, label="Norm. Close Price")
        ax2.plot(df["date"], sent_norm, color="#d4a017", linewidth=0.8, alpha=0.85, label="Norm. Sentiment")

        ax1.set_ylabel("Normalised Close Price", color="#1f77b4")
        ax2.set_ylabel("Normalised Sentiment", color="#d4a017")
        ax1.set_xlabel("Date")
        ax1.set_title(f"{ticker}  |  Same-Day $\\rho$ = {rho:.4f}", fontsize=9, fontweight="bold")

        ax1.tick_params(axis="y", colors="#1f77b4")
        ax2.tick_params(axis="y", colors="#d4a017")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                   frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=6)

        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"{ticker}_sentiment_price.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Volume (article count) vs association scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_volume_vs_association():
    print("Generating volume vs association scatter...")
    article_counts = {t: get_article_count(t) for t in TICKERS}

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for t in TICKERS:
        x = article_counts[t]
        y = SAME_DAY_RHO[t]
        c = SECTOR_COLOR[SECTOR[t]]
        ax.scatter(x, y, c=c, s=36, zorder=3, edgecolors="white", linewidths=0.4)
        ax.annotate(t, (x, y), fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    # Sector legend
    handles = [Patch(facecolor=SECTOR_COLOR[s], edgecolor="white", label=s)
               for s in SECTOR_ORDER]
    ax.legend(handles=handles, fontsize=5, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="0.8", ncol=2)

    ax.axhline(0, color="0.6", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Total Article Count")
    ax.set_ylabel("Same-Day Pearson $\\rho$")
    ax.set_title("Media Volume vs. Sentiment--Return Association", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "volume_vs_association.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Sector heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_sector_heatmap():
    print("Generating sector heatmap...")
    # Order tickers by sector
    ordered = []
    for s in SECTOR_ORDER:
        for t in TICKERS:
            if SECTOR[t] == s:
                ordered.append(t)

    metrics = ["Same-Day $\\rho$", "Lag-1 $\\rho$", "Lag-2 $\\rho$",
               "$\\beta_k$", "$R^2$"]
    data = np.zeros((len(ordered), len(metrics)))
    for i, t in enumerate(ordered):
        data[i, 0] = SAME_DAY_RHO[t]
        data[i, 1] = LAG1_RHO[t]
        data[i, 2] = LAG2_RHO[t]
        data[i, 3] = BETA_K[t]
        data[i, 4] = R2[t]

    fig, ax = plt.subplots(figsize=(4.0, 4.5))

    # Use diverging colormap for correlation columns, sequential for R2
    # Compromise: use RdBu_r diverging for all
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                   vmin=-0.5, vmax=0.85)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(ordered)))

    # Add sector prefix
    ylabels = []
    prev_sector = None
    for t in ordered:
        s = SECTOR[t]
        if s != prev_sector:
            ylabels.append(f"{t}  ({s})")
            prev_sector = s
        else:
            ylabels.append(t)
    ax.set_yticklabels(ylabels, fontsize=6.5)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if abs(val) > 0.35 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=5.5, color=color)

    # Sector dividers
    row = 0
    for s in SECTOR_ORDER:
        count = sum(1 for t in ordered if SECTOR[t] == s)
        if row > 0:
            ax.axhline(row - 0.5, color="0.3", linewidth=0.8)
        row += count

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Sentiment--Return Metrics by Sector", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "sector_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Market beta (gamma_k) bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_market_betas():
    print("Generating market beta bar chart...")
    # Sort by magnitude
    sorted_tickers = sorted(TICKERS, key=lambda t: abs(GAMMA_K[t]))

    vals = [GAMMA_K[t] for t in sorted_tickers]
    colors = [SECTOR_COLOR[SECTOR[t]] for t in sorted_tickers]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    bars = ax.barh(range(len(sorted_tickers)), vals, color=colors,
                   edgecolor="white", linewidth=0.3, height=0.7)

    labels = []
    for t in sorted_tickers:
        lbl = t
        if GAMMA_SIG[t]:
            lbl += "**"
        else:
            lbl += " (n.s.)"
        labels.append(lbl)

    ax.set_yticks(range(len(sorted_tickers)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Market Beta ($\\gamma_k$)")
    ax.set_title("Market Beta Coefficients (Sorted by |$\\gamma_k$|)", fontsize=9, fontweight="bold")
    ax.axvline(1.0, color="0.5", linewidth=0.5, linestyle="--", label="$\\gamma=1$")

    # Annotate values
    for i, (v, t) in enumerate(zip(vals, sorted_tickers)):
        xoff = 0.03 if v >= 0 else -0.03
        ha = "left" if v >= 0 else "right"
        ax.text(v + xoff, i, f"{v:.3f}", va="center", ha=ha, fontsize=5.5)

    ax.legend(fontsize=6, loc="lower right", frameon=True, edgecolor="0.8")

    # Sector legend
    handles = [Patch(facecolor=SECTOR_COLOR[s], edgecolor="white", label=s)
               for s in SECTOR_ORDER]
    leg2 = ax.legend(handles=handles, fontsize=4.5, loc="upper left",
                     frameon=True, framealpha=0.9, edgecolor="0.8", ncol=2,
                     title="Sector", title_fontsize=5)
    ax.add_artist(leg2)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "market_betas.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Extreme event analysis grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_extreme_events():
    print("Generating extreme events bar chart...")
    sorted_tickers = sorted(TICKERS, key=lambda t: EVENT_MEAN[t] - BASELINE_MEAN[t])

    x = np.arange(len(sorted_tickers))
    width = 0.35

    event_vals = [EVENT_MEAN[t] for t in sorted_tickers]
    base_vals = [BASELINE_MEAN[t] for t in sorted_tickers]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars1 = ax.bar(x - width / 2, event_vals, width, label="Event-Day Mean Return",
                   color="#d62728", edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + width / 2, base_vals, width, label="Baseline Mean Return",
                   color="#1f77b4", edgecolor="white", linewidth=0.3)

    # Highlight significant differences
    for i, t in enumerate(sorted_tickers):
        if EVENT_SIG[t]:
            ymax = max(EVENT_MEAN[t], BASELINE_MEAN[t])
            ax.annotate("*", (i, ymax + 0.003), ha="center", fontsize=10,
                        fontweight="bold", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_tickers, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("Mean Log Return")
    ax.set_title("Extreme Sentiment Event-Day vs. Baseline Returns", fontsize=9, fontweight="bold")
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.legend(fontsize=6, frameon=True, edgecolor="0.8")

    # Note about significance
    ax.text(0.98, 0.02, "* $p < 0.05$", transform=ax.transAxes,
            fontsize=5.5, ha="right", va="bottom", style="italic", color="0.4")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "extreme_events.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Generating publication-quality plots")
    print("=" * 60)
    plot_individual_overlays()
    plot_volume_vs_association()
    plot_sector_heatmap()
    plot_market_betas()
    plot_extreme_events()
    print("=" * 60)
    print(f"All plots saved to {OUT_DIR}/")
    print("=" * 60)
