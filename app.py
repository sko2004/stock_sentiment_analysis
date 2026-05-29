"""
📈 Stock News Sentiment + Price Movement Predictor
Streamlit Cloud deployment — Python 3.11 compatible
⚠️ Research & Analytics Only. Not financial advice.
"""

# ── Stdlib (always safe) ──────────────────────────────────────────────────────
import os
import sys
import json
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Core scientific (always available) ────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd

# ── Optional heavy imports — deferred & cached ────────────────────────────────
# We import torch/transformers INSIDE cached functions so the app shell
# loads instantly and shows a spinner only when the model is first needed.

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SentimentEdge — Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:      #06080f;
    --card:    #0d1117;
    --card2:   #13191f;
    --border:  #1e2733;
    --up:      #00e5a0;
    --down:    #ff3d5a;
    --neutral: #f59e0b;
    --accent:  #6366f1;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="metric-container"] {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-family: 'DM Mono', monospace !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-weight: 500;
    padding: 8px 18px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
}

.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.hero-header { text-align: center; padding: 28px 0 8px 0; }
.hero-header h1 {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #00e5a0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-header p { color: var(--muted); font-size: 1rem; }

.headline-card {
    background: var(--card2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 5px 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.positive-card { border-left: 4px solid var(--up); }
.negative-card  { border-left: 4px solid var(--down); }
.neutral-card   { border-left: 4px solid var(--neutral); }

.disclaimer-box {
    background: rgba(255,61,90,0.08);
    border: 1px solid rgba(255,61,90,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #fca5a5;
    line-height: 1.6;
}
.info-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    line-height: 1.6;
}
hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TICKER UNIVERSE
# Yahoo Finance ticker symbols for S&P 500 majors, Nifty 50, and Sensex.
# Indian stocks use the .NS suffix (NSE) on Yahoo Finance.
# ══════════════════════════════════════════════════════════════════════════════

TICKER_META = {

    # ── S&P 500 — Technology ──────────────────────────────────────────────────
    "AAPL":  {"name": "Apple Inc.",               "sector": "Technology",          "index": "S&P 500"},
    "MSFT":  {"name": "Microsoft Corp.",           "sector": "Technology",          "index": "S&P 500"},
    "NVDA":  {"name": "Nvidia Corp.",              "sector": "Technology",          "index": "S&P 500"},
    "GOOGL": {"name": "Alphabet Inc. (Class A)",   "sector": "Technology",          "index": "S&P 500"},
    "GOOG":  {"name": "Alphabet Inc. (Class C)",   "sector": "Technology",          "index": "S&P 500"},
    "META":  {"name": "Meta Platforms",            "sector": "Technology",          "index": "S&P 500"},
    "AVGO":  {"name": "Broadcom Inc.",             "sector": "Technology",          "index": "S&P 500"},
    "ORCL":  {"name": "Oracle Corp.",              "sector": "Technology",          "index": "S&P 500"},
    "CRM":   {"name": "Salesforce Inc.",           "sector": "Technology",          "index": "S&P 500"},
    "AMD":   {"name": "Advanced Micro Devices",    "sector": "Technology",          "index": "S&P 500"},
    "INTC":  {"name": "Intel Corp.",               "sector": "Technology",          "index": "S&P 500"},
    "QCOM":  {"name": "Qualcomm Inc.",             "sector": "Technology",          "index": "S&P 500"},
    "IBM":   {"name": "IBM Corp.",                 "sector": "Technology",          "index": "S&P 500"},
    "NOW":   {"name": "ServiceNow Inc.",           "sector": "Technology",          "index": "S&P 500"},
    "ADBE":  {"name": "Adobe Inc.",                "sector": "Technology",          "index": "S&P 500"},
    "TXN":   {"name": "Texas Instruments",         "sector": "Technology",          "index": "S&P 500"},

    # ── S&P 500 — Consumer Discretionary ─────────────────────────────────────
    "AMZN":  {"name": "Amazon.com Inc.",           "sector": "Consumer Disc.",      "index": "S&P 500"},
    "TSLA":  {"name": "Tesla Inc.",                "sector": "Consumer Disc.",      "index": "S&P 500"},
    "HD":    {"name": "Home Depot Inc.",           "sector": "Consumer Disc.",      "index": "S&P 500"},
    "MCD":   {"name": "McDonald's Corp.",          "sector": "Consumer Disc.",      "index": "S&P 500"},
    "NKE":   {"name": "Nike Inc.",                 "sector": "Consumer Disc.",      "index": "S&P 500"},
    "SBUX":  {"name": "Starbucks Corp.",           "sector": "Consumer Disc.",      "index": "S&P 500"},
    "TGT":   {"name": "Target Corp.",              "sector": "Consumer Disc.",      "index": "S&P 500"},
    "LOW":   {"name": "Lowe's Companies",          "sector": "Consumer Disc.",      "index": "S&P 500"},

    # ── S&P 500 — Financials ──────────────────────────────────────────────────
    "JPM":   {"name": "JPMorgan Chase",            "sector": "Financials",          "index": "S&P 500"},
    "V":     {"name": "Visa Inc.",                 "sector": "Financials",          "index": "S&P 500"},
    "MA":    {"name": "Mastercard Inc.",           "sector": "Financials",          "index": "S&P 500"},
    "BAC":   {"name": "Bank of America",           "sector": "Financials",          "index": "S&P 500"},
    "WFC":   {"name": "Wells Fargo & Co.",         "sector": "Financials",          "index": "S&P 500"},
    "GS":    {"name": "Goldman Sachs Group",       "sector": "Financials",          "index": "S&P 500"},
    "MS":    {"name": "Morgan Stanley",            "sector": "Financials",          "index": "S&P 500"},
    "AXP":   {"name": "American Express Co.",      "sector": "Financials",          "index": "S&P 500"},
    "BLK":   {"name": "BlackRock Inc.",            "sector": "Financials",          "index": "S&P 500"},
    "C":     {"name": "Citigroup Inc.",            "sector": "Financials",          "index": "S&P 500"},

    # ── S&P 500 — Healthcare ──────────────────────────────────────────────────
    "LLY":   {"name": "Eli Lilly & Co.",           "sector": "Healthcare",          "index": "S&P 500"},
    "UNH":   {"name": "UnitedHealth Group",        "sector": "Healthcare",          "index": "S&P 500"},
    "JNJ":   {"name": "Johnson & Johnson",         "sector": "Healthcare",          "index": "S&P 500"},
    "ABBV":  {"name": "AbbVie Inc.",               "sector": "Healthcare",          "index": "S&P 500"},
    "MRK":   {"name": "Merck & Co.",               "sector": "Healthcare",          "index": "S&P 500"},
    "PFE":   {"name": "Pfizer Inc.",               "sector": "Healthcare",          "index": "S&P 500"},
    "TMO":   {"name": "Thermo Fisher Scientific",  "sector": "Healthcare",          "index": "S&P 500"},
    "ABT":   {"name": "Abbott Laboratories",       "sector": "Healthcare",          "index": "S&P 500"},

    # ── S&P 500 — Consumer Staples ────────────────────────────────────────────
    "WMT":   {"name": "Walmart Inc.",              "sector": "Consumer Staples",    "index": "S&P 500"},
    "PG":    {"name": "Procter & Gamble",          "sector": "Consumer Staples",    "index": "S&P 500"},
    "KO":    {"name": "Coca-Cola Co.",             "sector": "Consumer Staples",    "index": "S&P 500"},
    "PEP":   {"name": "PepsiCo Inc.",              "sector": "Consumer Staples",    "index": "S&P 500"},
    "COST":  {"name": "Costco Wholesale",          "sector": "Consumer Staples",    "index": "S&P 500"},
    "MDLZ":  {"name": "Mondelez International",    "sector": "Consumer Staples",    "index": "S&P 500"},

    # ── S&P 500 — Energy ──────────────────────────────────────────────────────
    "XOM":   {"name": "Exxon Mobil Corp.",         "sector": "Energy",              "index": "S&P 500"},
    "CVX":   {"name": "Chevron Corp.",             "sector": "Energy",              "index": "S&P 500"},
    "COP":   {"name": "ConocoPhillips",            "sector": "Energy",              "index": "S&P 500"},
    "SLB":   {"name": "SLB (Schlumberger)",        "sector": "Energy",              "index": "S&P 500"},

    # ── S&P 500 — Industrials ─────────────────────────────────────────────────
    "CAT":   {"name": "Caterpillar Inc.",          "sector": "Industrials",         "index": "S&P 500"},
    "BA":    {"name": "Boeing Co.",                "sector": "Industrials",         "index": "S&P 500"},
    "HON":   {"name": "Honeywell International",   "sector": "Industrials",         "index": "S&P 500"},
    "UPS":   {"name": "United Parcel Service",     "sector": "Industrials",         "index": "S&P 500"},
    "GE":    {"name": "GE Aerospace",             "sector": "Industrials",         "index": "S&P 500"},
    "RTX":   {"name": "RTX Corp.",                "sector": "Industrials",         "index": "S&P 500"},
    "LMT":   {"name": "Lockheed Martin Corp.",     "sector": "Industrials",         "index": "S&P 500"},
    "DE":    {"name": "Deere & Company",           "sector": "Industrials",         "index": "S&P 500"},

    # ── S&P 500 — Communication Services ─────────────────────────────────────
    "NFLX":  {"name": "Netflix Inc.",              "sector": "Communication",       "index": "S&P 500"},
    "DIS":   {"name": "Walt Disney Co.",           "sector": "Communication",       "index": "S&P 500"},
    "CMCSA": {"name": "Comcast Corp.",             "sector": "Communication",       "index": "S&P 500"},
    "T":     {"name": "AT&T Inc.",                 "sector": "Communication",       "index": "S&P 500"},
    "VZ":    {"name": "Verizon Communications",    "sector": "Communication",       "index": "S&P 500"},
    "SPOT":  {"name": "Spotify Technology",        "sector": "Communication",       "index": "S&P 500"},

    # ── S&P 500 — Materials & Real Estate ─────────────────────────────────────
    "LIN":   {"name": "Linde plc",                "sector": "Materials",           "index": "S&P 500"},
    "APD":   {"name": "Air Products & Chemicals",  "sector": "Materials",           "index": "S&P 500"},
    "AMT":   {"name": "American Tower Corp.",      "sector": "Real Estate",         "index": "S&P 500"},
    "PLD":   {"name": "Prologis Inc.",             "sector": "Real Estate",         "index": "S&P 500"},

    # ── S&P 500 — Utilities ───────────────────────────────────────────────────
    "NEE":   {"name": "NextEra Energy Inc.",       "sector": "Utilities",           "index": "S&P 500"},
    "DUK":   {"name": "Duke Energy Corp.",         "sector": "Utilities",           "index": "S&P 500"},

    # ══════════════════════════════════════════════════════════════════════════
    # NIFTY 50  (NSE India — .NS suffix on Yahoo Finance)
    # ══════════════════════════════════════════════════════════════════════════

    # IT
    "TCS.NS":        {"name": "Tata Consultancy Services", "sector": "Technology",   "index": "Nifty 50"},
    "INFY.NS":       {"name": "Infosys Ltd.",              "sector": "Technology",   "index": "Nifty 50"},
    "WIPRO.NS":      {"name": "Wipro Ltd.",                "sector": "Technology",   "index": "Nifty 50"},
    "HCLTECH.NS":    {"name": "HCL Technologies",          "sector": "Technology",   "index": "Nifty 50"},
    "TECHM.NS":      {"name": "Tech Mahindra",             "sector": "Technology",   "index": "Nifty 50"},
    "LTI.NS":        {"name": "LTIMindtree",               "sector": "Technology",   "index": "Nifty 50"},

    # Financials
    "HDFCBANK.NS":   {"name": "HDFC Bank Ltd.",            "sector": "Financials",   "index": "Nifty 50"},
    "ICICIBANK.NS":  {"name": "ICICI Bank Ltd.",           "sector": "Financials",   "index": "Nifty 50"},
    "SBIN.NS":       {"name": "State Bank of India",       "sector": "Financials",   "index": "Nifty 50"},
    "KOTAKBANK.NS":  {"name": "Kotak Mahindra Bank",       "sector": "Financials",   "index": "Nifty 50"},
    "AXISBANK.NS":   {"name": "Axis Bank Ltd.",            "sector": "Financials",   "index": "Nifty 50"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance Ltd.",        "sector": "Financials",   "index": "Nifty 50"},
    "BAJAJFINSV.NS": {"name": "Bajaj Finserv Ltd.",        "sector": "Financials",   "index": "Nifty 50"},

    # Energy & Oil
    "RELIANCE.NS":   {"name": "Reliance Industries",       "sector": "Energy",       "index": "Nifty 50"},
    "ONGC.NS":       {"name": "Oil & Natural Gas Corp.",   "sector": "Energy",       "index": "Nifty 50"},
    "BPCL.NS":       {"name": "Bharat Petroleum Corp.",    "sector": "Energy",       "index": "Nifty 50"},
    "POWERGRID.NS":  {"name": "Power Grid Corp.",          "sector": "Utilities",    "index": "Nifty 50"},
    "NTPC.NS":       {"name": "NTPC Ltd.",                 "sector": "Utilities",    "index": "Nifty 50"},
    "ADANIGREEN.NS": {"name": "Adani Green Energy",        "sector": "Utilities",    "index": "Nifty 50"},
    "ADANIPORTS.NS": {"name": "Adani Ports & SEZ",         "sector": "Industrials",  "index": "Nifty 50"},

    # Consumer & FMCG
    "HINDUNILVR.NS": {"name": "Hindustan Unilever",        "sector": "Consumer Staples", "index": "Nifty 50"},
    "ITC.NS":        {"name": "ITC Ltd.",                  "sector": "Consumer Staples", "index": "Nifty 50"},
    "NESTLEIND.NS":  {"name": "Nestle India Ltd.",         "sector": "Consumer Staples", "index": "Nifty 50"},
    "BRITANNIA.NS":  {"name": "Britannia Industries",      "sector": "Consumer Staples", "index": "Nifty 50"},

    # Auto
    "MARUTI.NS":     {"name": "Maruti Suzuki India",       "sector": "Consumer Disc.", "index": "Nifty 50"},
    "TATAMOTORS.NS": {"name": "Tata Motors Ltd.",          "sector": "Consumer Disc.", "index": "Nifty 50"},
    "M&M.NS":        {"name": "Mahindra & Mahindra",       "sector": "Consumer Disc.", "index": "Nifty 50"},
    "HEROMOTOCO.NS": {"name": "Hero MotoCorp",             "sector": "Consumer Disc.", "index": "Nifty 50"},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto Ltd.",           "sector": "Consumer Disc.", "index": "Nifty 50"},
    "EICHERMOT.NS":  {"name": "Eicher Motors Ltd.",        "sector": "Consumer Disc.", "index": "Nifty 50"},

    # Healthcare & Pharma
    "SUNPHARMA.NS":  {"name": "Sun Pharmaceutical",        "sector": "Healthcare",   "index": "Nifty 50"},
    "DRREDDY.NS":    {"name": "Dr. Reddy's Laboratories",  "sector": "Healthcare",   "index": "Nifty 50"},
    "CIPLA.NS":      {"name": "Cipla Ltd.",                "sector": "Healthcare",   "index": "Nifty 50"},
    "DIVISLAB.NS":   {"name": "Divi's Laboratories",       "sector": "Healthcare",   "index": "Nifty 50"},
    "APOLLOHOSP.NS": {"name": "Apollo Hospitals Enterprise","sector": "Healthcare",  "index": "Nifty 50"},

    # Metals & Materials
    "TATASTEEL.NS":  {"name": "Tata Steel Ltd.",           "sector": "Materials",    "index": "Nifty 50"},
    "JSWSTEEL.NS":   {"name": "JSW Steel Ltd.",            "sector": "Materials",    "index": "Nifty 50"},
    "HINDALCO.NS":   {"name": "Hindalco Industries",       "sector": "Materials",    "index": "Nifty 50"},
    "COALINDIA.NS":  {"name": "Coal India Ltd.",           "sector": "Energy",       "index": "Nifty 50"},
    "VEDL.NS":       {"name": "Vedanta Ltd.",              "sector": "Materials",    "index": "Nifty 50"},

    # Industrials & Infra
    "LT.NS":         {"name": "Larsen & Toubro Ltd.",      "sector": "Industrials",  "index": "Nifty 50"},
    "ULTRACEMCO.NS": {"name": "UltraTech Cement",          "sector": "Materials",    "index": "Nifty 50"},
    "GRASIM.NS":     {"name": "Grasim Industries",         "sector": "Industrials",  "index": "Nifty 50"},

    # Telecom & Others
    "BHARTIARTL.NS": {"name": "Bharti Airtel Ltd.",        "sector": "Communication","index": "Nifty 50"},
    "ASIANPAINT.NS": {"name": "Asian Paints Ltd.",         "sector": "Materials",    "index": "Nifty 50"},
    "TITAN.NS":      {"name": "Titan Company Ltd.",        "sector": "Consumer Disc.","index": "Nifty 50"},
    "SHRIRAMFIN.NS": {"name": "Shriram Finance",           "sector": "Financials",   "index": "Nifty 50"},

    # ══════════════════════════════════════════════════════════════════════════
    # SENSEX 30  (BSE India — also available as .NS on Yahoo Finance)
    # All Sensex stocks are also in Nifty 50 above except these extras:
    # ══════════════════════════════════════════════════════════════════════════

    "INDUSINDBK.NS": {"name": "IndusInd Bank",             "sector": "Financials",   "index": "Sensex"},
    "HDFCLIFE.NS":   {"name": "HDFC Life Insurance",       "sector": "Financials",   "index": "Sensex"},
    "TATACONSUM.NS": {"name": "Tata Consumer Products",    "sector": "Consumer Staples","index": "Sensex"},
    "SBILIFE.NS":    {"name": "SBI Life Insurance",        "sector": "Financials",   "index": "Sensex"},

    # ══════════════════════════════════════════════════════════════════════════
    # BONUS — Popular Global Stocks
    # ══════════════════════════════════════════════════════════════════════════
    "TSM":   {"name": "Taiwan Semiconductor (TSMC)", "sector": "Technology",     "index": "Global"},
    "ASML":  {"name": "ASML Holding NV",             "sector": "Technology",     "index": "Global"},
    "SAP":   {"name": "SAP SE",                      "sector": "Technology",     "index": "Global"},
    "BABA":  {"name": "Alibaba Group",               "sector": "Technology",     "index": "Global"},
    "TM":    {"name": "Toyota Motor Corp.",          "sector": "Consumer Disc.", "index": "Global"},
    "SONY":  {"name": "Sony Group Corp.",            "sector": "Technology",     "index": "Global"},
    "HSBC":  {"name": "HSBC Holdings plc",           "sector": "Financials",     "index": "Global"},
    "SHEL":  {"name": "Shell plc",                   "sector": "Energy",         "index": "Global"},
    "UL":    {"name": "Unilever plc",                "sector": "Consumer Staples","index": "Global"},
    "NVO":   {"name": "Novo Nordisk A/S",            "sector": "Healthcare",     "index": "Global"},
    "LVMH":  {"name": "LVMH (MC.PA via ADR)",        "sector": "Consumer Disc.", "index": "Global"},
}

# ── Helper: group tickers by index for sidebar display ───────────────────────
def get_tickers_by_index() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for ticker, meta in TICKER_META.items():
        idx = meta["index"]
        groups.setdefault(idx, []).append(ticker)
    return groups

def get_tickers_by_sector(index_filter: str | None = None) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for ticker, meta in TICKER_META.items():
        if index_filter and meta["index"] != index_filter:
            continue
        sec = meta["sector"]
        groups.setdefault(sec, []).append(ticker)
    return groups


# ── Currency helper ────────────────────────────────────────────────────────────
def get_currency(ticker: str) -> tuple[str, str]:
    """
    Return (symbol, code) for a ticker.
    Indian stocks (.NS) → ₹ INR
    Everything else     → $ USD  (Yahoo Finance converts to USD by default)
    """
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹", "INR"
    return "$", "USD"


def fmt_price(value: float, ticker: str, decimals: int = 2) -> str:
    """Format a price with the correct currency symbol."""
    sym, _ = get_currency(ticker)
    if value >= 1_000:
        return f"{sym}{value:,.{decimals}f}"
    return f"{sym}{value:.{decimals}f}"

COLORS = {
    "up": "#00e5a0", "down": "#ff3d5a", "neutral": "#f59e0b",
    "accent": "#6366f1", "positive": "#00e5a0", "negative": "#ff3d5a",
}

# ══════════════════════════════════════════════════════════════════════════════
# FINBERT — lazy-loaded, cached for the whole session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_finbert():
    """
    Load FinBERT model once per session.
    Cached with st.cache_resource so it survives reruns.
    Returns (tokenizer, model, device) or (None, None, None) on failure.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cpu")   # Streamlit Cloud: CPU only
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        return None, None, str(e)


def run_finbert(texts: list[str]) -> list[dict]:
    """
    Run FinBERT on a list of headline strings.
    Falls back to rule-based scoring if model unavailable.
    """
    tokenizer, model, device = load_finbert()

    if tokenizer is None:
        # Graceful fallback: keyword-based sentiment
        return [_keyword_sentiment(t) for t in texts]

    import torch
    import torch.nn.functional as F

    results = []
    batch_size = 16
    label_names = ["positive", "negative", "neutral"]

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            d = dict(zip(label_names, row.tolist()))
            d["sentiment_score"] = float(d["positive"] - d["negative"])
            d["sentiment_label"] = max(label_names, key=lambda k: d[k])
            results.append(d)
    return results


def _keyword_sentiment(text: str) -> dict:
    """Fast rule-based fallback sentiment when FinBERT is unavailable."""
    pos_words = {"beats", "record", "surge", "strong", "raises", "upgrade",
                 "profit", "growth", "gain", "buyback", "partnership", "rally",
                 "outperform", "exceed", "positive", "expand", "high", "best"}
    neg_words = {"misses", "falls", "probe", "layoffs", "lowers", "downgrade",
                 "recall", "lawsuit", "loss", "decline", "drop", "cut", "weak",
                 "miss", "deficit", "concern", "risk", "low", "warn", "halt"}
    lower = text.lower()
    pos = sum(1 for w in pos_words if w in lower)
    neg = sum(1 for w in neg_words if w in lower)

    if pos > neg:
        label, score = "positive",  min(0.3 + pos * 0.15, 0.95)
    elif neg > pos:
        label, score = "negative", -min(0.3 + neg * 0.15, 0.95)
    else:
        label, score = "neutral", 0.0

    pos_p   = max(0.0, score)
    neg_p   = max(0.0, -score)
    neu_p   = 1.0 - pos_p - neg_p
    return {
        "positive": pos_p, "negative": neg_p, "neutral": neu_p,
        "sentiment_score": score, "sentiment_label": label,
    }

# ══════════════════════════════════════════════════════════════════════════════
# STOCK DATA  — cached per ticker + period
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance and engineer technical features."""
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance not installed. Add it to requirements.txt.")
        return pd.DataFrame()

    end   = datetime.today()
    start = end - timedelta(days=period_days + 60)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["ticker"] = ticker

    # ── Returns & labels ──────────────────────────────────────────────────────
    df["return_1d"]   = df["close"].pct_change()
    df["next_return"] = df["return_1d"].shift(-1)

    def _label(r):
        if pd.isna(r): return np.nan
        return "UP" if r > 0.005 else "DOWN" if r < -0.005 else "NEUTRAL"
    df["label"] = df["next_return"].apply(_label)

    # ── Technical features ────────────────────────────────────────────────────
    for w in [3, 7, 14, 30]:
        df[f"return_{w}d"]      = df["close"].pct_change(w)
        df[f"sma_{w}"]          = df["close"].rolling(w).mean()
        df[f"price_vs_sma_{w}"] = df["close"] / df[f"sma_{w}"] - 1

    df["ema_12"]    = df["close"].ewm(span=12).mean()
    df["ema_26"]    = df["close"].ewm(span=26).mean()
    df["macd"]      = df["ema_12"] - df["ema_26"]
    df["macd_sig"]  = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    for w in [7, 14, 30]:
        df[f"vol_{w}d"] = df["return_1d"].rolling(w).std()

    df["vol_ma10"]    = df["volume"].rolling(10).mean()
    df["vol_ratio"]   = df["volume"] / df["vol_ma10"]

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    bb_ma  = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - (bb_ma - 2*bb_std)) / (4*bb_std + 1e-8)

    for lag in [1, 2, 3]:
        df[f"ret_lag{lag}"] = df["return_1d"].shift(lag)

    return df.tail(period_days + 10).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADLINE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_headlines(ticker: str, n_days: int = 120) -> pd.DataFrame:
    """Generate realistic synthetic financial headlines (no API key needed)."""
    # Derive a short readable name from TICKER_META (strip Ltd., Inc., Corp. etc.)
    full_name = TICKER_META.get(ticker, {}).get("name", ticker)
    name = (full_name
            .replace(" Ltd.", "").replace(" Inc.", "").replace(" Corp.", "")
            .replace(" plc", "").replace(" NV", "").replace(" A/S", "")
            .replace(" Group", "").replace(" Holdings", "")
            .strip())

    positive = [
        f"{name} beats Q{{q}} earnings by {{pct}}%, shares surge after-hours",
        f"{name} reports record ${{rev}}B revenue; analysts raise price targets",
        f"{name} announces ${{bb}}B share buyback program boosting returns",
        f"Strong AI demand lifts {name} margins to {{margin}}% quarterly high",
        f"Analysts upgrade {name} to Strong Buy on accelerating cloud growth",
        f"{name} secures ${{val}}B government contract expanding revenue base",
        f"{name} raises full-year guidance, stock climbs {{pct}}% after-hours",
        f"Institutional investors increase {name} stakes to record levels",
    ]
    negative = [
        f"{name} misses Q{{q}} revenue by {{pct}}% on weakening demand signals",
        f"{name} faces antitrust probe in {{region}}, shares fall sharply",
        f"{name} lowers guidance citing macro headwinds and currency pressure",
        f"Supply chain issues hit {name} margins; analysts slash estimates",
        f"{name} announces {{n}}K layoffs in sweeping cost-restructuring plan",
        f"Analysts downgrade {name} citing stretched valuation and slowing growth",
        f"{name} loses ${{val}}B contract to rival, revenue outlook dims",
        f"Rising rates pressure {name} debt-heavy balance sheet, CFO warns",
    ]
    neutral = [
        f"{name} schedules Q{{q}} earnings call for {{month}} {{day}}",
        f"{name} reiterates full-year guidance at annual investor day",
        f"Analysts maintain Hold rating on {name} ahead of earnings",
        f"{name} confirms dividend of ${{div}} per share next quarter",
        f"{name} CEO speaks at Davos, reiterates long-term AI strategy",
        f"Trading volume in {name} normalizes following recent volatility",
        f"{name} files routine 10-Q with SEC covering quarterly period",
    ]

    rows = []
    for i in range(n_days):
        date = datetime.today() - timedelta(days=n_days - i)
        if date.weekday() >= 5:
            continue
        n_h = random.choices([1, 2, 3], weights=[0.4, 0.45, 0.15])[0]
        for _ in range(n_h):
            stype = random.choices(
                ["positive", "negative", "neutral"],
                weights=[0.35, 0.25, 0.40]
            )[0]
            tmpl = random.choice(
                positive if stype == "positive"
                else negative if stype == "negative"
                else neutral
            )
            try:
                headline = tmpl.format(
                    q=random.randint(1, 4),
                    pct=round(random.uniform(2, 18), 1),
                    rev=round(random.uniform(15, 200), 1),
                    bb=round(random.uniform(5, 60)),
                    val=round(random.uniform(2, 25), 1),
                    margin=round(random.uniform(30, 70), 1),
                    region=random.choice(["EU", "US", "China", "India"]),
                    n=random.choice([1, 2, 5, 10]),
                    month=random.choice(["January", "April", "July", "October"]),
                    day=random.randint(14, 28),
                    div=round(random.uniform(0.1, 2.5), 2),
                )
            except KeyError:
                headline = tmpl
            rows.append({"date": date.date(), "headline": headline,
                         "ticker": ticker, "_true": stype})

    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# DAILY SENTIMENT AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_daily(news_df: pd.DataFrame) -> pd.DataFrame:
    agg = news_df.groupby("date").agg(
        n_headlines=("headline", "count"),
        mean_score=("sentiment_score", "mean"),
        max_score=("sentiment_score", "max"),
        min_score=("sentiment_score", "min"),
        std_score=("sentiment_score", lambda x: x.std() if len(x) > 1 else 0.0),
        pos_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
        neg_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
    ).reset_index()
    agg["balance"] = agg["pos_ratio"] - agg["neg_ratio"]
    agg = agg.sort_values("date")
    for w in [3, 7]:
        agg[f"roll_{w}d"] = agg["mean_score"].rolling(w, min_periods=1).mean()
    for lag in [1, 2]:
        agg[f"lag_{lag}d"] = agg["mean_score"].shift(lag)
    agg["date"] = pd.to_datetime(agg["date"])
    agg["std_score"] = agg["std_score"].fillna(0)
    return agg

# ══════════════════════════════════════════════════════════════════════════════
# ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

SENT_FEATS = ["mean_score", "max_score", "min_score", "std_score",
              "pos_ratio", "neg_ratio", "balance",
              "roll_3d", "roll_7d", "lag_1d", "lag_2d", "n_headlines"]
TECH_FEATS = ["return_1d", "return_3d", "return_7d", "return_14d",
              "ret_lag1", "ret_lag2", "ret_lag3",
              "price_vs_sma_7", "price_vs_sma_14",
              "macd", "macd_hist", "vol_7d", "vol_14d",
              "rsi_14", "bb_pos", "vol_ratio"]


def train_all_models(price_df: pd.DataFrame, sent_agg: pd.DataFrame) -> dict:
    """
    Train ALL three models on the same data split, compute per-model
    probabilities for every test sample, and build an ensemble.

    Returns a dict with:
      models        — {name: {"pipe", "accuracy", "report", "cm",
                              "feat_imp", "y_pred", "y_prob"}}
      ensemble      — {"y_pred", "y_prob", "accuracy", "report", "cm"}
      shared        — {"le", "feats", "merged", "split_idx",
                       "X_test", "y_test"}
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import warnings
    warnings.filterwarnings("ignore")

    # ── Feature matrix ────────────────────────────────────────────────────────
    price_df["date"] = pd.to_datetime(price_df["date"])
    merged = price_df.merge(sent_agg, on="date", how="left")
    merged = (merged.dropna(subset=["label", "return_1d"])
                    .sort_values("date")
                    .reset_index(drop=True))

    feats = [f for f in SENT_FEATS + TECH_FEATS if f in merged.columns]
    X = merged[feats]
    le = LabelEncoder()
    y = le.fit_transform(merged["label"])

    split = int(len(merged) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # ── Define all three classifiers ──────────────────────────────────────────
    clf_defs = {
        "Gradient Boosting": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ]),
        "Logistic Regression": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
    }

    # ── Train each model ──────────────────────────────────────────────────────
    model_colors = {
        "Gradient Boosting": "#6366f1",
        "Random Forest":     "#00e5a0",
        "Logistic Regression": "#f59e0b",
    }
    models_out = {}
    all_probs  = []   # collect for ensemble

    for name, pipe in clf_defs.items():
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_prob  = pipe.predict_proba(X_test)   # (n_test, n_classes)
        acc     = accuracy_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred,
                                        target_names=le.classes_, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)

        clf_step = pipe.named_steps["clf"]
        if hasattr(clf_step, "feature_importances_"):
            fi = pd.Series(clf_step.feature_importances_, index=feats)
        else:
            fi = pd.Series(np.abs(clf_step.coef_).mean(axis=0), index=feats)
        fi = fi.sort_values(ascending=False)

        models_out[name] = {
            "pipe": pipe, "accuracy": acc, "report": report,
            "cm": cm, "feat_imp": fi,
            "y_pred": y_pred, "y_prob": y_prob,
            "color": model_colors[name],
        }
        all_probs.append(y_prob)

    # ── Ensemble: simple average of probabilities (soft voting) ───────────────
    ens_prob  = np.mean(all_probs, axis=0)          # (n_test, n_classes)
    ens_pred  = np.argmax(ens_prob, axis=1)
    ens_acc   = accuracy_score(y_test, ens_pred)
    ens_rep   = classification_report(y_test, ens_pred,
                                      target_names=le.classes_, output_dict=True)
    ens_cm    = confusion_matrix(y_test, ens_pred)

    return {
        "models":   models_out,
        "ensemble": {
            "y_pred": ens_pred, "y_prob": ens_prob,
            "accuracy": ens_acc, "report": ens_rep, "cm": ens_cm,
            "color": "#ffffff",
        },
        "shared": {
            "le": le, "feats": feats, "merged": merged,
            "split_idx": split, "X_test": X_test, "y_test": y_test,
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _theme(fig):
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        xaxis=dict(gridcolor="#1e2733", linecolor="#1e2733"),
        yaxis=dict(gridcolor="#1e2733", linecolor="#1e2733"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2733"),
        margin=dict(l=55, r=25, t=50, b=50),
    )
    return fig


def candlestick_chart(df: pd.DataFrame, ticker: str, sent_agg: pd.DataFrame):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("plotly not installed.")
        return None

    df = df.tail(90).copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=["Price & Moving Averages", "Volume", "RSI-14"],
    )
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=COLORS["up"],
        decreasing_line_color=COLORS["down"],
        increasing_fillcolor=COLORS["up"],
        decreasing_fillcolor=COLORS["down"],
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    for w, color, dash in [(7, "#6366f1", "dash"), (14, "#f59e0b", "dot")]:
        col_n = f"sma_{w}"
        if col_n in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col_n],
                name=f"SMA-{w}", line=dict(color=color, width=1.5, dash=dash),
            ), row=1, col=1)

    # Sentiment markers
    if sent_agg is not None and len(sent_agg):
        sa = sent_agg.copy()
        sa["date"] = pd.to_datetime(sa["date"])
        m = df.merge(sa[["date", "mean_score"]], on="date", how="left")
        pos_d = m[m["mean_score"] > 0.1]
        neg_d = m[m["mean_score"] < -0.1]
        if len(pos_d):
            fig.add_trace(go.Scatter(x=pos_d["date"], y=pos_d["high"] * 1.003,
                mode="markers", marker=dict(symbol="triangle-up", size=7, color=COLORS["up"]),
                name="Positive sentiment"), row=1, col=1)
        if len(neg_d):
            fig.add_trace(go.Scatter(x=neg_d["date"], y=neg_d["low"] * 0.997,
                mode="markers", marker=dict(symbol="triangle-down", size=7, color=COLORS["down"]),
                name="Negative sentiment"), row=1, col=1)

    colors_v = [COLORS["up"] if r >= 0 else COLORS["down"]
                for r in df["return_1d"].fillna(0)]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"],
        marker_color=colors_v, opacity=0.7, showlegend=False), row=2, col=1)

    if "rsi_14" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"],
            line=dict(color="#e879f9", width=1.5), name="RSI-14"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["down"], line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["up"],   line_width=1, row=3, col=1)

    fig.update_layout(
        title=f"<b>{ticker}</b> — Price Dashboard",
        xaxis_rangeslider_visible=False, height=600,
    )
    return _theme(fig)


def sentiment_timeline(sent_agg: pd.DataFrame):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    df = sent_agg.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.05,
                        subplot_titles=["Daily Sentiment Score (FinBERT)", "Headline Count"])

    pos_mask = df["mean_score"] >= 0
    fig.add_trace(go.Bar(x=df.loc[pos_mask, "date"], y=df.loc[pos_mask, "mean_score"],
        marker_color=COLORS["up"], opacity=0.8, name="Positive"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.loc[~pos_mask, "date"], y=df.loc[~pos_mask, "mean_score"],
        marker_color=COLORS["down"], opacity=0.8, name="Negative"), row=1, col=1)

    if "roll_7d" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["roll_7d"],
            line=dict(color=COLORS["neutral"], width=2.5), name="7-Day Avg"), row=1, col=1)

    fig.add_trace(go.Bar(x=df["date"], y=df["n_headlines"],
        marker_color="#6366f1", opacity=0.7, name="Headlines"), row=2, col=1)

    fig.update_layout(height=400, barmode="relative", title="<b>News Sentiment Timeline</b>")
    return _theme(fig)


def feature_importance_chart(fi: pd.Series):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    top = fi.head(18).sort_values()
    colors = [COLORS["up"] if f in SENT_FEATS else "#6366f1" for f in top.index]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
        marker=dict(color=colors), hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title="<b>Feature Importance — Top 18</b>",
                      xaxis_title="Importance", height=480)
    return _theme(fig)


def confusion_matrix_chart(cm: np.ndarray, classes: list):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    cm_pct = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-8)
    text = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.0%})"
             for j in range(len(classes))]
            for i in range(len(classes))]
    fig = go.Figure(go.Heatmap(z=cm_pct, x=classes, y=classes,
        text=text, texttemplate="%{text}", colorscale="Blues",
        showscale=True, hoverongaps=False))
    fig.update_layout(title="<b>Confusion Matrix</b>",
                      xaxis_title="Predicted", yaxis_title="Actual", height=360)
    return _theme(fig)


def prediction_history_chart(merged: pd.DataFrame, split_idx: int,
                              y_pred: np.ndarray, le):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    test = merged.iloc[split_idx:].copy().reset_index(drop=True)
    test["pred_label"] = le.inverse_transform(y_pred)
    test["correct"] = test["label"] == test["pred_label"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=test["date"], y=test["close"],
        line=dict(color="#4b5563", width=1.5), name="Price"), row=1, col=1)
    ok  = test[test["correct"]]
    bad = test[~test["correct"]]
    fig.add_trace(go.Scatter(x=ok["date"], y=ok["close"], mode="markers",
        marker=dict(size=6, color=COLORS["up"], symbol="circle"), name="✓ Correct"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bad["date"], y=bad["close"], mode="markers",
        marker=dict(size=6, color=COLORS["down"], symbol="x"), name="✗ Wrong"), row=1, col=1)

    test["cum_acc"] = test["correct"].cumsum() / (np.arange(len(test)) + 1)
    fig.add_trace(go.Scatter(x=test["date"], y=test["cum_acc"],
        line=dict(color=COLORS["neutral"], width=2), name="Cumulative Acc"), row=2, col=1)
    fig.add_hline(y=1/3, line_dash="dash", line_color="#4b5563", row=2, col=1)

    fig.update_layout(title="<b>Test-Set Predictions vs Price</b>", height=420)
    return _theme(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px;'>
        <div style='font-size:2.4rem;'>📈</div>
        <div style='font-weight:700;font-size:1.25rem;
                    background:linear-gradient(135deg,#6366f1,#00e5a0);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            SentimentEdge
        </div>
        <div style='color:#64748b;font-size:0.77rem;margin-top:4px;'>
            Research Analytics Platform
        </div>
    </div><hr>
    """, unsafe_allow_html=True)

    # ── Index filter → Sector filter → Ticker ────────────────────────────────
    all_indices = ["All"] + sorted({m["index"] for m in TICKER_META.values()})
    selected_index = st.selectbox("🌍 Market / Index", all_indices, index=0)

    # Build filtered sector list
    sectors_in_view = sorted({
        m["sector"] for t, m in TICKER_META.items()
        if selected_index == "All" or m["index"] == selected_index
    })
    selected_sector = st.selectbox("🏭 Sector", ["All"] + sectors_in_view, index=0)

    # Build final ticker list applying both filters
    filtered_tickers = [
        t for t, m in TICKER_META.items()
        if (selected_index == "All" or m["index"] == selected_index)
        and (selected_sector == "All" or m["sector"] == selected_sector)
    ]

    # Show count
    st.caption(f"{len(filtered_tickers)} companies available")

    ticker = st.selectbox(
        "🎯 Company",
        filtered_tickers,
        format_func=lambda x: f"{x} — {TICKER_META[x]['name']}",
    )
    period_days = st.slider("📅 Historical Days", 90, 730, 365, 30)
    show_raw = st.toggle("Show Raw Data Tables", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("""
    <div class='disclaimer-box' style='margin-top:16px;'>
        ⚠️ <b>Research Disclaimer</b><br>
        Educational use only. Not financial advice.
        Do not trade based on model outputs.
    </div>
    <div style='color:#1e2733;font-size:0.7rem;text-align:center;margin-top:12px;'>
        FinBERT · yfinance · scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='hero-header'>
    <h1>📈 SentimentEdge</h1>
    <p>Stock News Sentiment Analysis + Price Movement Research Platform</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME STATE
# ══════════════════════════════════════════════════════════════════════════════

if "done" not in st.session_state:
    st.session_state.done = False

if not run_btn and not st.session_state.done:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, body in [
        (c1, "🤖", "FinBERT NLP",
         "Finance-tuned BERT model scores headlines as positive, negative, or neutral with domain-specific accuracy."),
        (c2, "📊", "Technical Indicators",
         "RSI, MACD, Bollinger Bands, volume momentum, and 20+ rolling features from Yahoo Finance."),
        (c3, "🎯", "ML Classification",
         "Predicts next-day UP / DOWN / NEUTRAL using Gradient Boosting, Random Forest, or Logistic Regression."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#13191f;border:1px solid #1e2733;border-radius:14px;padding:20px;'>
                <div style='font-size:1.8rem;margin-bottom:8px;'>{icon}</div>
                <div style='font-weight:600;font-size:1rem;margin-bottom:6px;'>{title}</div>
                <div style='color:#64748b;font-size:0.84rem;line-height:1.6;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.info("👈 Select a ticker and click **Run Analysis** to begin.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    st.session_state.done = False
    bar = st.progress(0)
    status = st.empty()

    try:
        status.markdown("🤖 **Loading FinBERT model…** (first run ~45 s, then cached)")
        _ = load_finbert()
        bar.progress(15)

        status.markdown(f"📈 **Fetching {ticker} price data from Yahoo Finance…**")
        price_df = fetch_prices(ticker, period_days)
        if price_df.empty:
            st.error(f"No price data found for {ticker}.")
            st.stop()
        bar.progress(30)

        status.markdown("📰 **Generating financial headlines…**")
        news_df = generate_headlines(ticker, n_days=period_days)
        bar.progress(45)

        n_heads = len(news_df)
        status.markdown(f"🔍 **Running FinBERT on {n_heads:,} headlines…**")
        sent_results = run_finbert(news_df["headline"].tolist())
        sent_df = pd.DataFrame(sent_results)
        news_sent = pd.concat([news_df.reset_index(drop=True),
                               sent_df.reset_index(drop=True)], axis=1)
        bar.progress(68)

        status.markdown("🔗 **Aggregating daily sentiment features…**")
        sent_agg = aggregate_daily(news_sent)
        bar.progress(80)

        status.markdown("🧠 **Training all 3 models + ensemble…**")
        all_ml = train_all_models(price_df, sent_agg)
        bar.progress(100)

        st.session_state.update({
            "done": True, "ticker": ticker,
            "price_df": price_df, "news_sent": news_sent,
            "sent_agg": sent_agg, "all_ml": all_ml,
        })
        status.empty(); bar.empty()

    except Exception as e:
        bar.empty(); status.empty()
        st.error(f"Analysis failed: {e}")
        st.exception(e)
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("done"):
    S         = st.session_state
    price_df  = S["price_df"]
    news_sent = S["news_sent"]
    sent_agg  = S["sent_agg"]
    all_ml    = S["all_ml"]
    ticker    = S["ticker"]

    models  = all_ml["models"]
    ens     = all_ml["ensemble"]
    shared  = all_ml["shared"]
    le      = shared["le"]

    # Currency for this ticker
    cur_sym, cur_code = get_currency(ticker)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    last  = price_df.dropna(subset=["close"]).iloc[-1]
    prev  = price_df.dropna(subset=["close"]).iloc[-2]
    chg   = (last["close"] - prev["close"]) / prev["close"] * 100
    latest_sent = sent_agg.sort_values("date").iloc[-1]
    avg_s = float(latest_sent["mean_score"])
    s_lab = "Positive" if avg_s > 0.1 else "Negative" if avg_s < -0.1 else "Neutral"

    best_model_name = max(models, key=lambda k: models[k]["accuracy"])
    best_acc        = models[best_model_name]["accuracy"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{ticker} ({cur_code})",
              fmt_price(last["close"], ticker),
              f"{chg:+.2f}%")
    k2.metric("Best Model Accuracy",
              f"{best_acc:.1%}",
              f"{best_acc - 1/3:+.1%} vs random")
    k3.metric("Ensemble Accuracy",
              f"{ens['accuracy']:.1%}",
              f"{ens['accuracy'] - best_acc:+.1%} vs best single")
    k4.metric("Avg Sentiment",        f"{avg_s:+.3f}", s_lab)
    k5.metric("Headlines Analyzed",   f"{len(news_sent):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📈 Price Dashboard",
        "💬 Sentiment",
        "🤖 All Models",
        "🔍 Live Scorer",
        "📋 Raw Data",
    ])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with t1:
        fig = candlestick_chart(price_df, ticker, sent_agg)
        if fig: st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            try:
                import plotly.graph_objects as go
                ret = price_df["return_1d"].dropna() * 100
                fig_r = go.Figure()
                fig_r.add_trace(go.Histogram(x=ret, nbinsx=60,
                    marker_color="#6366f1", opacity=0.85))
                fig_r.add_vline(x=0, line_dash="dash", line_color="white")
                fig_r.update_layout(title="<b>Daily Returns Distribution</b>",
                                    xaxis_title="Return (%)", height=320)
                st.plotly_chart(_theme(fig_r), use_container_width=True)
            except Exception:
                pass
        with c2:
            try:
                lc = price_df["label"].value_counts()
                fig_p = go.Figure(go.Pie(labels=lc.index, values=lc.values,
                    marker=dict(colors=[COLORS["up"], COLORS["down"], COLORS["neutral"]]),
                    hole=0.45, textinfo="label+percent",
                    textfont=dict(color="white", size=13)))
                fig_p.update_layout(title="<b>Next-Day Movement Labels</b>",
                                    height=320, showlegend=False)
                st.plotly_chart(_theme(fig_p), use_container_width=True)
            except Exception:
                pass

        # Price stats with correct currency
        st.markdown(f"**📐 Key Price Statistics ({cur_code})**")
        ann_vol = price_df["return_1d"].std() * (252 ** 0.5) * 100
        st.dataframe(pd.DataFrame({
            "Metric": ["Latest Close", "52W High", "52W Low",
                       "Mean Close", "Annualised Volatility"],
            "Value":  [
                fmt_price(price_df["close"].iloc[-1], ticker),
                fmt_price(price_df["close"].max(), ticker),
                fmt_price(price_df["close"].min(), ticker),
                fmt_price(price_df["close"].mean(), ticker),
                f"{ann_vol:.1f}%",
            ],
        }), use_container_width=True, hide_index=True)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with t2:
        fig = sentiment_timeline(sent_agg)
        if fig: st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📰 Recent Headlines with FinBERT Scores**")
        recent = news_sent.sort_values("date", ascending=False).head(12)
        for _, row in recent.iterrows():
            lbl   = row["sentiment_label"]
            sc    = float(row["sentiment_score"])
            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[lbl]
            css   = f"{lbl}-card"
            col   = "#00e5a0" if sc > 0 else "#ff3d5a" if sc < 0 else "#f59e0b"
            st.markdown(f"""
            <div class='headline-card {css}'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                    <span style='font-size:0.73rem;color:#64748b;
                                 font-family:"DM Mono",monospace;'>{row['date']}</span>
                    <span style='font-size:0.75rem;font-weight:700;color:{col};
                                 font-family:"DM Mono",monospace;'>
                        {emoji} {lbl.upper()} {sc:+.3f}
                    </span>
                </div>
                <div>{row['headline']}</div>
            </div>""", unsafe_allow_html=True)

        try:
            import plotly.express as px
            price_df["date"] = pd.to_datetime(price_df["date"])
            sc_data = price_df.merge(
                sent_agg[["date", "mean_score"]], on="date", how="inner"
            ).dropna(subset=["next_return", "mean_score", "label"])
            sc_data["next_ret_pct"] = sc_data["next_return"] * 100
            fig_sc = px.scatter(sc_data, x="mean_score", y="next_ret_pct",
                color="label",
                color_discrete_map={"UP": COLORS["up"], "DOWN": COLORS["down"],
                                    "NEUTRAL": COLORS["neutral"]},
                trendline="ols", opacity=0.55,
                labels={"mean_score": "Sentiment Score",
                        "next_ret_pct": "Next-Day Return (%)"},
                title="<b>Sentiment vs Next-Day Return</b>")
            st.plotly_chart(_theme(fig_sc), use_container_width=True)
            corr = sc_data["mean_score"].corr(sc_data["next_return"])
            st.caption(f"Pearson r (sentiment ↔ next-day return): **{corr:.4f}**")
        except Exception:
            pass

    # ── TAB 3 — ALL MODELS ────────────────────────────────────────────────────
    with t3:
        import plotly.graph_objects as go

        # ── 3a: Accuracy comparison bar ───────────────────────────────────────
        st.markdown("### 🏆 Model Accuracy Comparison")
        all_names = list(models.keys()) + ["Ensemble (Avg)"]
        all_accs  = [models[n]["accuracy"] for n in models] + [ens["accuracy"]]
        all_cols  = [models[n]["color"] for n in models] + ["#ffffff"]

        fig_acc = go.Figure(go.Bar(
            x=all_names, y=all_accs,
            marker=dict(color=all_cols, line=dict(color="#0d1117", width=1.5)),
            text=[f"{a:.1%}" for a in all_accs],
            textposition="outside",
            textfont=dict(color="white", size=13, family="DM Mono"),
        ))
        fig_acc.add_hline(y=1/3, line_dash="dash", line_color="#ff3d5a",
                          annotation_text="Random baseline (33.3%)",
                          annotation_font_color="#ff3d5a")
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            title="<b>Test-Set Accuracy — All Models</b>",
            height=320,
        )
        st.plotly_chart(_theme(fig_acc), use_container_width=True)

        # ── 3b: Per-model probability breakdown on test set ───────────────────
        st.markdown("### 📊 Per-Model Class Probability Distribution")
        st.markdown("""
        <div class='info-box'>
        Each bar shows the <b>average predicted probability</b> each model assigned
        to each class across the entire test set. A well-calibrated model should
        assign higher probability to the class that actually occurs most often.
        </div>
        """, unsafe_allow_html=True)

        classes = le.classes_.tolist()   # e.g. ["DOWN", "NEUTRAL", "UP"]
        class_colors_map = {
            "UP": COLORS["up"], "DOWN": COLORS["down"], "NEUTRAL": COLORS["neutral"]
        }
        prob_cols = st.columns(len(models) + 1)

        for col_i, (mname, mdata) in enumerate(
                list(models.items()) + [("Ensemble (Avg)", ens)]):
            y_prob = mdata["y_prob"]
            avg_prob = y_prob.mean(axis=0)   # mean over test samples per class

            with prob_cols[col_i]:
                fig_pb = go.Figure(go.Bar(
                    x=classes,
                    y=avg_prob,
                    marker=dict(
                        color=[class_colors_map.get(c, "#6366f1") for c in classes],
                        line=dict(color="#0d1117", width=1),
                    ),
                    text=[f"{p:.1%}" for p in avg_prob],
                    textposition="outside",
                    textfont=dict(size=11, color="white", family="DM Mono"),
                ))
                acc_val = mdata["accuracy"]
                is_best = mname == best_model_name
                title_str = f"{'⭐ ' if is_best else ''}<b>{mname}</b><br>Acc {acc_val:.1%}"
                fig_pb.update_layout(
                    title=dict(text=title_str, font=dict(size=11)),
                    yaxis=dict(range=[0, 0.85], tickformat=".0%"),
                    height=280,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=60, b=30),
                )
                st.plotly_chart(_theme(fig_pb), use_container_width=True)

        # ── 3c: Side-by-side confusion matrices ───────────────────────────────
        st.markdown("### 🎯 Confusion Matrices — All Models")
        cm_cols = st.columns(len(models) + 1)

        for col_i, (mname, mdata) in enumerate(
                list(models.items()) + [("Ensemble", ens)]):
            with cm_cols[col_i]:
                cm      = mdata["cm"]
                cm_pct  = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-8)
                ann     = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.0%})"
                            for j in range(len(classes))]
                           for i in range(len(classes))]
                fig_cm  = go.Figure(go.Heatmap(
                    z=cm_pct, x=classes, y=classes,
                    text=ann, texttemplate="%{text}",
                    colorscale="Blues", showscale=False,
                ))
                fig_cm.update_layout(
                    title=dict(text=f"<b>{mname}</b><br>{mdata['accuracy']:.1%}",
                               font=dict(size=11)),
                    xaxis_title="Predicted", yaxis_title="Actual",
                    height=280,
                    margin=dict(l=40, r=10, t=60, b=40),
                )
                st.plotly_chart(_theme(fig_cm), use_container_width=True)

        # ── 3d: Feature importance comparison ─────────────────────────────────
        st.markdown("### 🔍 Feature Importance — All Tree-Based Models")
        fi_cols = st.columns(2)
        tree_models = {k: v for k, v in models.items()
                       if k != "Logistic Regression"}
        for col_i, (mname, mdata) in enumerate(tree_models.items()):
            with fi_cols[col_i]:
                fi   = mdata["feat_imp"].head(15).sort_values()
                cols = [COLORS["up"] if f in SENT_FEATS else "#6366f1"
                        for f in fi.index]
                fig_fi = go.Figure(go.Bar(
                    x=fi.values, y=fi.index, orientation="h",
                    marker=dict(color=cols),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig_fi.update_layout(
                    title=f"<b>{mname}</b> — Top 15",
                    xaxis_title="Importance", height=420,
                    margin=dict(l=10, r=10, t=50, b=40),
                )
                st.plotly_chart(_theme(fig_fi), use_container_width=True)

        # Colour legend
        st.markdown("""
        <div style='font-size:0.82rem; color:#64748b; margin-top:4px;'>
        <span style='color:#00e5a0;'>■</span> Sentiment feature &nbsp;&nbsp;
        <span style='color:#6366f1;'>■</span> Technical indicator
        </div>""", unsafe_allow_html=True)

        # ── 3e: Detailed classification reports ───────────────────────────────
        st.markdown("### 📋 Classification Reports")
        for mname, mdata in list(models.items()) + [("Ensemble (Avg)", ens)]:
            with st.expander(
                f"{'⭐ ' if mname == best_model_name else ''}{mname} "
                f"— Accuracy {mdata['accuracy']:.1%}",
                expanded=(mname == best_model_name),
            ):
                rep_rows = []
                for cls in classes:
                    if cls in mdata["report"]:
                        r = mdata["report"][cls]
                        rep_rows.append({
                            "Class":     cls,
                            "Precision": f"{r['precision']:.3f}",
                            "Recall":    f"{r['recall']:.3f}",
                            "F1":        f"{r['f1-score']:.3f}",
                            "Support":   int(r["support"]),
                        })
                st.dataframe(pd.DataFrame(rep_rows),
                             use_container_width=True, hide_index=True)

                # Per-class metrics bar
                fig_rep = go.Figure()
                for metric, color in [("precision", "#6366f1"),
                                      ("recall",    "#00e5a0"),
                                      ("f1-score",  "#f59e0b")]:
                    vals = [mdata["report"].get(c, {}).get(metric, 0) for c in classes]
                    fig_rep.add_trace(go.Bar(name=metric.title(),
                                            x=classes, y=vals,
                                            marker_color=color))
                fig_rep.update_layout(
                    barmode="group", height=280,
                    yaxis=dict(range=[0, 1.05]),
                    title=f"<b>Precision / Recall / F1 — {mname}</b>",
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(_theme(fig_rep), use_container_width=True)

        # ── 3f: Model agreement on each test sample ───────────────────────────
        st.markdown("### 🤝 Model Agreement Analysis")
        st.markdown("""
        <div class='info-box'>
        Shows how often all 3 models agree on the same prediction. When all 3 agree,
        the ensemble prediction is generally more reliable.
        </div>""", unsafe_allow_html=True)

        preds_matrix = np.column_stack([models[n]["y_pred"] for n in models])
        # Count how many models agree with the majority prediction
        from scipy.stats import mode as scipy_mode
        majority_pred, _ = scipy_mode(preds_matrix, axis=1, keepdims=True)
        agree_count = (preds_matrix == majority_pred).sum(axis=1)  # 1, 2 or 3

        agree_df = pd.DataFrame({
            "Agreements": [f"{c}/3 models agree" for c in agree_count],
            "Ensemble Correct": (ens["y_pred"] == shared["y_test"]).astype(int),
        })
        agree_summary = agree_df.groupby("Agreements").agg(
            Count=("Ensemble Correct", "count"),
            Ensemble_Accuracy=("Ensemble Correct", "mean"),
        ).reset_index()
        agree_summary["Ensemble_Accuracy"] = agree_summary["Ensemble_Accuracy"].map("{:.1%}".format)

        a1, a2 = st.columns([1, 2])
        with a1:
            st.dataframe(agree_summary, use_container_width=True, hide_index=True)
        with a2:
            cnt_vals = [int((agree_count == c).sum()) for c in [1, 2, 3]]
            fig_ag = go.Figure(go.Bar(
                x=["1/3 agree", "2/3 agree", "3/3 agree"],
                y=cnt_vals,
                marker_color=["#ff3d5a", "#f59e0b", "#00e5a0"],
                text=cnt_vals, textposition="outside",
                textfont=dict(color="white"),
            ))
            fig_ag.update_layout(title="<b>Model Agreement on Test Samples</b>",
                                 height=260, margin=dict(t=50, b=40))
            st.plotly_chart(_theme(fig_ag), use_container_width=True)

    # ── TAB 4 ─────────────────────────────────────────────────────────────────
    with t4:
        st.markdown("### 🔍 Score Any Headline with FinBERT")
        st.markdown("""
        <div class='info-box'>
        Type or paste any financial headline to see FinBERT sentiment scores,
        then watch all 3 models predict tomorrow's market direction.
        </div>""", unsafe_allow_html=True)

        user_hl = st.text_area("News Headline", height=90,
            placeholder='"Apple beats Q2 earnings by 12%, shares surge after-hours"')

        if st.button("🔬 Analyze", key="live"):
            if user_hl.strip():
                with st.spinner("Running FinBERT…"):
                    res = run_finbert([user_hl.strip()])[0]
                lbl   = res["sentiment_label"]
                sc    = res["sentiment_score"]
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[lbl]
                color = {"positive": COLORS["up"], "negative": COLORS["down"],
                         "neutral": COLORS["neutral"]}[lbl]
                st.markdown(f"""
                <div style='background:#13191f;border:1px solid {color};
                            border-radius:12px;padding:20px;margin-top:12px;'>
                    <div style='font-size:1.5rem;font-weight:700;color:{color};
                                font-family:"DM Mono",monospace;'>
                        {emoji} {lbl.upper()} — Score: {sc:+.4f}
                    </div>
                    <div style='color:#64748b;margin-top:8px;font-size:0.87rem;'>
                        "{user_hl.strip()}"
                    </div>
                </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 Positive", f"{res['positive']:.1%}")
                c2.metric("🔴 Negative", f"{res['negative']:.1%}")
                c3.metric("🟡 Neutral",  f"{res['neutral']:.1%}")
            else:
                st.warning("Please enter a headline.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**📚 Batch Examples**")
        examples = [
            "Microsoft reports record cloud revenue beating estimates by 12%",
            "Tesla faces production halt amid supply chain crisis, shares tumble",
            "Fed signals potential rate cut, markets rally across all sectors",
            "Amazon misses Q3 earnings amid slowing consumer spending trends",
            "Nvidia unveils next-generation AI chips to Wall Street enthusiasm",
            "Apple announces layoffs in some divisions as growth outlook dims",
        ]
        if st.button("🎯 Analyze All Examples"):
            with st.spinner("Analyzing…"):
                batch = run_finbert(examples)
            for hl, r in zip(examples, batch):
                lbl = r["sentiment_label"]
                sc  = r["sentiment_score"]
                css = f"{lbl}-card"
                col = {"positive": COLORS["up"],
                       "negative": COLORS["down"],
                       "neutral":  COLORS["neutral"]}[lbl]
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[lbl]
                st.markdown(f"""
                <div class='headline-card {css}'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div style='flex:1;padding-right:12px;'>{hl}</div>
                        <div style='font-family:"DM Mono",monospace;font-size:0.78rem;
                                    font-weight:700;color:{col};white-space:nowrap;'>
                            {emoji} {lbl.upper()} {sc:+.3f}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 5 ─────────────────────────────────────────────────────────────────
    with t5:
        sub1, sub2, sub3 = st.tabs(["Price Data", "News & Sentiment", "Daily Aggregates"])
        with sub1:
            cols = ["date", "open", "high", "low", "close", "volume",
                    "return_1d", "next_return", "label", "rsi_14", "macd"]
            disp = price_df[[c for c in cols if c in price_df]].copy()
            # Show price in correct currency
            for pc in ["open", "high", "low", "close"]:
                if pc in disp.columns:
                    disp[pc] = disp[pc].round(2)
            st.caption(f"Prices in {cur_code} ({cur_sym})")
            st.dataframe(disp.sort_values("date", ascending=False).head(100),
                         use_container_width=True)
        with sub2:
            cols2 = ["date", "headline", "sentiment_label",
                     "sentiment_score", "positive", "negative", "neutral"]
            st.dataframe(
                news_sent[[c for c in cols2 if c in news_sent]]
                .sort_values("date", ascending=False).head(100).round(4),
                use_container_width=True)
        with sub3:
            st.dataframe(
                sent_agg.sort_values("date", ascending=False).head(100).round(4),
                use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='disclaimer-box'>
        ⚠️ <b>Research & Educational Use Only</b> — Predictions are probabilistic research
        outputs. This is <b>NOT financial advice</b>. Do not trade based on this tool.
        Past patterns do not guarantee future results.
    </div>
    """, unsafe_allow_html=True)