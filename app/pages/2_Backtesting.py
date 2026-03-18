"""
Backtesting Page - Historical Strategy Simulation
===================================================
This page allows users to:
- Select a stock ticker
- Run a simulated trading strategy on historical data
- Compare the ML strategy vs a simple Buy-and-Hold baseline
- View portfolio performance over time
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from strategy import buy_and_sell_strategy, buy_and_hold_baseline, calculate_performance_metrics

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

FEATURE_COLUMNS = [
    "Returns", "SMA_5", "SMA_20", "EMA_12",
    "RSI_14", "Volatility_20", "Volume_Change", "High_Low_Range",
]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def load_processed_data(ticker: str) -> pd.DataFrame:
    """Load processed CSV from the ETL output."""
    filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_model_and_scaler(ticker: str):
    """Load trained model and scaler."""
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)


# ---------------------------------------------------------------------------
# PAGE CONTENT
# ---------------------------------------------------------------------------
st.title("📊 Backtesting — Strategy Simulator")
st.markdown("---")

st.write(
    """
    This page simulates our **Buy-and-Sell trading strategy** on historical data 
    and compares it against a simple **Buy-and-Hold** baseline.
    
    **Strategy rules:**
    - If the model predicts **UP** → **BUY** 1 share
    - If the model predicts **DOWN** → **SELL** 1 share
    - If we can't buy (no cash) or sell (no shares) → **HOLD**
    """
)

st.markdown("---")

# --- Settings ---
col1, col2 = st.columns(2)

with col1:
    selected_ticker = st.selectbox("Select a stock ticker:", TICKERS)

with col2:
    initial_cash = st.number_input(
        "Starting cash ($):", min_value=1000, max_value=100000, value=10000, step=1000
    )

st.markdown("---")

# --- Load Data and Model ---
df = load_processed_data(selected_ticker)
model, scaler = load_model_and_scaler(selected_ticker)

if df.empty:
    st.error(
        f"No processed data found for {selected_ticker}. "
        f"Run the ETL first: `python src/etl.py --ticker {selected_ticker}`"
    )
    st.stop()

if model is None:
    st.error(
        f"No trained model found for {selected_ticker}. "
        f"Train the model first: `python src/model.py --ticker {selected_ticker}`"
    )
    st.stop()

# --- Run Backtest ---
with st.spinner("Running backtest..."):
    # Generate predictions
    X = df[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X)
    predictions = pd.Series(model.predict(X_scaled))

    close_prices = df["Close"].reset_index(drop=True)

    # Run ML strategy
    strategy_results = buy_and_sell_strategy(predictions, close_prices, initial_cash)
    strategy_metrics = calculate_performance_metrics(strategy_results, initial_cash)

    # Run baseline
    baseline_results = buy_and_hold_baseline(close_prices, initial_cash)
    baseline_final = baseline_results["Portfolio_Value"].iloc[-1]
    baseline_return = ((baseline_final - initial_cash) / initial_cash) * 100

# --- Results ---
st.header(f"📈 Results for {selected_ticker}")

# Key metrics
met_col1, met_col2, met_col3, met_col4 = st.columns(4)

with met_col1:
    color = "normal" if strategy_metrics["total_return_pct"] >= 0 else "inverse"
    st.metric(
        label="ML Strategy Return",
        value=f"${strategy_metrics['final_value']:,.2f}",
        delta=f"{strategy_metrics['total_return_pct']:+.2f}%",
    )

with met_col2:
    st.metric(
        label="Buy-and-Hold Return",
        value=f"${baseline_final:,.2f}",
        delta=f"{baseline_return:+.2f}%",
    )

with met_col3:
    diff = strategy_metrics["total_return_pct"] - baseline_return
    winner = "ML Strategy" if diff > 0 else "Buy-and-Hold"
    st.metric(label="Winner", value=winner, delta=f"{diff:+.2f}% difference")

with met_col4:
    st.metric(label="Starting Cash", value=f"${initial_cash:,.2f}")

st.markdown("---")

# Portfolio value over time chart
st.header("💰 Portfolio Value Over Time")

chart_data = pd.DataFrame({
    "ML Strategy": strategy_results["Portfolio_Value"].values,
    "Buy-and-Hold": baseline_results["Portfolio_Value"].values,
})

if "Date" in df.columns:
    chart_data.index = df["Date"].reset_index(drop=True)

st.line_chart(chart_data)

st.markdown("---")

# Trading activity
st.header("📋 Trading Activity")

act_col1, act_col2, act_col3 = st.columns(3)

with act_col1:
    st.metric("Total BUY orders", strategy_metrics["total_buys"])
with act_col2:
    st.metric("Total SELL orders", strategy_metrics["total_sells"])
with act_col3:
    st.metric("Total HOLD days", strategy_metrics["total_holds"])

# Action distribution chart
st.write("**Action Distribution:**")
action_counts = strategy_results["Action"].value_counts()
st.bar_chart(action_counts)

st.markdown("---")

# Detailed results table
st.header("📑 Detailed Trade Log")

with st.expander("View full trade log"):
    display_df = strategy_results.copy()
    if "Date" in df.columns:
        display_df.insert(0, "Date", df["Date"].reset_index(drop=True))
    st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# Additional metrics
st.header("📊 Additional Metrics")

add_col1, add_col2 = st.columns(2)

with add_col1:
    st.write("**ML Strategy**")
    st.write(f"- Max portfolio value: **${strategy_metrics['max_portfolio_value']:,.2f}**")
    st.write(f"- Min portfolio value: **${strategy_metrics['min_portfolio_value']:,.2f}**")
    st.write(f"- Avg daily return: **{strategy_metrics['avg_daily_return_pct']:.4f}%**")

with add_col2:
    st.write("**Buy-and-Hold**")
    bh_max = baseline_results["Portfolio_Value"].max()
    bh_min = baseline_results["Portfolio_Value"].min()
    st.write(f"- Max portfolio value: **${bh_max:,.2f}**")
    st.write(f"- Min portfolio value: **${bh_min:,.2f}**")

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This is a simulated backtest for educational purposes. "
    "Past performance does not guarantee future results. This is not financial advice."
)
