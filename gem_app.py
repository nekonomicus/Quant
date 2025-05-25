# gem_app.py
from __future__ import annotations
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# Try importing Streamlit and set a flag
try:
    import streamlit as st
    _STREAMLIT_IMPORTED = True
except ImportError:
    _STREAMLIT_IMPORTED = False

# --- Constants ---
EQUITY_US, EQUITY_INTL, BONDS = "SPY", "VEU", "BND"
TICKERS = [EQUITY_US, EQUITY_INTL, BONDS]
LOOKBACK_MONTHS = 12
START_DATE = "2005-01-01"

# --- Core Logic Functions ---
def load_prices(s: str = START_DATE, e: str | None = None) -> pd.DataFrame:
    """Loads and resamples price data."""
    end_date = e or datetime.today().strftime("%Y-%m-%d")
    # Use auto_adjust=True so 'Close' is already adjusted for splits/dividends
    # Select 'Close' to get a DataFrame with tickers as columns and adjusted close prices as values
    prices_data = yf.download(TICKERS, start=s, end=end_date, progress=False, auto_adjust=True)
    if prices_data.empty:
        raise ValueError("Failed to download price data. Check tickers, date range, and internet connection.")
    
    prices = prices_data['Close']
    if prices.empty or prices.isnull().all().all(): # Check if the 'Close' DataFrame is empty or all NaNs
         raise ValueError("No 'Close' price data found after download. Check ticker validity and yfinance API.")

    return prices.resample("M").last().dropna(how="any")

def trailing_return(px: pd.DataFrame, m: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """Calculates trailing returns."""
    return px.pct_change(m)

def build_signals(prices: pd.DataFrame, lookback: int = LOOKBACK_MONTHS) -> pd.Series:
    """Builds allocation signals based on momentum."""
    if prices.empty:
        return pd.Series(dtype='object', name="Allocation") # Return empty series if no price data

    if len(prices.index) < lookback + 1: # Need at least lookback+1 data points for pct_change(lookback)
         # Not enough data to compute initial momentum after resample & dropna
         return pd.Series(np.nan, index=prices.index, name="Allocation")

    mom = trailing_return(prices, lookback)
    alloc = []
    
    # Date from which we can start calculating signals
    # (need 'lookback' months of prior data for the first momentum calculation)
    # prices.index[0] is the first date we have data FOR.
    # mom.loc[idx] requires idx to be a date for which momentum FROM 'lookback' months AGO can be calculated.
    # So, the first valid 'idx' for mom.loc[idx] will be prices.index[lookback].
    
    for idx in prices.index:
        # Check if momentum data is available for the current date index
        # (it will have NaNs for the first 'lookback' periods)
        if idx not in mom.index or mom.loc[idx].isnull().all(): # if mom row is all NaN for this idx
            alloc.append(np.nan) # Cannot determine signal yet
            continue
            
        us_m = mom.loc[idx, EQUITY_US]
        intl_m = mom.loc[idx, EQUITY_INTL]
        
        if pd.isna(us_m) or pd.isna(intl_m):
            # If any crucial momentum is NaN (e.g., new asset), default to bonds
            alloc.append(BONDS)
            continue

        winner = EQUITY_US if us_m >= intl_m else EQUITY_INTL
        
        # Check if winner's momentum is positive; if not, allocate to BONDS
        # Also check if winner's momentum value itself is valid
        if pd.notna(mom.loc[idx, winner]) and mom.loc[idx, winner] > 0:
            alloc.append(winner)
        else:
            alloc.append(BONDS)
            
    return pd.Series(alloc, index=prices.index, name="Allocation")

def backtest(prices: pd.DataFrame, alloc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Backtests the strategy against a benchmark."""
    pos = alloc.shift(1).ffill() # Trade on next period's open based on this period's signal
    rets = prices.pct_change().fillna(0.0) # Calculate returns, fill first NaN row with 0.0

    strat_individual_returns = pd.Series(np.nan, index=pos.index, dtype=float)
    for date_idx in pos.index:
        asset_for_period = pos.loc[date_idx]
        if pd.notna(asset_for_period):
            if date_idx in rets.index and asset_for_period in rets.columns:
                strat_individual_returns.loc[date_idx] = rets.loc[date_idx, asset_for_period]
            else: 
                # This case should ideally not be hit if data is clean
                strat_individual_returns.loc[date_idx] = 0.0 
        # else: asset_for_period is NaN (e.g. initial lookback period), return remains NaN
        # This means strategy value will also be NaN during these periods.

    strat_cumulative_returns = (1 + strat_individual_returns).cumprod()
    
    benchmark_returns = rets[EQUITY_US] # Already has .fillna(0.0) from rets calculation
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    
    return strat_cumulative_returns, benchmark_cumulative_returns

def cagr(curve: pd.Series) -> float:
    """Calculates Compound Annual Growth Rate robustly."""
    valid_curve = curve.dropna()
    if valid_curve.empty or len(valid_curve) < 2: # Need at least two points to calculate growth
        return np.nan

    start_val = valid_curve.iloc[0]
    end_val = valid_curve.iloc[-1]
    
    start_date = valid_curve.index[0]
    end_date = valid_curve.index[-1]
    
    years = (end_date - start_date).days / 365.25
    
    if years <= 1e-6: # Avoid division by zero or extremely small periods (e.g. less than an hour)
        return 0.0 # Or np.nan if preferred for very short periods
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return np.nan 
        
    return (end_val / start_val) ** (1 / years) - 1

# --- Streamlit UI Function ---
def run_streamlit_app() -> None:
    if not _STREAMLIT_IMPORTED:
        print("Streamlit is not installed. Cannot run UI.", file=sys.stderr)
        return

    st.set_page_config(page_title="GEM Dual Momentum", layout="wide")
    st.title("Dual Momentum (GEM) Strategy")

    # Placeholder for data loading status
    data_status = st.empty()
    data_status.info("🚀 Initializing and fetching data... Please wait.")

    try:
        with st.spinner("Loading and processing market data... This may take a moment."):
            prices_df = load_prices()
            alloc_series = build_signals(prices_df)
        data_status.success(f"✅ Data loaded and signals built successfully from {prices_df.index.min().date()} to {prices_df.index.max().date()}.")
    except Exception as e:
        data_status.error(f"❌ Error during data loading or signal generation: {e}")
        st.warning("Troubleshooting tips: Check ticker symbols (SPY, VEU, BND), historical date range availability, and your internet connection. yfinance might occasionally have temporary issues.")
        st.code(f"Details:\n{type(e).__name__}: {str(e)}")
        return # Stop execution if data loading/processing fails

    last_valid_alloc = alloc_series.dropna()
    if not last_valid_alloc.empty:
        latest_signal_date = last_valid_alloc.index[-1].strftime('%Y-%m-%d')
        latest_allocation = last_valid_alloc.iloc[-1]
        st.metric(
            label="Latest Allocation Signal",
            value=str(latest_allocation),
            help=f"Signal generated on: {latest_signal_date}"
        )
    else:
        st.warning("⚠️ No valid allocation signals could be generated. This might be due to insufficient historical data for the lookback period or issues with the input data.")

    if prices_df.empty or alloc_series.empty:
        st.info("Cannot proceed with backtest as price data or allocation signals are empty.")
        return
        
    strat_curve, bench_curve = backtest(prices_df, alloc_series)
    
    # Prepare chart data, ensuring consistent starting point at 1.0 for visual comparison if desired
    # For now, plot raw cumulative returns. NaNs at the start will be handled by plot.
    chart_df = pd.concat([strat_curve.rename("GEM Strategy"), bench_curve.rename(f"{EQUITY_US} Benchmark")], axis=1)
    
    st.subheader("Performance Chart")
    st.line_chart(chart_df)
    
    st.subheader("Key Performance Metrics")
    strat_cagr_val = cagr(strat_curve)
    bench_cagr_val = cagr(bench_curve)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Strategy CAGR", f"{strat_cagr_val:.2%}" if pd.notna(strat_cagr_val) else "N/A")
    with col2:
        st.metric(f"Benchmark ({EQUITY_US}) CAGR", f"{bench_cagr_val:.2%}" if pd.notna(bench_cagr_val) else "N/A")

    st.sidebar.header("Strategy Parameters")
    st.sidebar.markdown(f"""
    - **US Equity:** `{EQUITY_US}`
    - **Int'l Equity:** `{EQUITY_INTL}`
    - **Bonds (Safe Asset):** `{BONDS}`
    - **Momentum Lookback:** `{LOOKBACK_MONTHS}` months
    - **Data Start Date:** `{START_DATE}`
    """)

    with st.expander("View Detailed Data Tables"):
        st.subheader("Recent Monthly Prices (Adjusted Close)")
        st.dataframe(prices_df.tail())
        st.subheader("Recent Allocation Signals")
        st.dataframe(alloc_series.dropna().to_frame().tail())
        st.subheader("Recent Cumulative Returns (Strategy & Benchmark)")
        # Drop rows where all values are NaN for a cleaner view, especially at the start
        st.dataframe(chart_df.dropna(how='all').tail())

# --- CLI Function ---
def run_cli_mode() -> None:
    """Handles Command Line Interface execution."""
    mode = "signal" # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1] in ["signal", "backtest"]:
            mode = sys.argv[1]
        else:
            # Check if it's an argument from Streamlit itself, if so, ignore for CLI mode.
            if not sys.argv[1].startswith('--'):
                 print(f"Unknown CLI mode: '{sys.argv[1]}'. Use 'signal' or 'backtest'. Defaulting to 'signal'.", file=sys.stderr)
    
    print(f"--- GEM Strategy CLI ({mode} mode) ---")
    try:
        prices = load_prices()
        print(f"Data loaded: {len(prices)} months from {prices.index.min().date()} to {prices.index.max().date()}")
        allocations = build_signals(prices)
        
        if mode == "backtest":
            if allocations.dropna().empty:
                print("No allocation signals generated. Cannot run backtest.")
                return

            last_valid_alloc_idx = allocations.dropna().index[-1]
            print(f"Latest allocation ({last_valid_alloc_idx.date()}): {allocations.loc[last_valid_alloc_idx]}")
            
            s, b = backtest(prices, allocations)
            strat_cagr_val = cagr(s)
            bench_cagr_val = cagr(b)
            
            print(f"Strategy CAGR : {strat_cagr_val:.2%}" if pd.notna(strat_cagr_val) else "Strategy CAGR : N/A")
            print(f"{EQUITY_US} Buy & Hold CAGR: {bench_cagr_val:.2%}" if pd.notna(bench_cagr_val) else f"{EQUITY_US} Buy & Hold CAGR: N/A")

        else:  # "signal" mode
            latest_allocations = allocations.dropna()
            if not latest_allocations.empty:
                latest_signal = latest_allocations.iloc[-1]
                latest_date = latest_allocations.index[-1].strftime('%Y-%m-%d')
                print(f"Latest Allocation Signal ({latest_date}): {latest_signal}")
            else:
                print("No valid allocation signal available for the latest period.")
    except Exception as e:
        print(f"CLI Error: {e}", file=sys.stderr)
        # For debugging CLI issues:
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Check if Streamlit is running the script by looking for its specific environment variable.
    # _STREAMLIT_IMPORTED flag ensures we only try to run ui() if streamlit was successfully imported.
    # Using os.environ.get('STREAMLIT_SERVER_PORT') is also common.
    # Render sets the PORT environment variable. Streamlit also uses SERVER_PORT.
    # A more generic check if Streamlit itself thinks it's running a script:
    # hasattr(st, 'runtime') and st.runtime.exists() (for newer Streamlit versions)
    # For now, os.environ.get('SERVER_PORT') is simple for Render and common Streamlit deployments.
    
    is_streamlit_environment = False
    if _STREAMLIT_IMPORTED:
        # Check for an environment variable that Streamlit sets when it's serving
        if os.environ.get('SERVER_PORT') or os.environ.get('STREAMLIT_SERVER_PORT'):
            is_streamlit_environment = True
        # Fallback for some Streamlit execution contexts if env vars are not primary
        # This check is more about whether the script is being *processed* by streamlit's runner
        # For example, `streamlit run app.py` will make this true for the script it executes
        try:
            if st._is_running_with_streamlit: # Accessing internal flag, use with caution
                is_streamlit_environment = True
        except AttributeError: # If _is_running_with_streamlit doesn't exist in the version
            pass


    if is_streamlit_environment:
        run_streamlit_app()
    else:
        run_cli_mode()
