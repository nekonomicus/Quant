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
# <<! IMPORTANT FOR RENDER FREE TIER TEST !>>
# Start with a recent date to minimize memory usage on Render Free Tier
START_DATE = "2022-01-01" # Or "2023-01-01" for even less data
# START_DATE = "2005-01-01" # Original - use only if "2022-01-01" works and you consider upgrading Render

# --- Core Logic Functions ---
def load_prices(s: str = START_DATE, e: str | None = None) -> pd.DataFrame:
    print(f"[GEM_APP_DEBUG] load_prices called. Effective START_DATE: {s}")
    end_date = e or datetime.today().strftime("%Y-%m-%d")
    print(f"[GEM_APP_DEBUG] Attempting yf.download for {TICKERS} from {s} to {end_date}")
    
    prices_data = yf.download(TICKERS, start=s, end=end_date, progress=False, auto_adjust=True)
    
    if prices_data.empty:
        print("[GEM_APP_DEBUG] CRITICAL: yf.download returned an empty DataFrame.")
        raise ValueError("yf.download returned an empty DataFrame. Check tickers/yfinance API.")
    print(f"[GEM_APP_DEBUG] yf.download successful. Raw data shape: {prices_data.shape}")
    
    if 'Close' not in prices_data.columns.get_level_values(0):
        print(f"[GEM_APP_DEBUG] CRITICAL: 'Close' data not found in yf.download columns. Columns are: {prices_data.columns}")
        raise ValueError("Structure of yf.download output changed, 'Close' not found as expected.")

    prices_close = prices_data['Close']
    if prices_close.empty or prices_close.isnull().all().all():
         print("[GEM_APP_DEBUG] CRITICAL: 'Close' price data is empty or all NaNs after selection.")
         raise ValueError("No 'Close' price data found after yf.download selection (all NaNs or empty).")
    print(f"[GEM_APP_DEBUG] 'Close' data extracted. Shape: {prices_close.shape}")

    resampled_prices = prices_close.resample("M").last().dropna(how="any")
    if resampled_prices.empty:
        print("[GEM_APP_DEBUG] CRITICAL: Data is empty after resample and dropna.")
        raise ValueError("No data remaining after resampling and dropping NaNs. Shorten START_DATE or check data availability for ALL tickers.")
    print(f"[GEM_APP_DEBUG] Resampling complete. Final shape: {resampled_prices.shape}")
    return resampled_prices

def trailing_return(px: pd.DataFrame, m: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    print(f"[GEM_APP_DEBUG] trailing_return called with lookback {m}. Data shape: {px.shape}")
    return px.pct_change(m)

def build_signals(prices: pd.DataFrame, lookback: int = LOOKBACK_MONTHS) -> pd.Series:
    print(f"[GEM_APP_DEBUG] build_signals called. Data shape: {prices.shape}, Lookback: {lookback}")
    if prices.empty:
        print("[GEM_APP_DEBUG] build_signals: prices_df is empty, returning empty Series.")
        return pd.Series(dtype='object', name="Allocation")

    if len(prices.index) < lookback + 1:
         print(f"[GEM_APP_DEBUG] build_signals: Not enough data ({len(prices.index)} rows) for lookback ({lookback}), returning NaNs.")
         return pd.Series(np.nan, index=prices.index, name="Allocation")

    mom = trailing_return(prices, lookback)
    alloc = []
    
    print(f"[GEM_APP_DEBUG] build_signals: Starting loop through {len(prices.index)} price dates.")
    for idx_num, idx in enumerate(prices.index):
        if idx_num % 50 == 0 : # Print progress occasionally
            print(f"[GEM_APP_DEBUG] build_signals: Processing index {idx_num}, date {idx.date()}")

        if idx not in mom.index or mom.loc[idx].isnull().all():
            alloc.append(np.nan)
            continue
            
        us_m = mom.loc[idx, EQUITY_US]
        intl_m = mom.loc[idx, EQUITY_INTL]
        
        if pd.isna(us_m) or pd.isna(intl_m):
            alloc.append(BONDS)
            continue

        winner = EQUITY_US if us_m >= intl_m else EQUITY_INTL
        
        if pd.notna(mom.loc[idx, winner]) and mom.loc[idx, winner] > 0:
            alloc.append(winner)
        else:
            alloc.append(BONDS)
    print("[GEM_APP_DEBUG] build_signals: Loop finished.")
    return pd.Series(alloc, index=prices.index, name="Allocation")

def backtest(prices: pd.DataFrame, alloc: pd.Series) -> tuple[pd.Series, pd.Series]:
    print(f"[GEM_APP_DEBUG] backtest called. Prices shape: {prices.shape}, Alloc shape: {alloc.shape}")
    pos = alloc.shift(1).ffill()
    rets = prices.pct_change().fillna(0.0)

    strat_individual_returns = pd.Series(np.nan, index=pos.index, dtype=float)
    for date_idx in pos.index:
        asset_for_period = pos.loc[date_idx]
        if pd.notna(asset_for_period):
            if date_idx in rets.index and asset_for_period in rets.columns:
                strat_individual_returns.loc[date_idx] = rets.loc[date_idx, asset_for_period]
            else: 
                strat_individual_returns.loc[date_idx] = 0.0 
    
    strat_cumulative_returns = (1 + strat_individual_returns).cumprod()
    benchmark_returns = rets[EQUITY_US]
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    print("[GEM_APP_DEBUG] backtest finished.")
    return strat_cumulative_returns, benchmark_cumulative_returns

def cagr(curve: pd.Series) -> float:
    valid_curve = curve.dropna()
    if valid_curve.empty or len(valid_curve) < 2: return np.nan
    start_val, end_val = valid_curve.iloc[0], valid_curve.iloc[-1]
    start_date, end_date = valid_curve.index[0], valid_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    if years <= 1e-6 or start_val == 0 or pd.isna(start_val) or pd.isna(end_val): return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def run_streamlit_app() -> None:
    if not _STREAMLIT_IMPORTED:
        print("[GEM_APP_CRITICAL] Streamlit is not installed. Cannot run UI.", file=sys.stderr)
        return

    print("[GEM_APP_DEBUG] run_streamlit_app called.")
    st.set_page_config(page_title="GEM Dual Momentum", layout="wide")
    st.title("Dual Momentum (GEM) Strategy")

    data_status = st.empty()
    data_status.info("🚀 Initializing and fetching data... Please wait.")
    print("[GEM_APP_DEBUG] Streamlit page configured. Attempting data load.")

    try:
        with st.spinner("Loading and processing market data... This may take a moment."):
            prices_df = load_prices() # Uses START_DATE defined at top
            alloc_series = build_signals(prices_df)
        data_status.success(f"✅ Data loaded: {prices_df.index.min().date()} to {prices_df.index.max().date()}. Signals built.")
        print("[GEM_APP_DEBUG] Data load and signal build successful (within try block).")
    except Exception as e:
        print(f"[GEM_APP_ERROR] Exception during data load/signal build: {type(e).__name__}: {e}") # Log error
        data_status.error(f"❌ Error: {e}")
        st.warning("Could not load/process data. Tips: Check START_DATE for data volume, ticker validity (SPY, VEU, BND), and internet. yfinance can have temporary issues.")
        # import traceback # For more detailed error in st.expander if needed
        # st.expander("Error Details").code(traceback.format_exc())
        return

    # ... (rest of your Streamlit UI logic: metrics, chart, sidebar, expander from previous full example)
    last_valid_alloc = alloc_series.dropna()
    if not last_valid_alloc.empty:
        latest_signal_date = last_valid_alloc.index[-1].strftime('%Y-%m-%d')
        latest_allocation = last_valid_alloc.iloc[-1]
        st.metric(label="Latest Allocation Signal", value=str(latest_allocation), help=f"Signal date: {latest_signal_date}")
    else:
        st.warning("⚠️ No valid allocation signals generated.")

    if prices_df.empty or alloc_series.dropna().empty: # Check if alloc has any valid signals
        st.info("Cannot proceed with backtest as price data or valid allocation signals are missing.")
        return
        
    strat_curve, bench_curve = backtest(prices_df, alloc_series)
    
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
    - **Configured Data Start Date:** `{START_DATE}`
    """)
    if not prices_df.empty:
        st.sidebar.markdown(f"- **Actual Data Loaded From:** `{prices_df.index.min().date()}`")


    with st.expander("View Detailed Data Tables (Recent Rows)"):
        st.subheader("Monthly Prices (Adjusted Close)")
        st.dataframe(prices_df.tail())
        st.subheader("Allocation Signals (No NaNs)")
        st.dataframe(alloc_series.dropna().to_frame().tail())
        st.subheader("Cumulative Returns (Strategy & Benchmark)")
        st.dataframe(chart_df.dropna(how='all').tail())
    print("[GEM_APP_DEBUG] Streamlit UI rendering finished.")


def run_cli_mode() -> None:
    print("[GEM_APP_DEBUG] run_cli_mode called.")
    # ... (CLI logic remains the same as your last full version)
    mode = "signal" 
    if len(sys.argv) > 1 and sys.argv[1] in ["signal", "backtest"]: mode = sys.argv[1]
    print(f"--- GEM Strategy CLI ({mode} mode) ---")
    try:
        prices = load_prices()
        allocations = build_signals(prices)
        if mode == "backtest": cli(prices, allocations) # Assuming cli function exists for backtest output
        else: # signal mode
            latest_alloc = allocations.dropna()
            if not latest_alloc.empty: print(f"Latest Signal ({latest_alloc.index[-1].date()}): {latest_alloc.iloc[-1]}")
            else: print("No signal.")
    except Exception as e: print(f"CLI Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    print("[GEM_APP_DEBUG] Script execution started.")
    is_streamlit_environment = False
    if _STREAMLIT_IMPORTED:
        print("[GEM_APP_DEBUG] Streamlit is imported.")
        try:
            if hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit:
                is_streamlit_environment = True
                print("[GEM_APP_DEBUG] Detected Streamlit via st._is_running_with_streamlit.")
            elif hasattr(st, 'runtime') and hasattr(st.runtime, 'exists') and st.runtime.exists():
                 is_streamlit_environment = True
                 print("[GEM_APP_DEBUG] Detected Streamlit via st.runtime.exists().")
            elif os.environ.get('PORT'): # Render sets PORT
                is_streamlit_environment = True
                print(f"[GEM_APP_DEBUG] Detected Streamlit via PORT env var: {os.environ.get('PORT')}")
            elif os.environ.get('STREAMLIT_SERVER_PORT'): # Also check this one
                is_streamlit_environment = True
                print(f"[GEM_APP_DEBUG] Detected Streamlit via STREAMLIT_SERVER_PORT env var: {os.environ.get('STREAMLIT_SERVER_PORT')}")

        except Exception as e:
            print(f"[GEM_APP_DEBUG] Error during Streamlit detection: {e}")
            pass 

    if is_streamlit_environment:
        print("[GEM_APP_DEBUG] Running Streamlit app.")
        run_streamlit_app()
    else:
        print("[GEM_APP_DEBUG] Running CLI mode.")
        run_cli_mode()
