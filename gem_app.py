# gem_app.py
from __future__ import annotations
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# Try importing Streamlit and set a flag
try:
    import streamlit as st
    _STREAMLIT_IMPORTED = True
except ImportError:
    _STREAMLIT_IMPORTED = False

# --- Core Logic Functions (mostly unchanged, but inputs will be dynamic) ---
def load_prices(ticker_list: list[str], s_date: datetime.date, e_date: datetime.date) -> pd.DataFrame | None:
    """Loads and resamples price data for a dynamic list of tickers."""
    print(f"[APP_LOG] load_prices: Tickers: {ticker_list}, Start: {s_date}, End: {e_date}")
    if not ticker_list:
        st.warning("No tickers provided to load.")
        return None
    try:
        # yfinance expects strings for dates
        s_date_str = s_date.strftime("%Y-%m-%d")
        e_date_str = e_date.strftime("%Y-%m-%d")

        prices_data = yf.download(ticker_list, start=s_date_str, end=e_date_str, progress=False, auto_adjust=True)
        
        if prices_data.empty:
            st.error(f"Failed to download price data. yf.download returned empty. Are tickers valid for the period {s_date_str} to {e_date_str}?")
            return None

        # Handle single vs multiple tickers for column selection
        if len(ticker_list) == 1:
            if 'Close' not in prices_data.columns:
                 st.error(f"Could not find 'Close' column for single ticker {ticker_list[0]}. Available: {prices_data.columns}")
                 return None
            prices_close = prices_data[['Close']].rename(columns={'Close': ticker_list[0]})
        else:
            if 'Close' not in prices_data.columns.get_level_values(0):
                 st.error(f"'Close' data not found in yf.download multi-ticker columns. Columns are: {prices_data.columns}")
                 return None
            prices_close = prices_data['Close']
            # Ensure all requested tickers are present as columns
            for ticker in ticker_list:
                if ticker not in prices_close.columns:
                    st.error(f"Data for ticker '{ticker}' not found in downloaded 'Close' prices. Available: {prices_close.columns}")
                    # Attempt to continue with available data, or return None
                    # For simplicity now, we'll raise an error / return None if a key ticker is missing
                    # A more robust way would be to filter ticker_list to only those successfully downloaded.
                    return None


        if prices_close.isnull().all().all():
             st.error("Price data consists only of NaNs after selection.")
             return None

        resampled_prices = prices_close.resample("M").last()
        # We'll drop NaNs per column within functions that use it, rather than globally via dropna(how="any")
        # This allows for assets with different start dates to be used, though GEM logic assumes overlap.
        # For GEM, we often need all compared assets to have data for the lookback period.
        # We will handle this within build_signals by checking for NaNs in momentum.
        
        print(f"[APP_LOG] load_prices: Resampled prices shape: {resampled_prices.shape}")
        return resampled_prices

    except Exception as e:
        st.error(f"Error loading price data: {e}")
        import traceback
        st.expander("Traceback").code(traceback.format_exc())
        return None


def trailing_return(px: pd.DataFrame, m: int) -> pd.DataFrame:
    return px.pct_change(m)

def build_signals(prices: pd.DataFrame, etf1: str, etf2: str, fallback_etf: str, lookback: int) -> pd.Series:
    print(f"[APP_LOG] build_signals: ETFs: {etf1}, {etf2}, Fallback: {fallback_etf}. Lookback: {lookback}. Prices shape: {prices.shape}")
    alloc = pd.Series(np.nan, index=prices.index, name="Allocation", dtype=object)

    # Ensure all necessary columns exist in the price data for momentum calculation
    required_cols_for_mom = list(set([etf1, etf2])) # Fallback asset doesn't need momentum for decision
    if not all(col in prices.columns for col in required_cols_for_mom):
        missing_cols = [col for col in required_cols_for_mom if col not in prices.columns]
        st.error(f"build_signals: Price data missing for primary ETFs: {missing_cols}. Cannot build signals.")
        return alloc # Return series of NaNs

    mom = trailing_return(prices, lookback) # Calculate momentum for all available price columns
    
    for idx in prices.index:
        # Check if we have enough historical data to compute momentum for this date
        if idx < prices.index[0] + pd.DateOffset(months=lookback):
            # alloc.loc[idx] remains np.nan
            continue

        # Check if momentum values are available for the decision ETFs for this specific index
        if etf1 not in mom.columns or etf2 not in mom.columns: # Should be caught by earlier check but defensive
            alloc.loc[idx] = fallback_etf # Or np.nan
            continue
            
        mom_etf1 = mom.loc[idx, etf1]
        mom_etf2 = mom.loc[idx, etf2]
        
        if pd.isna(mom_etf1) or pd.isna(mom_etf2):
            # Not enough historical data for one of the ETFs at this point for momentum
            # or the ETF itself had NaNs. Default to fallback.
            alloc.loc[idx] = fallback_etf
            continue

        winner = etf1 if mom_etf1 >= mom_etf2 else etf2
        
        # Absolute momentum check on the winner
        mom_winner = mom.loc[idx, winner]
        if pd.notna(mom_winner) and mom_winner > 0:
            alloc.loc[idx] = winner
        else:
            alloc.loc[idx] = fallback_etf
            
    print(f"[APP_LOG] build_signals: Finished. Allocation example: {alloc.dropna().tail(3)}")
    return alloc


def backtest(prices: pd.DataFrame, alloc: pd.Series, benchmark_ticker: str) -> tuple[pd.Series | None, pd.Series | None]:
    print(f"[APP_LOG] backtest: Prices shape: {prices.shape}, Alloc unique: {alloc.dropna().unique()}, Benchmark: {benchmark_ticker}")
    if alloc.dropna().empty:
        st.warning("No valid allocation signals for backtest.")
        return None, None

    pos = alloc.shift(1).ffill()
    rets = prices.pct_change() # Daily/monthly returns, fillna for first period

    strat_individual_returns = pd.Series(np.nan, index=pos.index, dtype=float)
    for date_idx in pos.index:
        asset_for_period = pos.loc[date_idx]
        if pd.notna(asset_for_period):
            if asset_for_period in rets.columns and date_idx in rets.index: # Ensure asset and date are valid
                strat_individual_returns.loc[date_idx] = rets.loc[date_idx, asset_for_period]
            else: 
                strat_individual_returns.loc[date_idx] = 0.0 # Fallback for missing data after allocation
        else: # Asset is NaN (initial lookback period), return is NaN
             strat_individual_returns.loc[date_idx] = np.nan


    # Replace initial NaNs in returns with 0 for cumprod to start at 1 correctly after dropna
    strat_cumulative_returns = (1 + strat_individual_returns.fillna(0.0)).cumprod()
    
    # Benchmark
    benchmark_cumulative_returns = None
    if benchmark_ticker in rets.columns:
        benchmark_returns = rets[benchmark_ticker].fillna(0.0)
        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    else:
        st.warning(f"Benchmark ticker '{benchmark_ticker}' not found in price data. Cannot plot benchmark.")
        
    print(f"[APP_LOG] backtest: Finished. Strategy tail: {strat_cumulative_returns.dropna().tail(3)}")
    return strat_cumulative_returns, benchmark_cumulative_returns


def cagr(curve: pd.Series) -> float:
    if curve is None or curve.dropna().empty or len(curve.dropna()) < 2: return np.nan
    valid_curve = curve.dropna()
    start_val, end_val = valid_curve.iloc[0], valid_curve.iloc[-1]
    start_date, end_date = valid_curve.index[0], valid_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    if years <= 1e-6 or pd.isna(start_val) or pd.isna(end_val) or start_val == 0 : return np.nan
    return (end_val / start_val) ** (1 / years) - 1

# --- Streamlit UI ---
def run_streamlit_app() -> None:
    if not _STREAMLIT_IMPORTED:
        print("Streamlit is not installed. Cannot run UI.", file=sys.stderr)
        return

    st.set_page_config(page_title="Flexible Dual Momentum", layout="wide")
    st.title("Flexible Dual Momentum (GEM-Like) Strategy Backtester")

    st.sidebar.header("Strategy Configuration")
    
    # Set default start_date to 5 years ago from today (May 26, 2025)
    # For production, you might want it fixed or more robustly determined.
    default_start_date = datetime.now().date() - timedelta(days=5*365) # Approx 5 years
    # Since current simulated date is May 26, 2025:
    default_start_date = datetime(2020, 5, 26).date()


    etf1_default = "SPY"
    etf2_default = "QQQ" # Changed from VEU for variety
    fallback_default = "BND"
    benchmark_default = "SPY"
    lookback_default = 12

    etf1 = st.sidebar.text_input("Growth ETF 1 (e.g., SPY, IVV)", value=etf1_default).upper()
    etf2 = st.sidebar.text_input("Growth ETF 2 (e.g., QQQ, VEU, EFA)", value=etf2_default).upper()
    
    fallback_options = ["BND", "GLD", "BTC-USD", "SHY", "BIL", "TLT", "Cash (0% return)"]
    # For "Cash (0% return)" we'd need special handling (a dummy ticker with 0 returns, or logic in backtest)
    # For now, let's stick to actual tickers. User can type their preferred bond/cash-like ETF.
    # To simplify: just use text input for fallback.
    fallback_etf = st.sidebar.text_input("Fallback Asset (e.g., BND, GLD, BTC-USD, SHY, BIL, TLT)", value=fallback_default).upper()
    
    benchmark_etf = st.sidebar.text_input("Benchmark ETF (e.g., SPY, QQQ, VT)", value=benchmark_default).upper()
    
    start_date = st.sidebar.date_input("Start Date for Backtest", value=default_start_date, max_value=datetime.now().date() - timedelta(days=1))
    lookback_months = st.sidebar.number_input("Lookback Months for Momentum", min_value=1, max_value=36, value=lookback_default)

    run_button = st.sidebar.button("Run Strategy Backtest")

    # Initialize session state for results
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'latest_signal' not in st.session_state:
        st.session_state.latest_signal = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None


    if run_button:
        st.session_state.results_df = None # Clear previous results
        st.session_state.latest_signal = None
        st.session_state.metrics = None
        st.session_state.error_message = None
        print(f"[APP_LOG] Run button clicked. Params: {etf1}, {etf2}, {fallback_etf}, Benchmark: {benchmark_etf}, Start: {start_date}, Lookback: {lookback_months}")

        if not all([etf1, etf2, fallback_etf, benchmark_etf]):
            st.session_state.error_message = "Please ensure all ETF ticker inputs are filled."
        else:
            # Define all unique tickers needed for data download
            # Benchmark is included here in case it's different.
            all_unique_tickers = list(set([t for t in [etf1, etf2, fallback_etf, benchmark_etf] if t]))
            
            with st.spinner(f"Fetching data for {', '.join(all_unique_tickers)} and running backtest..."):
                end_date = datetime.now().date() # Use current date as end date
                prices_df = load_prices(all_unique_tickers, start_date, end_date)

                if prices_df is not None and not prices_df.empty:
                    # Check if all essential ETFs for strategy are in the loaded prices
                    essential_strategy_tickers = list(set([etf1, etf2, fallback_etf]))
                    missing_essential = [t for t in essential_strategy_tickers if t not in prices_df.columns]
                    if missing_essential:
                        st.session_state.error_message = f"Data for essential strategy tickers {missing_essential} could not be loaded. Please check tickers and chosen date range."
                    else:
                        alloc_series = build_signals(prices_df, etf1, etf2, fallback_etf, lookback_months)
                        
                        if alloc_series.dropna().empty:
                            st.session_state.error_message = "No valid allocation signals generated. This might be due to insufficient historical data for the lookback period or issues with the input data for the chosen ETFs."
                        else:
                            strat_curve, bench_curve = backtest(prices_df, alloc_series, benchmark_etf)
                            
                            st.session_state.results_df = pd.concat(
                                [s.rename("Strategy") if s is not None else None for s in [strat_curve, bench_curve]],
                                axis=1
                            ).dropna(how='all') # Drop rows where all values are NaN (start of series)
                            
                            last_valid_alloc = alloc_series.dropna()
                            if not last_valid_alloc.empty:
                                st.session_state.latest_signal = {
                                    "date": last_valid_alloc.index[-1].strftime('%Y-%m-%d'),
                                    "asset": last_valid_alloc.iloc[-1]
                                }
                            
                            st.session_state.metrics = {
                                "strat_cagr": cagr(strat_curve) if strat_curve is not None else np.nan,
                                "bench_cagr": cagr(bench_curve) if bench_curve is not None else np.nan
                            }
                            print("[APP_LOG] Backtest processing complete. Results stored in session state.")
                else:
                    if not st.session_state.error_message: # If load_prices didn't set a specific error
                        st.session_state.error_message = "Failed to load price data. Please check tickers and date range."

    # --- Display Results ---
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if st.session_state.latest_signal:
        st.subheader("Latest Allocation Signal")
        latest_sig = st.session_state.latest_signal
        st.metric(label=f"Signal for period after {latest_sig['date']}", value=str(latest_sig['asset']))

    if st.session_state.results_df is not None and not st.session_state.results_df.empty:
        st.subheader("Performance Chart")
        # Ensure column names match what was set, especially if benchmark failed
        plot_df = st.session_state.results_df.copy()
        if "Strategy" not in plot_df.columns and strat_curve is not None: # Should be there if results_df exists
             plot_df["Strategy"] = strat_curve
        if benchmark_etf not in plot_df.columns and bench_curve is not None:
             plot_df[benchmark_etf] = bench_curve
        
        # Filter out columns that might be all NaN if a curve is None
        plot_df = plot_df.loc[:, plot_df.columns[plot_df.notna().any()]]
        
        st.line_chart(plot_df)
        
        st.subheader("Key Performance Metrics")
        metrics = st.session_state.metrics
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Strategy CAGR", f"{metrics['strat_cagr']:.2%}" if pd.notna(metrics['strat_cagr']) else "N/A")
            with col2:
                st.metric(f"Benchmark ({benchmark_etf}) CAGR", f"{metrics['bench_cagr']:.2%}" if pd.notna(metrics['bench_cagr']) else "N/A")
    elif run_button and not st.session_state.error_message: # Run was clicked, no error, but no results_df (shouldn't happen ideally)
        st.info("Backtest run, but no chart data to display. This might indicate very short data period or issues post-processing.")


    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**What is this?** This app backtests a Dual Momentum-like strategy. You select two 'growth' ETFs and a 'fallback' (safe) asset. The strategy allocates to the stronger growth ETF if its own momentum is positive, otherwise it switches to the fallback asset.")
    st.sidebar.markdown("Data is fetched from Yahoo Finance.")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("[APP_LOG] Script execution started.")
    is_streamlit_environment = False
    if _STREAMLIT_IMPORTED:
        try:
            if hasattr(st, '_is_running_with_streamlit') and st._is_running_with_streamlit: is_streamlit_environment = True
            elif hasattr(st, 'runtime') and hasattr(st.runtime, 'exists') and st.runtime.exists(): is_streamlit_environment = True
            elif os.environ.get('PORT') or os.environ.get('STREAMLIT_SERVER_PORT'): is_streamlit_environment = True
        except Exception: pass 

    if is_streamlit_environment:
        print("[APP_LOG] Running Streamlit app.")
        run_streamlit_app()
    else: # Basic CLI placeholder if needed, not primary focus for this version
        print("[APP_LOG] Detected CLI mode (or Streamlit not running script). This app is primarily a Streamlit UI.")
        print("To run the Streamlit app, use: streamlit run gem_app.py")
