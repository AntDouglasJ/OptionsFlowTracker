import streamlit as st
import yfinance as yf
import pandas as pd
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor
import ssl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN", "SPY", "QQQ", "NVDA", "META", "GOOGL"]
DATA_FILE = "options_data.json"
REFRESH_INTERVAL_SEC = 60 * 10  # Refresh data every 10 minutes
LIVE_PRICE_INTERVAL = 5
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# --- Caching and Data Persistence ---
price_cache = {}
options_cache = {}
last_fetch_time = None


def load_data():
    global options_cache, last_fetch_time
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                data = json.load(f)
                options_cache = data.get("options", {})
                last_fetch_time = data.get("last_fetch_time")
            except json.JSONDecodeError:
                st.error("Error loading cached data. Starting fresh.")
                options_cache = {}
                last_fetch_time = None


def save_data():
    def convert_to_serializable(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        if isinstance(obj, pd.Series):
            return obj.apply(convert_to_serializable).tolist()
        return obj

    serializable_options = convert_to_serializable(options_cache)
    data = {"options": serializable_options, "last_fetch_time": datetime.now().isoformat()}
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


def is_cache_valid():
    global last_fetch_time
    if last_fetch_time is None:
        return False
    last_fetch_datetime = datetime.fromisoformat(last_fetch_time)
    return datetime.now() - last_fetch_datetime < timedelta(seconds=REFRESH_INTERVAL_SEC)


# --- Live Price Fetching ---
def fetch_live_price_yf(symbol, error_container):
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.info.get('regularMarketPrice', None)
        return current_price
    except Exception as e:
        with error_container:
            st.error(f"Error fetching live price for {symbol}: {e}")
        return None


def update_prices_in_cache(symbols, error_container):
    for symbol in symbols:
        price = fetch_live_price_yf(symbol, error_container)
        if price is not None:
            price_cache[symbol] = price


# --- Option Data Fetching ---
def fetch_options_for_symbol(symbol, error_container):
    options_data = []
    all_expiries = set()
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.info.get('regularMarketPrice', None)
        if current_price is None:
            history = ticker.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else None

        price_cache[symbol] = current_price

        options = ticker.options

        if not options:
            with error_container:
                st.warning(f"No options data available for {symbol}")
                return options_data, all_expiries

        for expiry in options:
            try:
                all_expiries.add(expiry)
                if (symbol not in options_cache) or (expiry not in options_cache[symbol]):
                    opt_chain = ticker.option_chain(expiry)
                    call_data = opt_chain.calls.to_dict('records') if not opt_chain.calls.empty else []
                    put_data = opt_chain.puts.to_dict('records') if not opt_chain.puts.empty else []
                    if symbol not in options_cache:
                        options_cache[symbol] = {}
                    options_cache[symbol][expiry] = {"calls": call_data, "puts": put_data}

                if symbol in options_cache and expiry in options_cache[symbol]:
                    call_data = options_cache[symbol][expiry].get("calls", [])
                    put_data = options_cache[symbol][expiry].get("puts", [])

                    for option_type, chain in [("call", call_data), ("put", put_data)]:
                        for row in chain:
                            options_data.append({
                                "Symbol": symbol,
                                "Current Price": current_price,
                                "Strike": row['strike'],
                                "Bid": row['bid'],
                                "Ask": row['ask'],
                                "Type": option_type.capitalize(),
                                "Expiry": expiry,
                                "Volume": row.get('volume', 0),
                                "Spread": row['ask'] - row['bid'],
                            })
            except Exception as e:
                with error_container:
                    st.error(f"Error fetching options for {symbol} at expiry {expiry}: {e}")
    except Exception as e:
        with error_container:
            st.error(f"Error fetching data for {symbol}: {e}")
        return [], set()
    return options_data, all_expiries


def fetch_options_data(symbols, error_container):
    global options_cache
    all_expiries = set()
    all_options_data = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda symbol: fetch_options_for_symbol(symbol, error_container), symbols)
        for options_data, expiries in results:
            all_options_data.extend(options_data)
            all_expiries.update(expiries)
    return pd.DataFrame(all_options_data), sorted(list(all_expiries))

def create_candlestick_chart(symbol, chart_placeholder, timeframe="3mo", include_after_hours=False):
    try:
        ticker = yf.Ticker(symbol)
        history_data = ticker.history(period=timeframe, prepost = include_after_hours)
        if not history_data.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                               row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=history_data.index,
                                         open=history_data['Open'],
                                         high=history_data['High'],
                                         low=history_data['Low'],
                                         close=history_data['Close'],
                                         name='Candlestick'), row=1, col=1)
            fig.add_trace(go.Bar(x=history_data.index,
                                y=history_data['Volume'],
                                marker_color='rgba(150,150,150,0.6)',
                                name='Volume'), row=2, col=1)
            fig.update_layout(title=f"{symbol} Price and Volume Chart",
                              xaxis_rangeslider_visible=False)
            fig.update_xaxes(title_text='Date', row=1, col=1)
            fig.update_xaxes(title_text='Date', row=2, col=1)
            fig.update_yaxes(title_text='Price', row=1, col=1)
            fig.update_yaxes(title_text='Volume', row=2, col=1)

            with chart_placeholder:
                st.plotly_chart(fig, use_container_width=True)
        else:
            with chart_placeholder:
                st.warning(f"No historical data available for {symbol}")
    except Exception as e:
        with chart_placeholder:
            st.error(f"Error generating chart for {symbol}: {e}")


# --- Streamlit UI ---

# Set up page navigation
st.set_page_config(page_title="Options Flow Tracker", layout="wide")

page = st.sidebar.selectbox("Navigation", ["Options Table", "Stock Chart"])


if page == "Options Table":
    st.title("Options Flow Tracker")
    # --- Sidebar ---
    st.sidebar.header("Filters")

    load_data()

    # Create an error container in the UI
    error_container = st.empty()

    # Fetch Data if Cache is invalid or if button is clicked
    if not is_cache_valid():
        with st.spinner("Fetching initial options data..."):
            data, all_expiries = fetch_options_data(SYMBOLS, error_container)
            save_data()
            error_container.empty()
    else:
        with st.spinner("Loading from cache..."):
            data, all_expiries = fetch_options_data(SYMBOLS, error_container)  # This loads from cache
            error_container.empty()

    # Update prices in background
    update_prices_in_cache(SYMBOLS, error_container)

    # Dropdowns
    unique_symbols = sorted(data["Symbol"].unique()) if not data.empty else []
    selected_symbol = st.sidebar.selectbox("Symbol", ["All"] + unique_symbols)
    selected_type = st.sidebar.selectbox("Option Type", ["All", "Call", "Put"])
    selected_expiry = st.sidebar.selectbox("Expiry Date (YYYY-MM-DD)", ["All"] + all_expiries)

    # Add "In the Money" filter
    in_out_options = ["All", "In the Money", "Out of the Money"]
    selected_in_out = st.sidebar.selectbox("In/Out of the Money", in_out_options)

    # Filter controls
    st.sidebar.subheader("Filter Ranges")
    min_strike, max_strike = st.sidebar.slider("Strike Price Range", min_value=0.0, max_value=max(data['Strike'], default=1000) * 2,
                                            value=(0.0, max(data['Strike'], default=1000) * 2), step=1.0)
    max_volume_val = max(data['Volume'], default=10000)
    min_volume, max_volume = st.sidebar.slider("Volume Range", min_value=0, max_value=int(max_volume_val),
                                            value=(0, int(max_volume_val)), step=10)

    # Add Bid/Ask filters
    st.sidebar.subheader("Bid/Ask Filter Ranges")
    min_bid, max_bid = st.sidebar.slider("Bid Range", min_value=0.0, max_value=max(data['Bid'], default=100) * 2,
                                       value=(0.0, max(data['Bid'], default=100) * 2), step=0.1)
    min_ask, max_ask = st.sidebar.slider("Ask Range", min_value=0.0, max_value=max(data['Ask'], default=100) * 2,
                                       value=(0.0, max(data['Ask'], default=100) * 2), step=0.1)

    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds")

    # Initialize placeholder for the table
    table_placeholder = st.empty()


    # --- Data Updating and Display ---

    def get_in_the_money_filter(data, symbol, current_price):
        def inner_filter(row):
            if symbol == "All" or not current_price:
                return True
            if row["Type"] == "Call":
                return current_price > row["Strike"]
            else:
                return current_price < row["Strike"]

        return inner_filter


    def format_price(price):
        return f"{price:.2f}" if price is not None else "N/A"

    def update_data(error_container):
        filtered_data = data.copy()

        # Apply filters
        if selected_symbol != "All":
            filtered_data = filtered_data[filtered_data['Symbol'] == selected_symbol]
        if selected_type != "All":
            filtered_data = filtered_data[filtered_data['Type'] == selected_type]
        if selected_expiry != "All":
            filtered_data = filtered_data[filtered_data['Expiry'] == selected_expiry]
        filtered_data = filtered_data[(filtered_data["Strike"] >= min_strike) & (filtered_data["Strike"] <= max_strike)]
        filtered_data = filtered_data[(filtered_data["Volume"] >= min_volume) & (filtered_data["Volume"] <= max_volume)]
        filtered_data = filtered_data[(filtered_data["Bid"] >= min_bid) & (filtered_data["Bid"] <= max_bid)]
        filtered_data = filtered_data[(filtered_data["Ask"] >= min_ask) & (filtered_data["Ask"] <= max_ask)]

        filtered_data['In the Money'] = filtered_data.apply(
            lambda row: "Yes" if get_in_the_money_filter(filtered_data, row["Symbol"], price_cache.get(row["Symbol"]))(
                row) else "No", axis=1)

        filtered_data["Current Price"] = filtered_data["Symbol"].apply(
            lambda symbol: format_price(price_cache.get(symbol)) if price_cache.get(symbol) else "N/A")

        # Filter by "In/Out of the Money" selection
        if selected_in_out != "All":
            if selected_in_out == "In the Money":
                filtered_data = filtered_data[filtered_data['In the Money'] == "Yes"]
            elif selected_in_out == "Out of the Money":
                filtered_data = filtered_data[filtered_data['In the Money'] == "No"]

        # Sort data by Volume in descending order
        if not filtered_data.empty:
            filtered_data = filtered_data.sort_values(by="Volume", ascending=False)

        # Display data
        if not filtered_data.empty:
            with table_placeholder.container():
                st.dataframe(filtered_data.reset_index(drop=True),
                            column_config={
                            "Bid": st.column_config.NumberColumn(format="%.2f"),
                            "Ask": st.column_config.NumberColumn(format="%.2f"),
                            "Spread": st.column_config.NumberColumn(format="%.2f")
                            },
                            use_container_width=True)
            error_container.empty()
        else:
            with table_placeholder.container():
                st.warning("No data available with the selected filters.")

        return filtered_data

    # Initial data load
    update_data(error_container)

    # Auto-refresh loop
    if auto_refresh:
        while True:
            with st.spinner("Updating prices and data..."):
                update_prices_in_cache(SYMBOLS, error_container)
                update_data(error_container)
            time.sleep(10)
            error_container.empty()

    # Refresh button if auto-refresh is not selected
    if not auto_refresh:
        if st.sidebar.button("Refresh Now"):
            with st.spinner("Fetching data..."):
                update_prices_in_cache(SYMBOLS, error_container)
                update_data(error_container)
            error_container.empty()

    # Periodic price updates
    if not auto_refresh:
        if "last_price_update" not in st.session_state:
            st.session_state["last_price_update"] = datetime.now()

        if (datetime.now() - st.session_state["last_price_update"]).total_seconds() >= LIVE_PRICE_INTERVAL:
            with st.spinner("Updating prices..."):
                update_prices_in_cache(SYMBOLS, error_container)
                update_data(error_container)
                st.session_state["last_price_update"] = datetime.now()
            error_container.empty()


elif page == "Stock Chart":
     st.title("Stock Chart")
     # --- Sidebar ---
     st.sidebar.header("Chart Options")

     # Load Data
     load_data()
     error_container = st.empty()

     if not is_cache_valid():
        with st.spinner("Fetching initial options data..."):
            data, all_expiries = fetch_options_data(SYMBOLS, error_container)
            save_data()
            error_container.empty()
     else:
        with st.spinner("Loading from cache..."):
            data, all_expiries = fetch_options_data(SYMBOLS, error_container)
            error_container.empty()

     # Update prices in background
     update_prices_in_cache(SYMBOLS, error_container)

     # Dropdowns
     unique_symbols = sorted(data["Symbol"].unique()) if not data.empty else []
     selected_chart_symbol = st.sidebar.selectbox("Select Stock Symbol",  unique_symbols)

     time_frames = {
          "1 Minute": "1m",
          "5 Minutes": "5m",
          "15 Minutes": "15m",
          "1 Day": "1d",
          "5 Days": "5d",
          "1 Month": "1mo",
          "3 Months": "3mo",
          "6 Months": "6mo",
          "1 Year": "1y",
          "2 Year": "2y",
          "5 Year": "5y",
          "Max": "max"
      }

     selected_timeframe = st.sidebar.selectbox("Select Time Frame", list(time_frames.keys()))
     selected_timeframe_value = time_frames[selected_timeframe]

     include_after_hours = st.sidebar.checkbox("Include After-Hours Data")

     if selected_chart_symbol:
       chart_placeholder = st.empty()
       current_price = price_cache.get(selected_chart_symbol)
       st.markdown(f"<h1 style='text-align: center;'>{current_price:.2f}</h1>", unsafe_allow_html=True)

       create_candlestick_chart(selected_chart_symbol, chart_placeholder, selected_timeframe_value, include_after_hours)
     else:
       st.warning("Please select a stock to display")
