import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# -----------------------------
# SETUP & CONFIG
# -----------------------------
st.set_page_config(page_title="Options Flow & Stock Chart", layout="wide")

SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN", "SPY", "QQQ", "NVDA", "META", "GOOGL"]

# Make sure session_state has a refresh counter
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0


# -----------------------------
# DATA FETCHING
# -----------------------------
@st.cache_data(ttl=300)
def fetch_options_data(symbols):
    options_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('regularMarketPrice', None)
            if current_price is None:
                hist = ticker.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else None

            if not ticker.options:
                continue

            for expiry in ticker.options:
                opt_chain = ticker.option_chain(expiry)
                for otype, chain_df in [("Call", opt_chain.calls), ("Put", opt_chain.puts)]:
                    for _, row in chain_df.iterrows():
                        options_data.append({
                            "Symbol": symbol,
                            "Current Price": current_price,
                            "Strike": row['strike'],
                            "Bid": row['bid'],
                            "Ask": row['ask'],
                            "Type": otype,
                            "Expiry": expiry,
                            "Volume": row.get('volume', 0)
                        })

            time.sleep(0.2)  # slightly reduce Yahoo rate-limit risk

        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {e}")
            continue

    df = pd.DataFrame(options_data)
    if not df.empty:
        df.sort_values("Volume", ascending=False, inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_stock_history(symbol, period, interval, after_hours):
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval, prepost=after_hours)


# -----------------------------
# PAGE: OPTIONS TABLE
# -----------------------------
def show_options_table():
    st.header("Options Flow")
    st.markdown("""
    **This page displays options data for popular symbols.**  
    Use the filters on the sidebar to refine the data.
    """)

    # Fetch the data (cached)
    with st.spinner("Fetching options data..."):
        data = fetch_options_data(SYMBOLS)

    # Sidebar filters
    st.sidebar.subheader("Filters (Options Table)")
    unique_symbols = ["All"] + sorted(data["Symbol"].unique()) if not data.empty else ["All"]
    symbol = st.sidebar.selectbox("Symbol", unique_symbols)

    opt_types = ["All", "Call", "Put"]
    opt_type = st.sidebar.selectbox("Option Type", opt_types)

    unique_exp = ["All"] + sorted(data["Expiry"].unique()) if not data.empty else ["All"]
    expiry = st.sidebar.selectbox("Expiry Date (YYYY-MM-DD)", unique_exp)

    # Volume slider
    if data.empty or "Volume" not in data.columns:
        vmin, vmax = 0, 100000
    else:
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0)
        vmin, vmax = int(data["Volume"].min()), int(data["Volume"].max())
        if vmin > vmax:
            vmin, vmax = 0, 100000

    vol_range = st.sidebar.slider("Volume Range", vmin, vmax, (vmin, vmax))

    # Bid slider
    if "Bid" not in data.columns:
        bid_min, bid_max = 0.0, 1000.0
    else:
        data["Bid"] = pd.to_numeric(data["Bid"], errors="coerce").fillna(0)
        bmin, bmax = data["Bid"].min(), data["Bid"].max()
        if bmin is None or bmax is None or bmin > bmax:
            bmin, bmax = 0.0, 1000.0
        bid_min, bid_max = float(bmin), float(bmax)

    chosen_bid = st.sidebar.slider("Bid Range", 0.0, max(1000.0, bid_max), (bid_min, bid_max))

    # Ask slider
    if "Ask" not in data.columns:
        ask_min, ask_max = 0.0, 1000.0
    else:
        data["Ask"] = pd.to_numeric(data["Ask"], errors="coerce").fillna(0)
        amin, amax = data["Ask"].min(), data["Ask"].max()
        if amin is None or amax is None or amin > amax:
            amin, amax = 0.0, 1000.0
        ask_min, ask_max = float(amin), float(amax)

    chosen_ask = st.sidebar.slider("Ask Range", 0.0, max(1000.0, ask_max), (ask_min, ask_max))

    # Apply filters
    filtered = data.copy()
    if symbol != "All":
        filtered = filtered[filtered["Symbol"] == symbol]
    if opt_type != "All":
        filtered = filtered[filtered["Type"] == opt_type]
    if expiry != "All":
        filtered = filtered[filtered["Expiry"] == expiry]

    filtered = filtered[
        (filtered["Volume"] >= vol_range[0]) & (filtered["Volume"] <= vol_range[1])
    ]
    filtered = filtered[
        (filtered["Bid"] >= chosen_bid[0]) & (filtered["Bid"] <= chosen_bid[1])
    ]
    filtered = filtered[
        (filtered["Ask"] >= chosen_ask[0]) & (filtered["Ask"] <= chosen_ask[1])
    ]

    filtered.reset_index(drop=True, inplace=True)

    if not filtered.empty:
        st.dataframe(filtered, use_container_width=True)
        st.write(f"**Total filtered rows:** {len(filtered):,}")
    else:
        st.warning("No data available with the selected filters.")


# -----------------------------
# PAGE: STOCK CHART
# -----------------------------
def show_stock_chart():
    st.header("Stock Chart & Price (Candlestick)")
    st.markdown("""
    **View the price history for a selected symbol.**  
    You can adjust the time period, interval, and whether to include after-hours data (if available).
    """)

    st.sidebar.subheader("Filters (Stock Chart)")
    symbol = st.sidebar.selectbox("Symbol", SYMBOLS)

    valid_periods = ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"]
    valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]

    period = st.sidebar.selectbox("Period", valid_periods, index=3)     # default '6mo'
    interval = st.sidebar.selectbox("Interval", valid_intervals, index=5)  # default '1d'
    after_hours = st.sidebar.checkbox("Include After-Hours Data?", value=False)

    if symbol:
        with st.spinner("Loading stock chart..."):
            df = fetch_stock_history(symbol, period, interval, after_hours)

        if df.empty:
            st.warning(f"No price data found for {symbol}.")
        else:
            current_price = df["Close"].iloc[-1]
            st.write(f"**Symbol:** {symbol} | **Last Price:** {current_price:.2f}")
            df.reset_index(inplace=True)

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["Date"] if "Date" in df.columns else df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name=symbol
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Data preview (latest 10 records)**")
            st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("Please select a symbol from the sidebar.")


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.sidebar.title("Navigation")
    page_options = ["Options Table", "Stock Chart"]
    page_selection = st.sidebar.selectbox("Go to Page", page_options)

    if page_selection == "Options Table":
        show_options_table()
    else:
        show_stock_chart()

    # Show last updated and session refresh count
    st.write("---")
    st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Refresh Count:** {st.session_state.refresh_count}")

    # Put refresh controls at the bottom
    st.sidebar.write("---")

    # Button to clear cache & increment refresh count
    if st.sidebar.button("Refresh Now"):
        fetch_options_data.clear()
        fetch_stock_history.clear()
        # Increment refresh_count to trigger a re-run
        st.session_state.refresh_count += 1

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 15 seconds", value=False)
    if auto_refresh:
        # Sleep, then clear cache, increment refresh_count to trigger re-run
        time.sleep(15)
        fetch_options_data.clear()
        fetch_stock_history.clear()
        st.session_state.refresh_count += 1


if __name__ == "__main__":
    main()
