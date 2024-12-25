import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import time
from datetime import datetime
import threading

# Optional schedule library
try:
    import schedule
except ImportError:
    pass

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Flow & Stock Tracker", layout="wide")

# --------------------------------------------------
# GLOBALS
# --------------------------------------------------
DB_NAME = "options_data.db"
SYMBOLS = ["AAPL", "TSLA", "MSFT", "AMZN", "SPY", "QQQ", "NVDA", "META", "GOOGL"]

# Initialize session state variables
if "scheduler_running" not in st.session_state:
    st.session_state.scheduler_running = False
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "stock_refresh" not in st.session_state:
    st.session_state.stock_refresh = 0
if "alert_ratio" not in st.session_state:
    st.session_state.alert_ratio = 2.0
if "alert_diff" not in st.session_state:
    st.session_state.alert_diff = 1000

# --------------------------------------------------
# HELPER: MULTISELECT WITH "ALL"
# --------------------------------------------------
def multiselect_with_all(label, options, sidebar=True):
    """
    A helper function that shows a multiselect with an "All" item.
    If user picks "All," we return the full list of actual items.
    Otherwise, we return whatever is selected.
    """
    extended = ["All"] + sorted(options)
    default_val = ["All"]
    if sidebar:
        selection = st.sidebar.multiselect(label, extended, default=default_val)
    else:
        selection = st.multiselect(label, extended, default=default_val)

    if "All" in selection:
        return list(options)
    else:
        return [x for x in selection if x != "All"]


# --------------------------------------------------
# 1. DATABASE SETUP
# --------------------------------------------------
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        # Table for options snapshots
        c.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshots (
                snapshot_time TEXT,
                symbol TEXT,
                type TEXT,
                expiry TEXT,
                strike REAL,
                volume REAL,
                bid REAL,
                ask REAL,
                current_price REAL
            )
        """)
        # Table for alerts
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_time TEXT,
                symbol TEXT,
                type TEXT,
                expiry TEXT,
                strike REAL,
                old_volume REAL,
                new_volume REAL,
                volume_diff REAL,
                volume_ratio REAL,
                bid REAL,
                ask REAL,
                current_price REAL
            )
        """)
        conn.commit()

def store_snapshot(df: pd.DataFrame, snapshot_time: str):
    if df.empty:
        return
    with sqlite3.connect(DB_NAME) as conn:
        df["snapshot_time"] = snapshot_time
        df_to_store = df[[
            "snapshot_time","Symbol","Type","Expiry",
            "Strike","Volume","Bid","Ask","current_price"
        ]].copy()
        df_to_store.to_sql("options_snapshots", conn, if_exists="append", index=False)

def store_alerts(df_alerts: pd.DataFrame):
    if df_alerts.empty:
        return
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        for _, row in df_alerts.iterrows():
            c.execute("""
                INSERT INTO alerts (
                    alert_time, symbol, type, expiry, strike,
                    old_volume, new_volume, volume_diff, volume_ratio,
                    bid, ask, current_price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                row["Symbol"],
                row["Type"],
                row["Expiry"],
                row["Strike"],
                row.get("Volume_old", 0),
                row.get("Volume", 0),
                row.get("Volume_Diff", 0),
                row.get("Volume_Ratio", 0),
                row.get("Bid", 0),
                row.get("Ask", 0),
                row.get("current_price", 0),
            ))
        conn.commit()

def get_all_snapshots():
    query = """
        SELECT snapshot_time,
               symbol AS Symbol,
               type AS Type,
               expiry AS Expiry,
               strike AS Strike,
               volume AS Volume,
               bid AS Bid,
               ask AS Ask,
               current_price AS current_price
          FROM options_snapshots
    """
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query(query, conn)
    return df

def get_latest_snapshot_time():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT MAX(snapshot_time) FROM options_snapshots")
        row = c.fetchone()
        return row[0] if row and row[0] else None

def get_snapshot_data(snapshot_time: str):
    query = """
        SELECT symbol AS Symbol,
               type AS Type,
               expiry AS Expiry,
               strike AS Strike,
               volume AS Volume,
               bid AS Bid,
               ask AS Ask,
               current_price AS current_price
          FROM options_snapshots
         WHERE snapshot_time = ?
    """
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query(query, conn, params=[snapshot_time])
    return df

def get_alert_history(limit=50):
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query(f"SELECT * FROM alerts ORDER BY id DESC LIMIT {limit}", conn)
    return df

# --------------------------------------------------
# 2. FETCH OPTIONS DATA
# --------------------------------------------------
@st.cache_data
def fetch_options_data(symbols, refresh_count):
    import time
    import yfinance as yf
    all_rows = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            cp = ticker.info.get("regularMarketPrice", None)
            if cp is None:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    cp = hist["Close"].iloc[-1]

            if not ticker.options:
                continue

            for expiry in ticker.options:
                opt_chain = ticker.option_chain(expiry)
                for opt_type, chain_df in [("Call", opt_chain.calls), ("Put", opt_chain.puts)]:
                    for _, row in chain_df.iterrows():
                        all_rows.append({
                            "Symbol": symbol,
                            "Type": opt_type,
                            "Expiry": expiry,
                            "Strike": row.get("strike", 0),
                            "Volume": row.get("volume", 0),
                            "Bid": row.get("bid", 0),
                            "Ask": row.get("ask", 0),
                            "current_price": cp,
                        })
            time.sleep(0.2)
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.sort_values("Volume", ascending=False, inplace=True)
    return df

# --------------------------------------------------
# 3. UNUSUAL VOLUME DETECTION
# --------------------------------------------------
def find_unusual_volume(new_df, old_df, ratio_thr, diff_thr):
    if new_df.empty or old_df.empty:
        return pd.DataFrame([])

    keys = ["Symbol","Type","Expiry","Strike"]
    merged = pd.merge(new_df, old_df, on=keys, how="outer", suffixes=("", "_old"))

    merged["Volume"] = merged["Volume"].fillna(0)
    merged["Volume_old"] = merged["Volume_old"].fillna(0)

    merged["Volume_Diff"] = merged["Volume"] - merged["Volume_old"]
    def ratio_func(row):
        return row["Volume"] / row["Volume_old"] if row["Volume_old"]>0 else float("inf")
    merged["Volume_Ratio"] = merged.apply(ratio_func, axis=1)

    cond = (merged["Volume_Ratio"]>=ratio_thr) & (merged["Volume_Diff"]>=diff_thr)
    unusual = merged[cond].copy()
    if unusual.empty:
        return unusual

    keep_cols = [
        "Symbol","Type","Expiry","Strike",
        "Volume_old","Volume","Volume_Diff","Volume_Ratio",
        "Bid","Ask","current_price"
    ]
    existing = [c for c in keep_cols if c in unusual.columns]
    unusual = unusual[existing].sort_values("Volume_Diff", ascending=False)
    return unusual

def handle_unusual_volume(new_data, old_data):
    ratio_thr = st.session_state.alert_ratio
    diff_thr = st.session_state.alert_diff
    alerts = find_unusual_volume(new_data, old_data, ratio_thr, diff_thr)
    if not alerts.empty:
        store_alerts(alerts)
    return alerts

# --------------------------------------------------
# 4. BACKGROUND SCHEDULER
# --------------------------------------------------
def background_fetch_job():
    while True:
        interval_minutes = 1
        try:
            fetch_options_data.clear()
            new_df = fetch_options_data(SYMBOLS, st.session_state.refresh_count)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not new_df.empty:
                old_time = get_latest_snapshot_time()
                old_df = pd.DataFrame([])
                if old_time:
                    old_df = get_snapshot_data(old_time)

                store_snapshot(new_df, now_str)
                handle_unusual_volume(new_df, old_df)
            else:
                print("[Scheduler] Empty snapshot...")

            print(f"[Scheduler] Snapshot stored at {now_str}, total rows: {len(new_df)}")
        except Exception as e:
            print(f"[Scheduler] error: {e}")

        for _ in range(interval_minutes*6):
            time.sleep(10)

def start_scheduler():
    if st.session_state.scheduler_running:
        st.write("Scheduler is already running.")
        return
    st.session_state.scheduler_running = True
    thread = threading.Thread(target=background_fetch_job, daemon=True)
    thread.start()
    st.write("Background scheduler thread started.")

# --------------------------------------------------
# PAGE 1: OPTIONS FLOW TRACKER
# --------------------------------------------------
def page_options_flow():
    st.title("Options Flow Tracker")

    # We have 3 tabs in this order:
    # 1) Flow Data
    # 2) Alert History
    # 3) Settings (includes old 'Settings' + 'Alert Settings')

    tab1, tab2, tab3 = st.tabs(["Flow Data", "Alert History", "Settings"])

    # ---- Tab1: Flow Data
    with tab1:
        st.subheader("Flow Data")
        df_all = get_all_snapshots()
        if df_all.empty:
            st.info("No snapshots in DB yet. Go to the 'Settings' tab to fetch data.")
            return

        st.sidebar.header("Filters")
        # Symbol
        syms = multiselect_with_all("Symbols", df_all["Symbol"].unique())
        # Type
        types = multiselect_with_all("Option Types", df_all["Type"].unique())
        # Expiry
        exps = multiselect_with_all("Expiries", df_all["Expiry"].unique())
        # Strike
        all_strikes = df_all["Strike"].unique().astype(str)
        chosen_strikes = multiselect_with_all("Strikes", all_strikes)

        filtered = df_all[
            df_all["Symbol"].isin(syms) &
            df_all["Type"].isin(types) &
            df_all["Expiry"].isin(exps)
        ].copy()

        filtered["Strike_str"] = filtered["Strike"].astype(str)
        filtered = filtered[filtered["Strike_str"].isin(chosen_strikes)]

        if filtered.empty:
            st.warning("No data matching these filters.")
        else:
            filtered.drop_duplicates(
                subset=["Symbol","Type","Expiry","Strike","Volume","Bid","Ask","current_price"],
                inplace=True
            )
            filtered = filtered.sort_values("Volume", ascending=False)
            filtered.rename(columns={"Expiry":"Expiry Date"}, inplace=True)

            final_cols = [
                "Symbol","current_price","Strike","Bid","Ask","Type","Expiry Date","Volume"
            ]
            existing = [c for c in final_cols if c in filtered.columns]
            disp = filtered[existing].copy()
            disp.rename(columns={"current_price":"Current Price"}, inplace=True)
            disp.drop(columns=["Strike_str"], errors="ignore", inplace=True)

            st.dataframe(disp.reset_index(drop=True))

    # ---- Tab2: Alert History
    with tab2:
        st.subheader("Alert History")
        df_alerts = get_alert_history(limit=100)
        if df_alerts.empty:
            st.info("No alerts triggered yet.")
        else:
            st.dataframe(df_alerts)

    # ---- Tab3: Settings (combined old “Settings” + “Alert Settings”)
    with tab3:
        st.subheader("Background Scheduler & Manual Snapshot")

        if st.button("Start Background Scheduler"):
            start_scheduler()

        if st.button("Fetch Snapshot Now"):
            fetch_options_data.clear()
            new_data = fetch_options_data(SYMBOLS, st.session_state.refresh_count)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if new_data.empty:
                st.warning("No data returned.")
            else:
                old_time = get_latest_snapshot_time()
                old_df = pd.DataFrame([])
                if old_time:
                    old_df = get_snapshot_data(old_time)

                store_snapshot(new_data, now_str)
                st.success(f"Snapshot stored at {now_str}, total rows: {len(new_data)}")

                # Check for unusual volume => alerts
                alerts = handle_unusual_volume(new_data, old_df)
                if alerts.empty:
                    st.info("No unusual volume found.")
                else:
                    st.success(f"Detected {len(alerts)} unusual volume rows.")
                    st.dataframe(alerts)

        if st.button("Clear Cache"):
            fetch_options_data.clear()
            st.success("Options data cache cleared.")

        st.write("---")
        st.write("**Alert Settings**")
        ratio_val = st.number_input("Volume Ratio Threshold (e.g. 2 = 2x)",
                                    min_value=1.0, value=st.session_state.alert_ratio, step=0.5)
        diff_val = st.number_input("Absolute Volume Increase Threshold",
                                   min_value=1, value=st.session_state.alert_diff, step=100)

        if st.button("Save Alert Settings"):
            st.session_state.alert_ratio = ratio_val
            st.session_state.alert_diff = diff_val
            st.success(f"Saved ratio={ratio_val} diff={diff_val}")

        st.write(f"**Current Ratio**: {st.session_state.alert_ratio}, "
                 f"**Current Diff**: {st.session_state.alert_diff}")

        st.write("---")
        st.write(f"**Refresh Count (session)**: {st.session_state.refresh_count}")
        if st.session_state.scheduler_running:
            st.info("Background scheduler is running in a separate thread.")

# --------------------------------------------------
# PAGE 2: STOCK CHART
# --------------------------------------------------
@st.cache_data
def load_stock_data(symbol, period, interval, after_hours, refresh_counter):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, prepost=after_hours)
    return df

def page_stock_chart():
    st.header("Stock Chart & Price (Line or Candlestick)")
    st.markdown("""
    View the price history for a selected symbol.  
    You can adjust the time period, interval, whether to include after-hours data,  
    and choose between a **Line** or **Candlestick** chart.
    """)

    st.sidebar.subheader("Filters (Stock Chart)")

    symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=2)  # default "MSFT"
    valid_periods = ["1d","5d","1mo","6mo","1y","5y","max"]
    period = st.sidebar.selectbox("Period", valid_periods, index=1)
    valid_intervals = ["1m","5m","15m","30m","1h","1d","1wk","1mo"]
    interval = st.sidebar.selectbox("Interval", valid_intervals, index=5)
    after_hours = st.sidebar.checkbox("Include After-Hours Data?", value=False)
    chart_type = st.sidebar.radio("Chart Type", ["Line","Candlestick"], index=1)

    if st.sidebar.button("Refresh Now"):
        st.session_state.stock_refresh += 1

    auto_refresh = st.sidebar.checkbox("Auto-refresh every 15 seconds", value=False)

    if st.sidebar.button("Clear Cache"):
        load_stock_data.clear()
        st.success("Stock data cache cleared.")

    df = load_stock_data(symbol, period, interval, after_hours, st.session_state.stock_refresh)

    if df.empty:
        st.warning(f"No data found for {symbol} using period='{period}' and interval='{interval}'.")
        return

    last_price = df["Close"].iloc[-1]
    st.write(f"**Symbol**: {symbol} | **Last Price**: {last_price:.2f}")

    if chart_type == "Line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
        fig.update_layout(
            title=f"{symbol} - {period} ({interval})",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name=symbol
                )
            ]
        )
        fig.update_layout(
            title=f"{symbol} - {period} ({interval})",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Latest 10 Data Points**")
    st.dataframe(df.tail(10))

    if auto_refresh:
        time.sleep(15)
        st.session_state.stock_refresh += 1

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    init_db()

    pages = ["Options Flow Tracker", "Stock Chart"]
    chosen_page = st.sidebar.selectbox("Navigation", pages, index=0)

    if chosen_page == "Options Flow Tracker":
        page_options_flow()
    else:
        page_stock_chart()

    st.write("---")
    st.write(f"**Refresh Count (session):** {st.session_state.refresh_count}")
    if st.session_state.scheduler_running:
        st.info("Background scheduler is running in a separate thread.")

if __name__ == "__main__":
    main()
