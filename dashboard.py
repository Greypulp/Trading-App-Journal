# --- Next Market Open Helper ---
def get_next_market_open():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # Monday=0, Sunday=6
    hour = now.hour
    minute = now.minute
    # Market closed from Fri 21:00 UTC to Sun 22:00 UTC
    if (weekday == 4 and (hour > 21 or (hour == 21 and minute >= 0))) or (weekday == 5) or (weekday == 6 and (hour < 22)):
        # Next open is Sunday 22:00 UTC
        days_ahead = (6 - weekday) % 7
        next_open = now.replace(hour=22, minute=0, second=0, microsecond=0) + pd.Timedelta(days=days_ahead)
        if now > next_open:
            next_open += pd.Timedelta(days=7)
        return next_open
    else:
        # Market is open, next open is after next close
        # Find next Friday 21:00 UTC
        days_ahead = (4 - weekday) % 7
        next_close = now.replace(hour=21, minute=0, second=0, microsecond=0) + pd.Timedelta(days=days_ahead)
        if now >= next_close:
            # Already past this week's close, so next week's open
            next_open = next_close + pd.Timedelta(days=2, hours=1)
            return next_open
        else:
            # Market is open, so no countdown
            return None
# --- Trading Session Helper ---
def get_active_session():
    # Sessions in UTC
    now = datetime.now(timezone.utc)
    hour = now.hour
    # Asia: 00:00-07:59, London: 08:00-15:59, New York: 16:00-20:59, Closed: 21:00-23:59
    if 0 <= hour < 8:
        return 'Asia'
    elif 8 <= hour < 16:
        return 'London'
    elif 16 <= hour < 21:
        return 'New York'
    else:
        return 'Closed'
# --- Session Status Helper ---
def get_session_status():
    bot_status_path = os.path.join(os.path.dirname(__file__), 'bot_status.txt')
    if os.path.exists(bot_status_path):
        try:
            with open(bot_status_path, 'r') as f:
                status = f.read().strip().lower()
            if status == 'running':
                return 'Active'
            elif status == 'stopped':
                return 'Inactive'
        except Exception:
            pass
    return 'Unknown'
# --- Market Status Helper ---
def get_market_status():
    # XAUUSD: open Sun 22:00 UTC, close Fri 21:00 UTC (typical for gold/forex)
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # Monday=0, Sunday=6
    hour = now.hour
    minute = now.minute
    # Market closed from Fri 21:00 UTC to Sun 22:00 UTC
    if (weekday == 4 and (hour > 21 or (hour == 21 and minute >= 0))) or (weekday == 5) or (weekday == 6 and (hour < 22)):
        return 'Closed'
    return 'Open'
 # dashboard.py (Streamlit) - includes risk management metrics and live limits
import os, time, json
import sys

## License key requirement removed
from datetime import datetime, timezone, date, timedelta
import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import plotly.graph_objects as go
st.set_page_config(page_title="MT5 Dashboard", layout="wide")

# --- Timeframe Map (must be above sidebar controls) ---
TIMEFRAME_MAP = {'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1}


with st.sidebar:
    st.title("Controls & Risk Settings")
    # --- MT5 Connection Status ---
    mt5_status = mt5.initialize() if not mt5.initialize() else True
    if mt5_status:
        st.success("🟢 MT5 Connected", icon="✅")
    else:
        st.error("🔴 MT5 Not Connected", icon="❌")
    # --- Bot Connection Status ---
    bot_status_path = os.path.join(os.path.dirname(__file__), 'bot_status.txt')
    bot_status = 'unknown'
    if os.path.exists(bot_status_path):
        try:
            with open(bot_status_path, 'r') as f:
                bot_status = f.read().strip().lower()
        except Exception:
            bot_status = 'unknown'
    if bot_status == 'running':
        st.success("🤖 Bot Running", icon="✅")
    elif bot_status == 'stopped':
        st.error("🤖 Bot Stopped", icon="❌")
    else:
        st.warning("🤖 Bot Status Unknown", icon="⚠️")
    selected_tab = st.radio("Select View", ["Dashboard", "Trading Journal", "Backtesting"], key="main_tab")
    if selected_tab == "Backtesting":
        pass
    else:
        symbol = st.selectbox("Symbol", ["XAUUSD"], index=0, key="symbol_select")
        timeframe = st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=0, key="tf_select")
        bars = st.slider("Bars to Load", min_value=500, max_value=3000, value=2000, step=100, key="bars_slider")
        st.markdown("---")
        st.subheader("Risk Controls")
        max_daily_dd = st.number_input("Max Daily Drawdown (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.1, key="max_dd_input")
        max_risk_trade = st.number_input("Max Risk per Trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="max_risk_input")
        max_total_exposure = st.number_input("Max Total Exposure (lots)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, key="max_exposure_input")
        st.markdown("---")
        st.subheader("Pivot Detection Settings")
        L = st.slider("Pivot Left Bars (L)", min_value=5, max_value=30, value=15, step=1, key="pivot_L")
        R = st.slider("Pivot Right Bars (R)", min_value=5, max_value=30, value=10, step=1, key="pivot_R")
# --- Page Routing ---
if selected_tab == "Backtesting":
    from backtest_logic import PivotATRBacktester
    st.title("📊 Strategy Backtesting")
    st.markdown("""
    Upload your historical data CSV and test your strategy logic. Results and metrics will be displayed below.
    """)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="backtest_csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        st.subheader("Column Mapping")
        columns = df.columns.tolist()
        col_time = st.selectbox("Time column", columns, index=0, key="bt_col_time")
        col_open = st.selectbox("Open column", columns, index=1 if len(columns)>1 else 0, key="bt_col_open")
        col_high = st.selectbox("High column", columns, index=2 if len(columns)>2 else 0, key="bt_col_high")
        col_low = st.selectbox("Low column", columns, index=3 if len(columns)>3 else 0, key="bt_col_low")
        col_close = st.selectbox("Close column", columns, index=4 if len(columns)>4 else 0, key="bt_col_close")
        df_bt = df[[col_time, col_open, col_high, col_low, col_close]].copy()
        df_bt.columns = ['time', 'open', 'high', 'low', 'close']
        df_bt['time'] = pd.to_datetime(df_bt['time'])
        df_bt.set_index('time', inplace=True)
        st.subheader("Backtest Parameters")
        swing_left = st.number_input("Swing Left (pivot lookback)", min_value=1, max_value=50, value=15, key="bt_swing_left")
        swing_right = st.number_input("Swing Right (pivot lookforward)", min_value=1, max_value=50, value=10, key="bt_swing_right")
        atr_period = st.number_input("ATR Period", min_value=1, max_value=100, value=14, key="bt_atr_period")
        atr_mult = st.number_input("ATR Multiplier (for SL)", min_value=0.1, max_value=10.0, value=1.5, step=0.1, key="bt_atr_mult")
        risk_pct = st.number_input("Risk % per Trade", min_value=0.01, max_value=10.0, value=0.5, step=0.01, key="bt_risk_pct")
        # Use MT5 live balance if available
        live_balance = None
        try:
            import MetaTrader5 as mt5
            acc = mt5.account_info()
            if acc:
                live_balance = acc.balance
        except Exception:
            pass
        initial_balance = st.number_input(
            "Initial Balance",
            min_value=100.0,
            max_value=1000000.0,
            value=live_balance if live_balance is not None else 10000.0,
            step=100.0,
            key="bt_initial_balance"
        )
        contract_size = st.number_input("Contract Size", min_value=1, max_value=100000, value=100, key="bt_contract_size")
        pip_size = st.number_input("Pip Size", min_value=0.00001, max_value=1.0, value=0.01, step=0.00001, format="%f", key="bt_pip_size")
        min_lot = st.number_input("Min Lot Size", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key="bt_min_lot")
        if st.button("Run Backtest", key="bt_run_btn"):
            with st.spinner("Running backtest..."):
                bt = PivotATRBacktester(
                    df_bt,
                    swing_left=swing_left,
                    swing_right=swing_right,
                    atr_period=atr_period,
                    atr_mult=atr_mult,
                    risk_pct=risk_pct,
                    initial_balance=initial_balance,
                    contract_size=contract_size,
                    pip_size=pip_size,
                    min_lot=min_lot
                )
                trades = bt.run()
                st.success(f"Backtest complete. Final balance: {bt.balance:,.2f}")
                st.write(f"Total trades: {len(trades)}")
                st.dataframe(trades)
                if not trades.empty:
                    st.line_chart(trades['profit'].cumsum())
                    # --- Risk Metrics ---
                    wins = trades[trades['profit']>0]['profit'].sum() if not trades.empty else 0.0
                    losses = -trades[trades['profit']<0]['profit'].sum() if not trades.empty else 0.0
                    win_rate = (trades['profit']>0).mean()*100 if not trades.empty else 0.0
                    profit_factor = (wins / losses) if losses>0 else float('inf') if wins>0 else 0.0
                    eq = trades['profit'].cumsum() + initial_balance if not trades.empty else [initial_balance]
                    peak = np.maximum.accumulate(eq)
                    dd = ((peak - eq).max() / peak.max())*100 if len(eq)>0 and peak.max()>0 else 0.0
                    st.markdown("### Backtest Risk Metrics")
                    st.write(f"**Win Rate:** {win_rate:.2f}%")
                    st.write(f"**Profit Factor:** {profit_factor:.2f}")
                    st.write(f"**Max Drawdown:** {dd:.2f}%")
                    st.write(f"**Total Net Profit:** {eq.iloc[-1] - initial_balance:,.2f}")
    else:
        st.info("Please upload a CSV file to begin backtesting.")

DEFAULT_SYMBOL = "XAUUSD"
TIMEFRAME_MAP = {'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1}
BOX_WID = 0.7

def init_mt5():
    return mt5.initialize()

mt5_ok = init_mt5()

def fetch_rates(symbol, timeframe, n_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates); df['time']=pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    return df

def find_pivots(df, L, R):
    h = df['high'].values; l = df['low'].values; n = len(df)
    piv_hi = np.zeros(n, dtype=bool); piv_lo = np.zeros(n, dtype=bool)
    for i in range(L, n-R):
        window_h = h[i-L:i+R+1]; window_l = l[i-L:i+R+1]
        if np.nanmax(window_h)==h[i] and (window_h==h[i]).sum()==1: piv_hi[i]=True
        if np.nanmin(window_l)==l[i] and (window_l==l[i]).sum()==1: piv_lo[i]=True
    return pd.Series(piv_hi, index=df.index), pd.Series(piv_lo, index=df.index)

def get_account_info():
    acc = mt5.account_info()
    if acc: return {'Balance': acc.balance, 'Equity': acc.equity, 'Margin': acc.margin, 'Free': acc.margin_free}
    return None

def get_open_positions_df(symbol):
    pos = mt5.positions_get(symbol=symbol); 
    if not pos: return pd.DataFrame()
    rows=[] 
    for p in pos: rows.append([p.ticket, p.symbol, p.volume, p.price_open, p.sl, p.tp, p.profit, datetime.fromtimestamp(p.time, timezone.utc)])
    return pd.DataFrame(rows, columns=['Ticket','Symbol','Volume','Open','SL','TP','Profit','Time'])

def get_recent_trades_df(days=30):

    now = time.time()
    frm = now - days * 86400
    deals = mt5.history_deals_get(frm, now)
    if not deals:
        return pd.DataFrame()
    rows = []
    for d in deals:
        rows.append([d.ticket, d.symbol, d.volume, d.price, d.profit, datetime.fromtimestamp(d.time, timezone.utc)])
    return pd.DataFrame(rows, columns=['Ticket','Symbol','Volume','Price','Profit','Time'])


    # ...existing code...

with st.sidebar.expander('Chart Display Options', expanded=True):
    st.write('Showing only today\'s data.')
    st.caption('Adjust symbol, timeframe, and bars above.')

# --- Account Details in Sidebar ---
acc = get_account_info()
if acc:
    st.sidebar.markdown('---')
    st.sidebar.subheader('💼 MT5 Account Details')
    st.sidebar.write(f"**Balance:** ${acc['Balance']:.2f}")
    st.sidebar.write(f"**Equity:** ${acc['Equity']:.2f}")
    st.sidebar.write(f"**Margin:** ${acc['Margin']:.2f}")
    st.sidebar.write(f"**Free Margin:** ${acc['Free']:.2f}")
else:
    st.sidebar.warning('No account info available.')


# Only run live data code if required variables are defined (i.e., not on Backtesting page)
if all(x in globals() for x in ['timeframe', 'symbol', 'bars']):
    tf = TIMEFRAME_MAP[timeframe]
    df = fetch_rates(symbol, tf, bars)
    if df.empty:
        st.error('No data from MT5'); st.stop()
    # Ensure df.index is timezone-aware UTC for safe comparison
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    # Filter df to only today's data using UTC+3:00 as the trading day start
    import pytz
    tz_moscow = timezone.utc if not hasattr(pytz, 'timezone') else pytz.timezone('Etc/GMT-3')
    now_utc = datetime.now(timezone.utc)
    now = now_utc.astimezone(tz_moscow)
    today_start = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if now.hour < 3:
        today_start = today_start - pd.Timedelta(days=1)
    today_start_utc = today_start.astimezone(timezone.utc)
    next_day_start_utc = (today_start + pd.Timedelta(days=1)).astimezone(timezone.utc)
    df_today = df[(df.index >= today_start_utc) & (df.index < next_day_start_utc)]
    if df_today.empty:
        # Find the most recent previous day with data (using UTC+3 day boundaries)
        prev_day = today_start - pd.Timedelta(days=1)
        while prev_day > df.index.min():
            prev_start_utc = prev_day.astimezone(timezone.utc)
            prev_end_utc = (prev_day + pd.Timedelta(days=1)).astimezone(timezone.utc)
            day_df = df[(df.index >= prev_start_utc) & (df.index < prev_end_utc)]
            if not day_df.empty:
                df_today = day_df
                st.warning(f"No data for today. Showing data for {prev_day.strftime('%Y-%m-%d')} (UTC+3) instead.")
                break
            prev_day -= pd.Timedelta(days=1)
        else:
            st.error('No data for today or any previous day from MT5'); st.stop()
    df = df_today



# --- Data Preparation ---
if selected_tab == "Dashboard":
    # --- Auto-refresh dashboard every 60 seconds ---
    import streamlit as st
    import time
    if 'last_refresh' not in st.session_state:
        st.session_state['last_refresh'] = time.time()
    if time.time() - st.session_state['last_refresh'] > 60:
        st.session_state['last_refresh'] = time.time()
        st.rerun()

    # --- Countdown timer to next session open ---
    def get_next_session_open():
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        # Sessions: Asia 00:00-07:59, London 08:00-15:59, NY 16:00-20:59, Closed 21:00-23:59
        if 0 <= hour < 8:
            # Next: London 08:00 UTC
            next_open = now.replace(hour=8, minute=0, second=0, microsecond=0)
        elif 8 <= hour < 16:
            # Next: New York 16:00 UTC
            next_open = now.replace(hour=16, minute=0, second=0, microsecond=0)
        elif 16 <= hour < 21:
            # Next: Closed 21:00 UTC
            next_open = now.replace(hour=21, minute=0, second=0, microsecond=0)
        else:
            # Next: Asia 00:00 UTC (next day)
            next_open = (now + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if next_open <= now:
            next_open += pd.Timedelta(days=1)
        return next_open

    next_session_open = get_next_session_open()
    if next_session_open:
        delta = next_session_open - datetime.now(timezone.utc)
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        st.info(f"⏳ Time to Next Session Open: {hours}h {minutes}m {seconds}s (at {next_session_open.strftime('%Y-%m-%d %H:%M UTC')})")

    # --- Bell notification when market opens (only once per open event) ---
    market_status = get_market_status()
    if 'prev_market_status' not in st.session_state:
        st.session_state['prev_market_status'] = market_status
    play_bell = False
    if st.session_state['prev_market_status'] == 'Closed' and market_status == 'Open':
        play_bell = True
    st.session_state['prev_market_status'] = market_status
    if play_bell:
        bell_js = """
        <script>
        var audio = new Audio('https://cdn.pixabay.com/audio/2022/07/26/audio_124bfae5b6.mp3');
        audio.volume = 0.7;
        audio.play();
        </script>
        """
        st.markdown(bell_js, unsafe_allow_html=True)

    piv_hi, piv_lo = find_pivots(df, L, R)

    positions_df = get_open_positions_df(symbol)
    trades_df = get_recent_trades_df(days=30)
    balance = acc['Balance'] if acc else 0.0

    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    deals_today = mt5.history_deals_get(start.timestamp(), time.time()) or []
    realized_today = sum(getattr(d,'profit',0.0) for d in deals_today)
    realized_pct = (realized_today / balance * 100.0) if balance else 0.0

    closed = trades_df[trades_df['Symbol']==symbol] if not trades_df.empty else pd.DataFrame()
    lastN = 50; sample = closed.tail(lastN)
    wins = sample[sample['Profit']>0]['Profit'].sum() if not sample.empty else 0.0
    losses = -sample[sample['Profit']<0]['Profit'].sum() if not sample.empty else 0.0
    win_rate = (sample['Profit']>0).mean()*100 if not sample.empty else 0.0
    profit_factor = (wins / losses) if losses>0 else float('inf') if wins>0 else 0.0

    # --- Weekly P&L and Account Growth ---
    week_ago = now - pd.Timedelta(days=7)
    deals_week = [d for d in mt5.history_deals_get(week_ago.timestamp(), time.time()) or []]
    weekly_pnl = sum(getattr(d, 'profit', 0.0) for d in deals_week)

    # Estimate balance 7 days ago (current balance - closed P&L since then)
    balance_7d_ago = balance - weekly_pnl
    account_growth = ((balance - balance_7d_ago) / balance_7d_ago * 100.0) if balance_7d_ago else 0.0

    try:
        eq = (sample['Profit'].cumsum() + balance).ffill().values if not sample.empty else np.array([balance])
        peak = np.maximum.accumulate(eq); dd = ((peak - eq).max() / peak.max())*100 if len(eq)>0 and peak.max()>0 else 0.0
    except Exception:
        dd = 0.0

    exposure_lots = positions_df['Volume'].sum() if not positions_df.empty else 0.0

    # --- Example lot size calculation for current risk setting ---
    info = mt5.symbol_info(symbol); digits = info.digits if info else 5
    pip = 0.0001 if digits>=4 else 0.01
    default_sl = 50
    def example_lot_for_risk(balance, stop_pips, risk_pct):
        try:
            contract = info.trade_contract_size if info and hasattr(info,'trade_contract_size') else 100000
            pip_val = contract * (0.0001 if digits>=4 else 0.01)
            risk_amount = balance * (risk_pct/100.0)
            lot = risk_amount / (stop_pips * pip_val)
            return max(0.01, round(lot,2)), risk_amount, pip_val
        except Exception:
            return 0.01, 0.0, 0.0

    example_lot, example_risk_usd, example_pip_val = example_lot_for_risk(balance, default_sl, max_risk_trade)

    # --- Display autocalculated lot size and formula in sidebar ---
    with st.sidebar.expander('🧮 Lot Size Calculation', expanded=False):
        st.write(f"**Autocalculated lot size:** <span style='color:#2563eb;font-size:1.2em'>{example_lot:.2f}</span>", unsafe_allow_html=True)
        st.write(f"**Risk in $:** <span style='color:#eab308;font-size:1.1em'>${example_risk_usd:,.2f}</span> (for {max_risk_trade:.2f}% of balance)", unsafe_allow_html=True)
        st.caption(f"For risk = {max_risk_trade:.2f}% of balance, SL = {default_sl} pips.")
        st.markdown("""
        **Formula:**
        
        `lot = (balance * risk% / 100) / (stop_loss_pips * pip_value_per_lot)`
        
        Where:
        - balance = account balance
        - risk% = max risk per trade (from above)
        - stop_loss_pips = default 50
        - pip_value_per_lot = contract size × pip size
        """, unsafe_allow_html=True)



# --- Page Routing ---
if selected_tab == "Dashboard":
    # --- Streamlit-native Summary Section ---
    with st.container():
        summary_cols = st.columns(7)
        summary_cols[0].metric('💰 Balance', f"${balance:,.2f}")    
        summary_cols[1].metric('💵 Realized P&L Today', f"${realized_today:,.2f}", delta=f"{realized_pct:.2f}%", delta_color="normal" if realized_today>=0 else "inverse")
        summary_cols[2].metric('📆 Weekly P&L', f"${weekly_pnl:,.2f}")
        summary_cols[3].metric('✅ Win Rate', f"{win_rate:.1f}%")
        summary_cols[4].metric('📈 Account Growth (7d)', f"{account_growth:.2f}%")
        # Market Status
        market_status = get_market_status()
        if market_status == 'Open':
            summary_cols[5].success('🟢 Market Open')
        else:
            summary_cols[5].error('🔴 Market Closed')
        # Session Status
        session_status = get_session_status()
        if session_status == 'Active':
            summary_cols[6].success('🟢 Session Active')
        elif session_status == 'Inactive':
            summary_cols[6].error('🔴 Session Inactive')
        else:
            summary_cols[6].warning('⚠️ Session Unknown')

        # Show which session is active
        active_session = get_active_session()
        if active_session == 'Asia':
            st.info('🌏 Asia Session Active (00:00-07:59 UTC)')
        elif active_session == 'London':
            st.info('🇬🇧 London Session Active (08:00-15:59 UTC)')
        elif active_session == 'New York':
            st.info('🇺🇸 New York Session Active (16:00-20:59 UTC)')
        else:
            st.warning('⏸️ No Major Session Active (21:00-23:59 UTC)')
        extra_cols = st.columns(4)
        extra_cols[0].metric('🧮 Example Lot', f"{example_lot:.2f}")
        extra_cols[1].metric('📊 Exposure (lots)', f"{exposure_lots:.2f}")
        extra_cols[2].metric('🛡️ Drawdown %', f"{dd:.2f}%")
        extra_cols[3].metric('📊 Profit Factor', f"{profit_factor:.2f}")
        st.markdown('<hr style="margin-top:8px;margin-bottom:8px;border:0;border-top:1px solid #e0e7ef;">', unsafe_allow_html=True)
        col1, col2 = st.columns([2,1], gap="small")
        with col1:
            st.header('📈 Today\'s Price Chart')
            st.caption('Live price chart with pivots and risk overlays. Data auto-refreshes.')
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            hi_idx = np.where(piv_hi.values)[0]; lo_idx = np.where(piv_lo.values)[0]
            if len(hi_idx):
                fig.add_trace(go.Scatter(x=df.index[hi_idx], y=df['high'].iloc[hi_idx], mode='markers', marker=dict(color='red', size=8), name='Pivot Highs'))
            if len(lo_idx):
                fig.add_trace(go.Scatter(x=df.index[lo_idx], y=df['low'].iloc[lo_idx], mode='markers', marker=dict(color='green', size=8), name='Pivot Lows'))
            for idx in hi_idx:
                price = df['high'].iat[idx]; top,bottom = price*(1+0.001*BOX_WID), price; t0=df.index[idx]; t1=df.index[-1]
                fig.add_shape(type='rect', x0=t0, x1=t1, y0=bottom, y1=top, fillcolor='rgba(200,50,50,0.12)', line_width=0)
            for idx in lo_idx:
                price = df['low'].iat[idx]; top,bottom = price, price*(1-0.001*BOX_WID); t0=df.index[idx]; t1=df.index[-1]
                fig.add_shape(type='rect', x0=t0, x1=t1, y0=bottom, y1=top, fillcolor='rgba(50,180,80,0.12)', line_width=0)
            fig.update_layout(
                xaxis_rangeslider_visible=True,
                xaxis_title='Time',
                yaxis_title='Price',
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor='#f8fafc',
                paper_bgcolor='#f8fafc',
                font=dict(size=13)
            )
            st.plotly_chart(fig, width='stretch')
        with col2:
            st.subheader('📂 Open Positions')
            if not positions_df.empty:
                df_pos = positions_df.copy()
                if 'Time' in df_pos.columns:
                    df_pos['Time'] = df_pos['Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(
                    df_pos,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        'Time': st.column_config.Column('Time', width='medium', help='Open time (UTC)'),
                        'Profit': st.column_config.Column('Profit', width='small', help='Unrealized profit in $'),
                        'Volume': st.column_config.Column('Volume', width='small', help='Position size (lots)'),
                    }
                )
            else:
                st.info('No open positions')

    # --- Footer ---
    st.markdown('---')
    st.markdown('<div style="text-align:center; color:#888; font-size:0.95em;">MT5 Dashboard &copy; 2025 &mdash; Powered by Streamlit | For support, contact your developer.</div>', unsafe_allow_html=True)

    if st.button('Shutdown MT5', help='Disconnect and close MT5 terminal'):
        mt5.shutdown(); st.rerun()


elif selected_tab == "Trading Journal":
    notes_path = os.path.join(os.path.dirname(__file__), 'journal_notes.json')
    if 'journal_notes' not in st.session_state:
        if os.path.exists(notes_path):
            try:
                with open(notes_path, 'r') as f:
                    st.session_state['journal_notes'] = json.load(f)
            except Exception:
                st.session_state['journal_notes'] = {}
        else:
            st.session_state['journal_notes'] = {}

    # --- Card-style header ---
    st.markdown("""
        <div style="background:linear-gradient(90deg,#0f172a 0%,#2563eb 100%);padding:24px 32px 18px 32px;border-radius:18px;margin-bottom:18px;box-shadow:0 2px 12px rgba(0,0,0,0.07);color:#fff;">
            <div style="font-size:2.1em;font-weight:700;letter-spacing:0.5px;">📓 Trading Journal</div>
            <div style="font-size:1.1em;font-weight:400;opacity:0.92;">Calendar view of daily P&L from MT5 live data.<br><span style='color:#4ade80;'>Green = profit</span>, <span style='color:#f87171;'>Red = loss</span>.</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        .journal-calendar-card {
            background: #f8fafc;
            border-radius: 18px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            padding: 24px 18px 18px 18px;
            margin-bottom: 18px;
        }
        .journal-cell {
            transition: box-shadow 0.2s, border 0.2s;
            cursor: pointer;
        }
        .journal-cell:hover {
            box-shadow: 0 0 0 2px #2563eb33;
            border: 1.5px solid #2563eb;
        }
        .journal-date {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 2px;
        }
        .notes-section {
            background: #f1f5f9;
            border-radius: 12px;
            padding: 18px 18px 10px 18px;
            margin-bottom: 18px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        }
        .notes-title {
            font-size: 1.2em;
            font-weight: 700;
            color: #2563eb;
            margin-bottom: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Month and Year selectors ---
    import calendar
    from datetime import date, timedelta
    import json
    today = date.today()
    min_year = 2020
    max_year = today.year
    years = list(range(min_year, max_year+1))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    col_month, col_year = st.columns([2,1])
    selected_month = col_month.selectbox('Month', options=months, index=today.month-1, key='journal_month_select')
    selected_year = col_year.selectbox('Year', options=years[::-1], index=0, key='journal_year_select')
    month_idx = months.index(selected_month) + 1
    first_day = date(selected_year, month_idx, 1)
    start_day = first_day - timedelta(days=first_day.weekday())
    days = [start_day + timedelta(days=i) for i in range(35)]

    # --- Fetch trades for the selected month and year ---
    from calendar import monthrange
    from datetime import datetime, timezone
    month_days = monthrange(selected_year, month_idx)[1]
    month_start = datetime(selected_year, month_idx, 1, 0, 0, 0, tzinfo=timezone.utc)
    month_end = datetime(selected_year, month_idx, month_days, 23, 59, 59, tzinfo=timezone.utc)
    deals = mt5.history_deals_get(month_start.timestamp(), month_end.timestamp())
    if deals:
        trades = pd.DataFrame([
            [d.ticket, d.symbol, d.volume, d.price, d.profit, datetime.fromtimestamp(d.time, timezone.utc)]
            for d in deals
        ], columns=['Ticket','Symbol','Volume','Price','Profit','Time'])
    else:
        trades = pd.DataFrame(columns=['Ticket','Symbol','Volume','Price','Profit','Time'])
    if not trades.empty:
        trades['Date'] = trades['Time'].dt.date
        daily_pnl = trades.groupby('Date')['Profit'].sum().reset_index()
        daily_pnl['Type'] = np.where(daily_pnl['Profit']>=0, 'Profit', 'Loss')
    else:
        daily_pnl = pd.DataFrame(columns=['Date','Profit','Type'])

    # Build a dict for fast lookup
    pnl_map = {row['Date']: row for _, row in daily_pnl.iterrows()}

    # Build a dict for fast lookup
    pnl_map = {row['Date']: row for _, row in daily_pnl.iterrows()}

    # --- Calendar grid with tooltips (static, not interactive) ---
    st.markdown('<div class="journal-calendar-card">', unsafe_allow_html=True)
    week_days = ['Mon','Tue','Wed','Thu','Fri']
    num_days = len(week_days)
    cols = st.columns(num_days)
    for i, wd in enumerate(week_days):
        cols[i].markdown(f"<div style='text-align:center;font-weight:bold;font-size:1.08em;color:#334155'>{wd}</div>", unsafe_allow_html=True)
    for week in range(5):
        cols = st.columns(num_days)
        for day in range(num_days):
            d = days[week*7+day]
            if d.weekday() > 4:
                continue
            cell = ''
            date_str = f"<div class='journal-date'>{d.day}</div>"
            note_str = st.session_state['journal_notes'].get(d.strftime('%Y-%m-%d'), "")
            def escape_html(text):
                import html
                return html.escape(text)
            tooltip = ""
            if d.month == month_idx and d.year == selected_year:
                if d in pnl_map:
                    row = pnl_map[d]
                    if row['Type']=='Profit':
                        tooltip = f"Profit: ${row['Profit']:.2f}"
                        font_color = '#059669'  # deep green
                    else:
                        tooltip = f"Loss: ${-row['Profit']:.2f}"
                        font_color = '#dc2626'  # deep red
                    if note_str:
                        tooltip += f"\nNote: {note_str}"
                    tooltip_html = escape_html(tooltip).replace('\n', '&#10;')
                    cell = f"<div style='position:relative'><div class='journal-cell' title='{tooltip_html}' style='background:#f8fafc;color:{font_color};border-radius:8px;padding:4px 0 2px 0;font-weight:700;font-size:1.08em;'>{date_str}{'Profit: $' if row['Type']=='Profit' else 'Loss: $'}{abs(row['Profit']):.2f}</div></div>"
                else:
                    tooltip = note_str if note_str else ""
                    tooltip_html = escape_html(tooltip).replace('\n', '&#10;')
                    cell = f"<div style='position:relative'><div class='journal-cell empty' title='{tooltip_html}'>{date_str}&nbsp;</div></div>"
            else:
                cell = f"<div style='position:relative'><div class='journal-cell empty'>{date_str}&nbsp;</div></div>"
            cols[day].markdown(cell, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # --- Day selector for notes ---
    valid_days = [d.day for d in days if d.month == month_idx and d.year == selected_year]
    selected_day = st.selectbox('Select Day for Notes', options=valid_days, index=valid_days.index(today.day) if today.month == month_idx and today.year == selected_year else 0, key='journal_day_select')

    # --- Notes section for selected date ---
    selected_date = date(selected_year, month_idx, selected_day)
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    st.markdown(f'<div class="notes-section"><div class="notes-title">📝 Notes for {selected_date_str}</div>', unsafe_allow_html=True)
    note_key = f"note_{selected_date_str}"
    current_note = st.session_state['journal_notes'].get(selected_date_str, "")
    new_note = st.text_area("Add/Edit Note", value=current_note, key=note_key)
    if st.button("Save Note", key=f"save_{note_key}"):
        st.session_state['journal_notes'][selected_date_str] = new_note
        try:
            with open(notes_path, 'w') as f:
                json.dump(st.session_state['journal_notes'], f, indent=2)
            st.success("Note saved for this day.")
        except Exception as e:
            st.error(f"Failed to save note: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.18em;font-weight:600;color:#334155;margin-bottom:8px;">🕒 Recent Trades</div>', unsafe_allow_html=True)
    if not trades.empty:
        # Show trades for the selected date if selected, else for the selected month
        selected_date_trades = trades[(trades['Time'].dt.date == selected_date)]
        if not selected_date_trades.empty:
            trades_sorted = selected_date_trades.sort_values('Time', ascending=False).copy()
            st.markdown(f'<div style="font-size:1.05em;color:#2563eb;font-weight:500;margin-bottom:4px;">Showing trades for <b>{selected_date.strftime("%Y-%m-%d")}</b></div>', unsafe_allow_html=True)
            if 'Time' in trades_sorted.columns:
                trades_sorted['Time'] = trades_sorted['Time'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(trades_sorted, width='stretch', hide_index=True)
        else:
            trades_month = trades[(trades['Time'].dt.month == month_idx) & (trades['Time'].dt.year == selected_year)]
            trades_sorted = trades_month.sort_values('Time', ascending=False).copy()
            st.markdown(f'<div style="font-size:1.05em;color:#2563eb;font-weight:500;margin-bottom:4px;">No trades for selected day. Showing all trades for <b>{selected_month} {selected_year}</b></div>', unsafe_allow_html=True)
            if 'Time' in trades_sorted.columns:
                trades_sorted['Time'] = trades_sorted['Time'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(trades_sorted, width='stretch', hide_index=True)
    # --- Date range selector for chart ---
    min_date = df.index.min().date() if not df.empty else date.today() - timedelta(days=30)
    max_date = df.index.max().date() if not df.empty else date.today()
    date_range = st.date_input(
        "Select date or range to view on chart",
        value=(max_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="chart_date_range"
    )
    # Filter df to selected date(s)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        df_chart = df[mask]
    else:
        mask = (df.index.date == date_range)
        df_chart = df[mask]
    # If no data for selected range, show warning
    if df_chart.empty:
        st.warning(f"No data for selected date(s): {date_range}")
    else:
        piv_hi_chart, piv_lo_chart = find_pivots(df_chart, L, R)
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'])])
        hi_idx = np.where(piv_hi_chart.values)[0]
        lo_idx = np.where(piv_lo_chart.values)[0]
        if len(hi_idx):
            fig.add_trace(go.Scatter(x=df_chart.index[hi_idx], y=df_chart['high'].iloc[hi_idx], mode='markers', marker=dict(color='red', size=8), name='Pivot Highs'))
        if len(lo_idx):
            fig.add_trace(go.Scatter(x=df_chart.index[lo_idx], y=df_chart['low'].iloc[lo_idx], mode='markers', marker=dict(color='green', size=8), name='Pivot Lows'))
        fig.update_layout(
            xaxis_rangeslider_visible=True,
            xaxis_title='Time',
            yaxis_title='Price',
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='#f8fafc',
            font=dict(size=13)
        )
        st.plotly_chart(fig, width='stretch')
