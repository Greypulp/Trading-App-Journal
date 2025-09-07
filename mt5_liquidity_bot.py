# mt5_liquidity_bot.py
# MT5 Liquidity Sweep Bot with Trading and Risk Management (pause new entries)
import time

## License key requirement removed
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import threading
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import warnings
import os, json

import requests
import xml.etree.ElementTree as ET

from datetime import timedelta

# ========== CONFIG ==========

# ========== NEWS BLOCKING ==========
NEWS_WINDOW_MINUTES = 5  # Block trading 5 min before/after news
NEWS_EVENTS_PATH = os.path.join(os.path.dirname(__file__), 'news_events.json')  # Placeholder for news events

def load_news_events():
    """
    Load news events from a local JSON file. Each event should be a dict with 'time' (ISO8601 string, UTC) and 'impact' (e.g., 'high', 'medium', 'low').
    Example: [{"time": "2025-09-01T13:30:00Z", "impact": "high", "symbol": "XAUUSD"}]
    """
    try:
        with open(NEWS_EVENTS_PATH, 'r') as f:
            events = json.load(f)
        # Convert to datetime
        for e in events:
            e['dt'] = datetime.fromisoformat(e['time'].replace('Z', '+00:00'))
        return events
    except Exception as e:
        print(f"Could not load news events: {e}")
        return []

def fetch_and_update_news_events():
    """
    Fetch news events from Forex Factory XML feed and update news_events.json.
    Only high/medium impact news for XAUUSD (USD, XAU, and global events).
    """
    url = 'https://nfs.faireconomy.media/ff_calendar_thisweek.xml'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        events = []
        for item in root.findall('event'):
            impact = item.find('impact').text.lower() if item.find('impact') is not None else ''
            if impact not in ('high', 'medium'):
                continue
            currency = item.find('currency').text if item.find('currency') is not None else ''
            # Only USD, XAU, or global news for XAUUSD
            if currency not in ('USD', 'XAU', 'ALL'):
                continue
            date_str = item.find('date').text if item.find('date') is not None else ''
            time_str = item.find('time').text if item.find('time') is not None else ''
            # Parse date/time to UTC
            # Forex Factory times are in New York time; need to convert to UTC
            from datetime import datetime as dt, timedelta
            import pytz
            ny_tz = pytz.timezone('America/New_York')
            try:
                # Example: date_str = 'Sep 1', time_str = '10:00am'
                year = datetime.now().year
                dt_naive = dt.strptime(f"{date_str} {year} {time_str}", "%b %d %Y %I:%M%p")
                dt_ny = ny_tz.localize(dt_naive)
                dt_utc = dt_ny.astimezone(timezone.utc)
                iso_time = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                continue
            events.append({
                'time': iso_time,
                'impact': impact,
                'symbol': 'XAUUSD',
                'title': item.find('title').text if item.find('title') is not None else ''
            })
        # Save to file
        with open(NEWS_EVENTS_PATH, 'w') as f:
            json.dump(events, f, indent=2)
        print(f"Fetched and updated {len(events)} news events.")
    except Exception as e:
        print(f"Failed to fetch news events: {e}")

def is_news_time(symbol):
    """
    Returns True if now is within NEWS_WINDOW_MINUTES before/after a news event for the symbol.
    """
    now = datetime.now(timezone.utc)
    events = load_news_events()
    for e in events:
        if e.get('symbol') and e['symbol'] != symbol:
            continue
        dt = e['dt']
        if abs((now - dt).total_seconds()) <= NEWS_WINDOW_MINUTES * 60:
            return True
    return False

SYMBOLS = ["XAUUSD"]
TIMEFRAMES = [mt5.TIMEFRAME_M3, mt5.TIMEFRAME_M5]
BARS_HISTORY = 2000

# Swing settings (Pine parity)
swingSizeR = 10
swingSizeL = 15

# Trading / Risk settings
ENABLE_TRADING = True   # must be set True to send real orders
TEST_MODE = False         # if True do not place real orders (prints only)
MAX_DAILY_DRAWDOWN_PCT = 0.1    # pause entries when daily loss (realized) >= this percent

# --- Load max_total_exposure from config.json ---
def get_max_total_exposure_lots():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return float(config.get('max_total_exposure', 5.0))
    except Exception:
        return 5.0

MAX_TOTAL_EXPOSURE_LOTS = get_max_total_exposure_lots()  # do not open new trades if total lots >= this
DEFAULT_SL_PIPS = 50
DEFAULT_TP_MULT = 4.0
MAGIC = 123456
DEVIATION = 20


# --- Standard risk settings ---
STANDARD_RISK_PER_TRADE_PCT = 0.1  # 0.1% per trade
MAX_RISK_PER_SYMBOL_PCT = 0.1     # 0.1% max risk per symbol per day

def get_max_risk_per_trade_pct():
    return STANDARD_RISK_PER_TRADE_PCT



# Telegram settings
TELEGRAM_BOT_TOKEN = '8177282153:AAHf7HJlNwUG23JMJa9dvnEstihel68VjPU'
TELEGRAM_CHAT_ID = '8426349009'
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Store pending trade for approval
pending_trade = {}

import asyncio
async def approval_callback(update, context: ContextTypes.DEFAULT_TYPE):
    global pending_trade
    query = update.callback_query
    await query.answer()
    if query.data == 'approve' and pending_trade:
        # Execute trade
        trade = pending_trade
        send_order(**trade)
        await query.edit_message_text(text="Trade Approved and Executed.")
        pending_trade = {}
    elif query.data == 'decline' and pending_trade:
        await query.edit_message_text(text="Trade Declined.")
        pending_trade = {}

application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
application.add_handler(CallbackQueryHandler(approval_callback))
def run_telegram_bot():
    application.run_polling()

# Start Telegram bot in a background thread

# Suppress specific warnings globally
warnings.filterwarnings("ignore", message="python-telegram-bot is using upstream urllib3.")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")

# Send Telegram message on bot startup
def send_startup_telegram():
    import datetime
    msg = f"MT5 Liquidity Bot activated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        # Use the running Application's event loop for sending messages
        from telegram.ext import Application
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        async def send():
            try:
                await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            except Exception as e:
                print(f"Startup Telegram notification failed: {e}")
        import asyncio
        asyncio.run(send())
    except Exception as e:
        print(f"Startup Telegram notification failed (refactored): {e}")

send_startup_telegram()

# ========== DATA STRUCTURES ==========
@dataclass
class Level:
    kind: str
    price: float
    pivot_index: int
    active: bool = True
    filled: bool = False

# ========== MT5 HELPERS ==========
def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account:
        print(f"[MT5] Connected to live account: {account.login}, Balance: {account.balance}")
    else:
        print("[MT5] Connected, but no account info available.")
    for sym in SYMBOLS:
        if not mt5.symbol_select(sym, True):
            raise RuntimeError(f"Symbol select failed for {sym}")

def fetch_rates(symbol, n_bars=2000):
    # Default to M15 if not specified
    timeframe = mt5.TIMEFRAME_M15
    if hasattr(fetch_rates, 'timeframe'):
        timeframe = fetch_rates.timeframe
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def account_balance():
    ai = mt5.account_info()
    return ai.balance if ai else None

def today_realized_pnl():
    # realized PnL from closed deals today (UTC)
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    deals = mt5.history_deals_get(start.timestamp(), time.time())
    if not deals:
        return 0.0
    pnl = 0.0
    for d in deals:
        pnl += getattr(d, 'profit', 0.0)
    return float(pnl)

def current_exposure_lots(symbol):
    pos = mt5.positions_get(symbol=symbol)
    if not pos:
        return 0.0
    return sum([p.volume for p in pos])

def lot_size_by_risk(symbol, balance, stop_pips, risk_pct):
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.01
    point = info.point
    digits = info.digits
    pip = 0.0001 if digits >= 4 else 0.01
    try:
        pip_value_per_lot = info.trade_contract_size * pip
    except Exception:
        pip_value_per_lot = 10.0
    risk_amount = balance * (risk_pct / 100.0)
    stop_value_per_lot = abs(stop_pips) * pip_value_per_lot
    if stop_value_per_lot <= 0:
        return 0.01
    lots = risk_amount / stop_value_per_lot
    step = info.volume_step if info.volume_step and info.volume_step > 0 else 0.01
    lots = max(info.volume_min if info.volume_min else 0.01, round(lots / step) * step)
    return float(lots)

def send_order(symbol, side, volume, price, sl, tp, comment=''):
    if TEST_MODE or not ENABLE_TRADING:
        print(f"[TEST ORDER] {side} {volume} @ {price} SL={sl} TP={tp} comment={comment}")
        return None
    order_type = mt5.ORDER_TYPE_BUY if side=='buy' else mt5.ORDER_TYPE_SELL
    req = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': float(volume),
        'type': order_type,
        'price': float(price),
        'deviation': DEVIATION,
        'magic': MAGIC,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,
        'sl': float(sl),
        'tp': float(tp)
    }
    res = mt5.order_send(req)
    print('order_send result:', res)

    # Send Telegram notification (async)
    import asyncio
    def safe_telegram_send():
        import asyncio
        async def send():
            try:
                strategy_info = comment if comment else 'Unknown strategy'
                msg = (
                    f"Trade Activated\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Volume: {volume}\n"
                    f"Price: {price}\n"
                    f"SL: {sl}  TP: {tp}\n"
                    f"Strategy: {strategy_info}"
                )
                await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            except Exception as e:
                print(f"Telegram notification failed: {e}")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading
                threading.Thread(target=lambda: asyncio.run(send())).start()
            else:
                loop.run_until_complete(send())
        except RuntimeError:
            asyncio.run(send())
        except Exception as e:
            print(f"Telegram notification failed (asyncio): {e}")
    safe_telegram_send()
    return res

 # Telegram bot and approval logic removed

# ========== Pivots (simple) ==========
def find_pivots(df, L, R):
    h = df['high'].values; l = df['low'].values; n = len(df)
    piv_hi = [False]*n; piv_lo = [False]*n
    for i in range(L, n-R):
        window_h = h[i-L:i+R+1]; window_l = l[i-L:i+R+1]
        if np.nanmax(window_h) == h[i] and (window_h==h[i]).sum()==1:
            piv_hi[i] = True
        if np.nanmin(window_l) == l[i] and (window_l==l[i]).sum()==1:
            piv_lo[i] = True
    return piv_hi, piv_lo

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

# ========== ENGINE ==========
class Engine:
    def find_fvg(self, df):
        # Returns list of FVGs as tuples: (start_idx, end_idx, high, low)
        fvg_list = []
        for i in range(2, len(df)):
            prev_high = df['high'].iat[i-2]
            prev_low = df['low'].iat[i-2]
            curr_high = df['high'].iat[i]
            curr_low = df['low'].iat[i]
            # Bullish FVG: previous high < current low
            if prev_high < curr_low:
                fvg_list.append((i-2, i, prev_high, curr_low))
            # Bearish FVG: previous low > current high
            if prev_low > curr_high:
                fvg_list.append((i-2, i, curr_high, prev_low))
        return fvg_list
    def __init__(self):
        self.levels = deque(maxlen=500)

    def scan_and_maybe_trade(self):
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                # Block trading during news window
                if is_news_time(symbol):
                    print(f"Trading blocked for {symbol}: within {NEWS_WINDOW_MINUTES} min of news event.")
                    continue
                df = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_HISTORY)
                if df is None:
                    continue
                df = pd.DataFrame(df)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
                if df.empty:
                    continue
                piv_hi, piv_lo = find_pivots(df, swingSizeL, swingSizeR)
                # detect pivots and append
                for i in range(len(piv_hi)):
                    if piv_hi[i]:
                        price = float(df['high'].iat[i]); self.levels.append((symbol, Level('high', price, i)))
                    if piv_lo[i]:
                        price = float(df['low'].iat[i]); self.levels.append((symbol, Level('low', price, i)))
                # check latest bar for fills and trade
                last = df.iloc[-1]
                highv = float(last['high']); lowv = float(last['low'])
                fvg_list = self.find_fvg(df)
                for sym, lvl in list(self.levels):
                    if sym != symbol or not lvl.active: continue
                    # --- Order block rejection and candle close confluence ---
                    touched = highv >= lvl.price and lowv <= lvl.price
                    if touched:
                        prev = df.iloc[-2] if len(df) > 1 else None
                        if prev is not None:
                            # --- FVG rejection confluence ---
                            fvg_rejected = False
                            for fvg in fvg_list:
                                _, _, fvg_high, fvg_low = fvg
                                if (lowv <= fvg_high and highv >= fvg_low):
                                    if prev['close'] < fvg_low or prev['close'] > fvg_high:
                                        fvg_rejected = True
                                        break
                            # --- Order block rejection confluence ---
                            ob_rejected = False
                            if lvl.kind == 'high' and prev['close'] < lvl.price:
                                ob_rejected = True
                            elif lvl.kind == 'low' and prev['close'] > lvl.price:
                                ob_rejected = True
                            # --- Break of Structure Confluence ---
                            bos_ok = False
                            if lvl.kind == 'high':
                                recent_highs = df['high'].iloc[-10:-1] if len(df) > 10 else df['high'].iloc[:-1]
                                bos_ok = highv > recent_highs.max()
                            elif lvl.kind == 'low':
                                recent_lows = df['low'].iloc[-10:-1] if len(df) > 10 else df['low'].iloc[:-1]
                                bos_ok = lowv < recent_lows.min()

                            # --- Engulfing Candle Confluence ---
                            engulf_ok = False
                            # The candle that caused the liquidity sweep is the previous candle (prev)
                            # Engulfing: current candle's body fully engulfs previous candle's body
                            if prev is not None:
                                curr_open = last['open']; curr_close = last['close']
                                prev_open = prev['open']; prev_close = prev['close']
                                curr_body_high = max(curr_open, curr_close)
                                curr_body_low = min(curr_open, curr_close)
                                prev_body_high = max(prev_open, prev_close)
                                prev_body_low = min(prev_open, prev_close)
                                if curr_body_high > prev_body_high and curr_body_low < prev_body_low:
                                    engulf_ok = True

                            # --- Swing Match Confluence ---
                            swing_match_ok = False
                            # Check if recent swing high/low equals any older swing high/low
                            if lvl.kind == 'high':
                                recent_idx = None
                                for j in range(len(piv_hi)-2, -1, -1):
                                    if piv_hi[j]:
                                        recent_idx = j
                                        break
                                if recent_idx is not None:
                                    recent_swing_high = float(df['high'].iat[recent_idx])
                                    for k in range(recent_idx-1, -1, -1):
                                        if piv_hi[k]:
                                            older_swing_high = float(df['high'].iat[k])
                                            if np.isclose(recent_swing_high, older_swing_high, atol=1e-5):
                                                swing_match_ok = True
                                                break
                            elif lvl.kind == 'low':
                                recent_idx = None
                                for j in range(len(piv_lo)-2, -1, -1):
                                    if piv_lo[j]:
                                        recent_idx = j
                                        break
                                if recent_idx is not None:
                                    recent_swing_low = float(df['low'].iat[recent_idx])
                                    for k in range(recent_idx-1, -1, -1):
                                        if piv_lo[k]:
                                            older_swing_low = float(df['low'].iat[k])
                                            if np.isclose(recent_swing_low, older_swing_low, atol=1e-5):
                                                swing_match_ok = True
                                                break

                            # --- Final trade setup trigger: any confluence satisfied ---
                            confluence_ok = (ob_rejected or fvg_rejected or engulf_ok or swing_match_ok)
                            if bos_ok and confluence_ok:
                                lvl.filled = True
                                self.attempt_trade_on_fill(sym, lvl)
                                lvl.active = False

    def attempt_trade_on_fill(self, symbol, lvl):
        global pending_trade
        # Allow up to 2 open positions per symbol, but block if any is in drawdown
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions and len(open_positions) >= 2:
            print(f"Trade blocked: already have 2 open positions for {symbol}.")
            return
        # Block new trade if any open position is in drawdown
        if open_positions:
            for pos in open_positions:
                # Drawdown: current price less favorable than entry (for buys, price < entry; for sells, price > entry)
                tick = mt5.symbol_info_tick(symbol)
                if pos.type == mt5.ORDER_TYPE_BUY and tick.bid < pos.price_open:
                    print(f"Trade blocked: open BUY position in drawdown for {symbol}.")
                    return
                elif pos.type == mt5.ORDER_TYPE_SELL and tick.ask > pos.price_open:
                    print(f"Trade blocked: open SELL position in drawdown for {symbol}.")
                    return
        balance = account_balance()
        if balance is None:
            print('No account info; skipping trade'); return
        realized = today_realized_pnl()
        realized_pct = (realized / balance) * 100.0 if balance else 0.0
        if realized_pct <= -MAX_DAILY_DRAWDOWN_PCT:
            print(f'Trade blocked: daily realized loss {realized_pct:.2f}% <= -{MAX_DAILY_DRAWDOWN_PCT}%'); return
        exposure = current_exposure_lots(symbol)
        if exposure >= MAX_TOTAL_EXPOSURE_LOTS:
            print(f'Trade blocked: exposure {exposure} lots >= {MAX_TOTAL_EXPOSURE_LOTS}'); return
        info = mt5.symbol_info(symbol)
        digits = info.digits
        pip = 0.0001 if digits >= 4 else 0.01
        # Calculate ATR-based stop loss
        df_atr = fetch_rates(symbol, 100)
        atr_mult = 1.5  # ATR multiplier for stop loss
        if df_atr.empty:
            print('ATR data unavailable; skipping trade.')
            return
        atr = calculate_atr(df_atr).iloc[-1]
        stop_pips = atr * atr_mult / pip
        tick = mt5.symbol_info_tick(symbol)
        entry = tick.ask if lvl.kind=='low' else tick.bid
        side = 'buy' if lvl.kind=='low' else 'sell'

        # --- Calculate total risk for this symbol today ---
        # For simplicity, assume only one trade per fill, but sum up open positions risk

        # (Retain risk calculation for other checks, but not for position count)
        open_risk_pct = 0.0
        if open_positions:
            for pos in open_positions:
                # Estimate risk % for each open position (approximate)
                pos_lots = pos.volume
                pos_stop_pips = abs(pos.price_open - pos.sl) / pip if pos.sl else DEFAULT_SL_PIPS
                pos_risk = (pos_lots * pos_stop_pips * info.trade_contract_size * pip) / balance * 100.0
                open_risk_pct += pos_risk

        # Standard risk per trade
        risk_pct = STANDARD_RISK_PER_TRADE_PCT
        # Cap risk so total for symbol does not exceed MAX_RISK_PER_SYMBOL_PCT
        if open_risk_pct + risk_pct > MAX_RISK_PER_SYMBOL_PCT:
            print(f'Trade blocked: would exceed max risk {MAX_RISK_PER_SYMBOL_PCT}% for {symbol} (open: {open_risk_pct:.2f}%)'); return

        lots = lot_size_by_risk(symbol, balance, stop_pips, risk_pct)
        if exposure + lots > MAX_TOTAL_EXPOSURE_LOTS:
            print('Trade blocked: would exceed total exposure'); return

        # --- Calculate risk in $ ---
        pip_val = info.trade_contract_size * (0.0001 if info.digits >= 4 else 0.01)
        risk_amount = lots * stop_pips * pip_val

        # --- Find next pivot for TP ---
        df = fetch_rates(symbol, BARS_HISTORY)
        piv_hi, piv_lo = find_pivots(df, swingSizeL, swingSizeR)
        idx = df.index.get_loc(df.index[-1])
        if side == 'buy':
            # Next pivot high after entry bar
            next_pivot_idx = next((i for i in range(idx+1, len(piv_hi)) if piv_hi[i]), None)
            tp = float(df['high'].iloc[next_pivot_idx]) if next_pivot_idx is not None else entry + stop_pips * pip * DEFAULT_TP_MULT
            sl = entry - stop_pips * pip
        else:
            # Next pivot low after entry bar
            next_pivot_idx = next((i for i in range(idx+1, len(piv_lo)) if piv_lo[i]), None)
            tp = float(df['low'].iloc[next_pivot_idx]) if next_pivot_idx is not None else entry - stop_pips * pip * DEFAULT_TP_MULT
            sl = entry + stop_pips * pip

        # --- Telegram approval removed: auto-approve all trades ---
        # --- Telegram manual approval with inline buttons ---
        trade_details = f"Trade Setup:\nSymbol: {symbol}\nSide: {side}\nLots: {lots}\nEntry: {entry}\nSL: {sl}\nTP: {tp}"
        keyboard = [
            [InlineKeyboardButton("Approve", callback_data='approve'), InlineKeyboardButton("Decline", callback_data='decline')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=trade_details, reply_markup=reply_markup)
            pending_trade = {
                'symbol': symbol,
                'side': side,
                'volume': lots,
                'price': entry,
                'sl': sl,
                'tp': tp,
                'comment': "swing_fill_riskctrl"
            }
            print(f"Trade setup sent for approval via Telegram.")
        except Exception as e:
            print(f"Telegram approval send failed: {e}")

# --- Monitor open trades and move SL to BE when price moves in favor by SL distance ---

# ========== RUN ==========

def main():
    # Fetch and update news events at startup
    try:
        fetch_and_update_news_events()
    except Exception as e:
        print(f"News fetch failed: {e}")
    # Write 'running' to bot_status.txt on start
    status_path = os.path.join(os.path.dirname(__file__), 'bot_status.txt')
    try:
        with open(status_path, 'w') as f:
            f.write('running\n')
    except Exception:
        pass
    init_mt5(); eng = Engine()

    # If bot is logged and active, send a summary of trade logic for the day to Telegram
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                bot_status = f.read().strip().lower()
        except Exception:
            bot_status = 'unknown'
        if bot_status == 'running':
            # Prepare updated trade logic summary
            trade_summary = (
                "Trade Logic Summary for Today:\n"
                "- Symbol: XAUUSD\n"
                "- Timeframes: 3m, 5m\n"
                "- Confluences (any triggers a trade, plus break of structure):\n"
                "    • Order block rejection\n"
                "    • FVG rejection\n"
                "    • Engulfing candle\n"
                "    • Recent swing high/low equals older swing high/low\n"
                "- Risk Controls:\n"
                "    • Max daily drawdown: 0.1%\n"
                "    • Max total exposure: 5 lots\n"
                "    • Standard risk per trade: 0.1%\n"
                "    • Max risk per symbol per day: 0.1%\n"
                "- Manual trade approval via Telegram inline buttons\n"
                "- Trading paused during news events\n"
            )
            import asyncio
            async def send_summary_signal():
                try:
                    await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=trade_summary)
                except Exception as e:
                    print(f"Telegram summary failed: {e}")
            asyncio.run(send_summary_signal())

    def is_market_open():
        # XAUUSD: open Sun 22:00 UTC, close Fri 21:00 UTC (typical for gold/forex)
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        minute = now.minute
        # Market closed from Fri 21:00 UTC to Sun 22:00 UTC
        if (weekday == 4 and (hour > 21 or (hour == 21 and minute >= 0))) or (weekday == 5) or (weekday == 6 and (hour < 22)):
            return False
        return True

    market_open = None  # Unknown at start
    import asyncio
    async def send_telegram_market_open():
        try:
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text='Market Opened. Bot Activated.')
        except Exception as e:
            print(f"Telegram notification failed: {e}")

    async def send_telegram_market_closed():
        try:
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text='Market Closed. Bot Paused.')
        except Exception as e:
            print(f"Telegram notification failed: {e}")
    try:
        while True:
            currently_open = is_market_open()
            if market_open is None:
                market_open = currently_open
            if currently_open:
                if not market_open:
                    # Market just opened
                    print("Market opened. Trading resumed.")
                    try:
                        asyncio.run(send_telegram_market_open())
                    except Exception as e:
                        print(f"Telegram notification failed (asyncio): {e}")
                eng.scan_and_maybe_trade()
            else:
                if market_open:
                    # Market just closed
                    print("Market closed. Trading paused.")
                    try:
                        asyncio.run(send_telegram_market_closed())
                    except Exception as e:
                        print(f"Telegram notification failed (asyncio): {e}")
                # Do not trade when market is closed
            market_open = currently_open
            time.sleep(30)
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        # Write 'stopped' to bot_status.txt on exit
        try:
            with open(status_path, 'w') as f:
                f.write('stopped\n')
        except Exception:
            pass
        mt5.shutdown()

if __name__ == "__main__":
    main()
