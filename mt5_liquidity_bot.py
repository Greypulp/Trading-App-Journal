# mt5_liquidity_bot.py
# MT5 Liquidity Sweep Bot with Trading and Risk Management (pause new entries)
import time
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import threading
from telegram import Bot
import warnings
import os, json

# ========== CONFIG ==========

SYMBOLS = ["XAUUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15
BARS_HISTORY = 2000

# Swing settings (Pine parity)
swingSizeR = 10
swingSizeL = 15

# Trading / Risk settings
ENABLE_TRADING = True   # must be set True to send real orders
TEST_MODE = False         # if True do not place real orders (prints only)
MAX_DAILY_DRAWDOWN_PCT = 5.0    # pause entries when daily loss (realized) >= this percent

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
DEFAULT_TP_MULT = 2.0
MAGIC = 123456
DEVIATION = 20


# --- Standard risk settings ---
STANDARD_RISK_PER_TRADE_PCT = 0.5  # 0.5% per trade
MAX_RISK_PER_SYMBOL_PCT = 1.0      # 1% max risk per symbol per day

def get_max_risk_per_trade_pct():
    return STANDARD_RISK_PER_TRADE_PCT


# Telegram settings
TELEGRAM_BOT_TOKEN = '8177282153:AAHf7HJlNwUG23JMJa9dvnEstihel68VjPU'
TELEGRAM_CHAT_ID = '8426349009'
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Suppress specific warnings globally
warnings.filterwarnings("ignore", message="python-telegram-bot is using upstream urllib3.")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")

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
    for sym in SYMBOLS:
        if not mt5.symbol_select(sym, True):
            raise RuntimeError(f"Symbol select failed for {sym}")

def fetch_rates(symbol, n_bars=2000):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_bars)
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
    async def send_telegram():
        try:
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text='Trade Activated')
        except Exception as e:
            print(f"Telegram notification failed: {e}")
    try:
        asyncio.run(send_telegram())
    except Exception as e:
        print(f"Telegram notification failed (asyncio): {e}")
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
    def __init__(self):
        self.levels = deque(maxlen=500)

    def scan_and_maybe_trade(self):
        for symbol in SYMBOLS:
            df = fetch_rates(symbol, BARS_HISTORY)
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
            for sym, lvl in list(self.levels):
                if sym != symbol or not lvl.active: continue
                if highv >= lvl.price and lowv <= lvl.price:
                    lvl.filled = True
                    self.attempt_trade_on_fill(sym, lvl)
                    lvl.active = False

    def attempt_trade_on_fill(self, symbol, lvl):
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
        open_positions = mt5.positions_get(symbol=symbol)
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
        print(f'Auto-approving trade for {symbol} {side} {lots} lots at {entry}')
        send_order(symbol, side, lots, entry, sl, tp, comment="swing_fill_riskctrl")

# --- Monitor open trades and move SL to BE when price moves in favor by SL distance ---

# ========== RUN ==========

def main():
    # Write 'running' to bot_status.txt on start
    status_path = os.path.join(os.path.dirname(__file__), 'bot_status.txt')
    try:
        with open(status_path, 'w') as f:
            f.write('running\n')
    except Exception:
        pass
    init_mt5(); eng = Engine()

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
