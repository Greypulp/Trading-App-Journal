import pandas as pd
import numpy as np
from datetime import datetime

class PivotATRBacktester:
    def __init__(self, df, swing_left=15, swing_right=10, atr_period=14, atr_mult=1.5, risk_pct=0.5, initial_balance=10000, contract_size=100, pip_size=0.01, min_lot=0.01):
        self.df = df.copy()
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.risk_pct = risk_pct
        self.initial_balance = initial_balance
        self.contract_size = contract_size
        self.pip_size = pip_size
        self.min_lot = min_lot
        self.trades = []
        self.balance = initial_balance

    def find_pivots(self):
        h = self.df['high'].values
        l = self.df['low'].values
        n = len(self.df)
        piv_hi = np.zeros(n, dtype=bool)
        piv_lo = np.zeros(n, dtype=bool)
        for i in range(self.swing_left, n - self.swing_right):
            window_h = h[i - self.swing_left:i + self.swing_right + 1]
            window_l = l[i - self.swing_left:i + self.swing_right + 1]
            if np.nanmax(window_h) == h[i] and (window_h == h[i]).sum() == 1:
                piv_hi[i] = True
            if np.nanmin(window_l) == l[i] and (window_l == l[i]).sum() == 1:
                piv_lo[i] = True
        self.df['pivot_high'] = piv_hi
        self.df['pivot_low'] = piv_lo

    def calculate_atr(self):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=self.atr_period, min_periods=1).mean()

    def lot_size(self, stop_pips):
        risk_amount = self.balance * (self.risk_pct / 100.0)
        pip_value_per_lot = self.contract_size * self.pip_size
        lots = risk_amount / (stop_pips * pip_value_per_lot)
        lots = max(self.min_lot, round(lots, 2))
        return lots

    def run(self):
        self.find_pivots()
        self.calculate_atr()
        open_trade = None
        for i in range(max(self.swing_left, self.atr_period), len(self.df)):
            row = self.df.iloc[i]
            if open_trade is not None:
                # Check for TP/SL hit
                if open_trade['side'] == 'buy':
                    if row['low'] <= open_trade['sl']:
                        profit = (open_trade['sl'] - open_trade['entry']) * open_trade['lots'] * self.contract_size
                        self.balance += profit
                        open_trade['exit'] = row.name
                        open_trade['exit_price'] = open_trade['sl']
                        open_trade['profit'] = profit
                        self.trades.append(open_trade)
                        open_trade = None
                        continue
                    elif row['high'] >= open_trade['tp']:
                        profit = (open_trade['tp'] - open_trade['entry']) * open_trade['lots'] * self.contract_size
                        self.balance += profit
                        open_trade['exit'] = row.name
                        open_trade['exit_price'] = open_trade['tp']
                        open_trade['profit'] = profit
                        self.trades.append(open_trade)
                        open_trade = None
                        continue
                else:
                    if row['high'] >= open_trade['sl']:
                        profit = (open_trade['entry'] - open_trade['sl']) * open_trade['lots'] * self.contract_size
                        self.balance += profit
                        open_trade['exit'] = row.name
                        open_trade['exit_price'] = open_trade['sl']
                        open_trade['profit'] = profit
                        self.trades.append(open_trade)
                        open_trade = None
                        continue
                    elif row['low'] <= open_trade['tp']:
                        profit = (open_trade['entry'] - open_trade['tp']) * open_trade['lots'] * self.contract_size
                        self.balance += profit
                        open_trade['exit'] = row.name
                        open_trade['exit_price'] = open_trade['tp']
                        open_trade['profit'] = profit
                        self.trades.append(open_trade)
                        open_trade = None
                        continue
            # Entry logic: on new pivot
            if row['pivot_low']:
                entry = row['close']
                atr = row['atr']
                stop_pips = (atr * self.atr_mult) / self.pip_size
                sl = entry - stop_pips * self.pip_size
                # TP: next pivot high after this bar
                next_pivot_idx = self.df.index[i+1:][self.df['pivot_high'].iloc[i+1:]].min() if self.df['pivot_high'].iloc[i+1:].any() else None
                if next_pivot_idx is not None:
                    tp = self.df.loc[next_pivot_idx]['high']
                else:
                    tp = entry + stop_pips * self.pip_size * 2
                lots = self.lot_size(stop_pips)
                open_trade = {'side': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'lots': lots, 'entry_time': row.name}
            elif row['pivot_high']:
                entry = row['close']
                atr = row['atr']
                stop_pips = (atr * self.atr_mult) / self.pip_size
                sl = entry + stop_pips * self.pip_size
                # TP: next pivot low after this bar
                next_pivot_idx = self.df.index[i+1:][self.df['pivot_low'].iloc[i+1:]].min() if self.df['pivot_low'].iloc[i+1:].any() else None
                if next_pivot_idx is not None:
                    tp = self.df.loc[next_pivot_idx]['low']
                else:
                    tp = entry - stop_pips * self.pip_size * 2
                lots = self.lot_size(stop_pips)
                open_trade = {'side': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'lots': lots, 'entry_time': row.name}
        # Close any open trade at the end
        if open_trade is not None:
            open_trade['exit'] = self.df.index[-1]
            open_trade['exit_price'] = self.df.iloc[-1]['close']
            open_trade['profit'] = (open_trade['exit_price'] - open_trade['entry']) * open_trade['lots'] * self.contract_size if open_trade['side']=='buy' else (open_trade['entry'] - open_trade['exit_price']) * open_trade['lots'] * self.contract_size
            self.balance += open_trade['profit']
            self.trades.append(open_trade)
        return pd.DataFrame(self.trades)
