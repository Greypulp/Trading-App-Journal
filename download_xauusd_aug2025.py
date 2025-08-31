import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

# Set your symbol and timeframe
SYMBOL = 'XAUUSD'
TIMEFRAME = mt5.TIMEFRAME_M15

# Define the start and end of August 2025 (UTC)
start = datetime(2025, 8, 1, tzinfo=timezone.utc)
end = datetime(2025, 8, 31, 23, 59, tzinfo=timezone.utc)

# Initialize MT5
if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

# Download data
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start, end)
if rates is None or len(rates) == 0:
    print("No data downloaded. Check your broker, symbol, or date range.")
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.to_csv('XAUUSD_Aug2025_M15.csv')
    print(f"Downloaded {len(df)} bars. Saved to XAUUSD_Aug2025_M15.csv.")

mt5.shutdown()
