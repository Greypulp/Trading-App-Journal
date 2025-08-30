# Utility to check if the market is open for a given symbol using MT5
import MetaTrader5 as mt5
from datetime import datetime, timezone

def is_market_open(symbol):
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        return False
    # Check trading hours (UTC):
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    # Market closed from Friday 21:00 UTC to Sunday 21:00 UTC
    if weekday == 5 or weekday == 6:
        return False
    if weekday == 4 and now.hour >= 21:
        return False
    if weekday == 6 and now.hour < 21:
        return False
    return True
