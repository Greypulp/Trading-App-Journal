# MT5 Liquidity Bot - Visual Strategy Flow (Markdown)

```mermaid
graph TD
    A[Start/Loop: Fetch latest price data] --> B{Detect new pivot high/low?}
    B -- No --> A
    B -- Yes --> C[Add pivot to levels list]
    C --> D{Did price fill a level?}
    D -- No --> A
    D -- Yes --> E[Prepare trade (side, lots, entry, SL, TP)]
    E --> F{Risk/Exposure checks}
    F -- Blocked --> A
    F -- Pass --> G[Send trade approval to Telegram]
    G --> H{User approves in Telegram?}
    H -- No/Timeout --> A
    H -- Yes --> I[Send order to MT5]
    I --> A
```

**Legend:**
- All trades require Telegram approval before execution.
- Risk and exposure checks block trades if limits are exceeded.
- The loop runs continuously, scanning for new pivot fills.
