"""
Stock Signal Notifier — RSI + Moving Averages + Valuation + Email

Quick start (Windows / macOS / Linux):
1) Install dependencies:
   pip install yfinance pandas ta yagmail python-dotenv

2) Create a .env file (same folder) with:
   EMAIL_USER=your_gmail_address@gmail.com
   EMAIL_PASS=your_gmail_app_password   # create via Google Account > Security > 2‑Step Verification > App passwords
   EMAIL_TO=recipient_email@gmail.com            # can be the same as EMAIL_USER

3) Edit TICKERS below.
4) Run:
   python stock_signal_notifier.py

Notes:
- Supports HK tickers like "0700.HK" (Tencent) and US tickers like "AAPL".
- If email variables are missing, it will just print the report.
- Safe defaults; adjust thresholds in CONFIG.
- Schedule daily via Windows Task Scheduler or cron once you verify output.
"""
from __future__ import annotations
import os
import sys
import textwrap
import datetime as dt
from dataclasses import dataclass

import pandas as pd

# Third‑party libs
try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Missing dependency: yfinance. Install with `pip install yfinance`.\n" + str(e))

try:
    from ta.momentum import RSIIndicator
except Exception as e:
    raise SystemExit("Missing dependency: ta. Install with `pip install ta`.\n" + str(e))

# Optional email + .env
YAGMAIL_AVAILABLE = True
try:
    import yagmail
except Exception:
    YAGMAIL_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; environment variables can be set by the OS
    pass

# ---------------------------- CONFIG ---------------------------------
TICKERS: list[str] = [
    "AAPL",       # Apple
    "TSLA",       # Tesla
    "NVDA",    # Tencent
]

LOOKBACK_PERIOD = "6mo"   # for indicator calculations
LOW_WINDOW_DAYS = 60       # ~3 months of trading days for recent low

# Strategy thresholds (tweak to taste)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
PE_MAX_FOR_BUY = 30.0      # treat PE above this as expensive (skip buy)
NEAR_LOW_BUFFER = 1.05     # within +5% of recent low counts as "near low"
ABOVE_SMA20_SELL_BUFFER = 1.05  # >5% above SMA20 considered stretched → sell/trim

SEND_EMAIL_IF_NO_BUY = False  # if True, always email; if False, only email when any Buy/Sell appears

# ---------------------------------------------------------------------

@dataclass
class Indicators:
    last_price: float
    sma20: float | None
    sma50: float | None
    rsi: float | None
    low_3m: float | None
    trailing_pe: float | None
    price_to_book: float | None


def fetch_history(ticker: str, period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker} (check symbol or internet connection)")
    return df


def compute_indicators(df: pd.DataFrame, ticker: str) -> Indicators:
    close = df["Close"].copy()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = RSIIndicator(close).rsi()
    low_3m = close.tail(LOW_WINDOW_DAYS).min() if len(close) >= LOW_WINDOW_DAYS else close.min()

    # Valuation via yfinance — try modern attrs first, then fallbacks
    trailing_pe = None
    price_to_book = None
    try:
        t = yf.Ticker(ticker)
        # Some yfinance versions provide .get_info(), others .info
        info = None
        for getter in (getattr(t, "get_info", None), lambda: getattr(t, "info", None)):
            if getter:
                try:
                    possible = getter() if callable(getter) else getter
                    if possible:
                        info = possible
                        break
                except Exception:
                    pass
        if isinstance(info, dict):
            trailing_pe = _safe_float(info.get("trailingPE"))
            price_to_book = _safe_float(info.get("priceToBook"))
    except Exception:
        pass

    return Indicators(
        last_price=float(close.iloc[-1]),
        sma20=_safe_float(sma20.iloc[-1]),
        sma50=_safe_float(sma50.iloc[-1]),
        rsi=_safe_float(rsi.iloc[-1]),
        low_3m=_safe_float(low_3m),
        trailing_pe=trailing_pe,
        price_to_book=price_to_book,
    )


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_signal(ticker: str, ind: Indicators) -> tuple[str, str]:
    """Return (signal, rationale). Signals: BUY / SELL / HOLD."""
    rationale = []

    # Guard values
    price = ind.last_price
    rsi = ind.rsi
    sma20 = ind.sma20
    sma50 = ind.sma50
    low_3m = ind.low_3m
    pe = ind.trailing_pe

    # BUY conditions: momentum washed out + price near recent low + not too expensive
    buy_reasons = []
    if rsi is not None and rsi < RSI_OVERSOLD:
        buy_reasons.append(f"RSI {rsi:.1f} < {RSI_OVERSOLD} (oversold)")
    if low_3m is not None and price <= low_3m * NEAR_LOW_BUFFER:
        buy_reasons.append(f"Price near 3m low ({price:.2f} vs low {low_3m:.2f})")
    if pe is None or pe <= PE_MAX_FOR_BUY:
        buy_reasons.append("Valuation ok (PE not high)")
    if sma20 is not None and price < sma20:
        buy_reasons.append(f"Below SMA20 ({price:.2f} < {sma20:.2f})")

    # SELL conditions: stretched / momentum hot
    sell_reasons = []
    if rsi is not None and rsi > RSI_OVERBOUGHT:
        sell_reasons.append(f"RSI {rsi:.1f} > {RSI_OVERBOUGHT} (overbought)")
    if sma20 is not None and price > sma20 * ABOVE_SMA20_SELL_BUFFER:
        sell_reasons.append(f"> {int((ABOVE_SMA20_SELL_BUFFER-1)*100)}% above SMA20")

    # Decision rubric
    if len(buy_reasons) >= 3:  # strong confluence
        rationale.extend(buy_reasons)
        return "BUY", "; ".join(rationale)
    if len(sell_reasons) >= 1:
        rationale.extend(sell_reasons)
        return "SELL", "; ".join(rationale)

    # Otherwise hold; include a couple of context bullets
    if rsi is not None:
        rationale.append(f"RSI {rsi:.1f}")
    if sma20 is not None and sma50 is not None:
        trend = "up" if sma20 > sma50 else "down"
        rationale.append(f"Trend: SMA20 {sma20:.2f} vs SMA50 {sma50:.2f} ({trend})")
    return "HOLD", "; ".join(rationale)


def format_row(t: str, ind: Indicators, signal: str, why: str) -> str:
    bits = [
        f"{t:10s}",
        f"Px {ind.last_price:.2f}",
        f"RSI {ind.rsi:.1f}" if ind.rsi is not None else "RSI n/a",
        f"SMA20 {ind.sma20:.2f}" if ind.sma20 is not None else "SMA20 n/a",
        f"SMA50 {ind.sma50:.2f}" if ind.sma50 is not None else "SMA50 n/a",
        f"3mLow {ind.low_3m:.2f}" if ind.low_3m is not None else "3mLow n/a",
        f"PE {ind.trailing_pe:.1f}" if ind.trailing_pe is not None else "PE n/a",
        f"PB {ind.price_to_book:.2f}" if ind.price_to_book is not None else "PB n/a",
        f"=> {signal}",
    ]
    line = " | ".join(bits)
    return line + (f"\n  ↳ {why}" if why else "")


def build_report(rows: list[str]) -> str:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    header = f"Stock Signals — {today}\n" + "-" * 80
    return header + "\n" + "\n".join(rows)


def send_email(subject: str, body: str) -> bool:
    user = os.getenv("EMAIL_USER")
    app_pw = os.getenv("EMAIL_PASS")
    to_addr = os.getenv("EMAIL_TO", user or "")
    if not (YAGMAIL_AVAILABLE and user and app_pw and to_addr):
        return False
    try:
        yag = yagmail.SMTP(user, app_pw)
        yag.send(to=to_addr, subject=subject, contents=body)
        return True
    except Exception as e:
        print(f"[email error] {e}")
        return False


def main():
    rows: list[str] = []
    summary_flags = {"BUY": 0, "SELL": 0}

    for t in TICKERS:
        try:
            df = fetch_history(t)
            ind = compute_indicators(df, t)
            signal, why = build_signal(t, ind)
            row = format_row(t, ind, signal, why)
            rows.append(row)
            if signal in summary_flags:
                summary_flags[signal] += 1
        except Exception as e:
            rows.append(f"{t:10s} | ERROR: {e}")

    report = build_report(rows)
    print(report)

    should_email = SEND_EMAIL_IF_NO_BUY or any(v > 0 for v in summary_flags.values())
    if should_email:
        subject = "Daily Stock Signals"
        emailed = send_email(subject, report)
        if emailed:
            print("[info] Email sent.")
        else:
            print("[info] Email not sent (missing creds or yagmail).")
    else:
        print("[info] No BUY/SELL signals today; email suppressed.")


if __name__ == "__main__":
    main()
