# v3 headless scanner: A/B/C logic, entries-only CSV log
import os, math, csv
from datetime import datetime
import pandas as pd, numpy as np, yfinance as yf
from pytz import timezone

# ----- CONFIG -----
TICKERS = ["SPY","QQQ","AAPL","NVDA","AMZN","C","JPM","BTC-USD","ETH-USD","PLTR","SOFI","AMC","XLE","GLD"]
EQUITY = float(os.getenv("EQUITY", "10000"))     # override via GH Secrets if you want
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1%
ATR_MULT = float(os.getenv("ATR_MULT", "1.5"))
VOL_MULT = float(os.getenv("VOL_MULT", "3.0"))   # change to 2.5 if too few signals
CONSENSUS_MIN = int(os.getenv("CONSENSUS_MIN", "2"))
CSV_LOG = "trades_log_v3_min.csv"
ET = timezone("America/Toronto")

def now_et(): return datetime.now(ET)

def market_hours_now():
    n = now_et()
    o = n.replace(hour=9, minute=30, second=0, microsecond=0)
    c = n.replace(hour=16, minute=0, second=0, microsecond=0)
    # also skip final 30 min to avoid end-of-day noise
    cut = n.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= n < cut

def fetch_5m(ticker, period="7d"):
    df = yf.download(ticker, period=period, interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    return df.dropna()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vwap(df):
    pv = (df["Close"]*df["Volume"]).cumsum()
    vv = df["Volume"].cumsum().replace(0, np.nan)
    return pv / vv

# ----- A) Regime -----
def regime_ok(df):
    if df is None or df.empty or len(df) < 200:
        return False
    close = df["Close"]
    ema200 = close.ewm(span=200, adjust=False).mean()
    last = close.iloc[-1]
    ema200_ok = last > ema200.iloc[-1]
    a = atr(df, 14).iloc[-1]
    vol_pct = float(a / max(1e-9, last))
    atr_band_ok = (0.005 <= vol_pct <= 0.08)  # 0.5%–8%
    return bool(ema200_ok and atr_band_ok)

# ----- B) Signals -----
def sig_flow(df):
    if df is None or len(df) < 21: return False
    vol = df["Volume"]; vol_ma20 = vol.rolling(20).mean()
    return bool((vol.iloc[-1] >= VOL_MULT*vol_ma20.iloc[-1]) and (df["Close"].iloc[-1] > vwap(df).iloc[-1]))

def sig_boyh(ticker, df):
    d = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
    if d is None or d.empty or len(d) < 2: return False
    yh = float(d["High"].iloc[-2])
    return bool(df["Close"].iloc[-1] > yh)

# ----- C) Risk -----
def stop_distance(df):
    return float(atr(df, 14).iloc[-1] * ATR_MULT)

def shares_for(entry, stop_dist):
    if stop_dist <= 0 or math.isnan(stop_dist): return 0
    return int((EQUITY * RISK_PER_TRADE) // max(0.01, stop_dist))

def log_row(row: dict):
    new = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)

def decide_trade(ticker):
    dfi = fetch_5m(ticker)
    if dfi is None or dfi.empty: return None
    if not regime_ok(dfi): return None
    f = sig_flow(dfi); b = sig_boyh(ticker, dfi)
    if int(f) + int(b) < CONSENSUS_MIN: return None
    entry = float(dfi["Close"].iloc[-1])
    sd = stop_distance(dfi)
    sh = shares_for(entry, sd)
    if sh <= 0 or math.isnan(sd): return None
    return {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "date": str(now_et().date()),
        "ticker": ticker,
        "signals": f"flow={int(f)},boyh={int(b)}",
        "entry": round(entry,4),
        "stop": round(entry - sd,4),
        "stop_dist": round(sd,4),
        "shares": sh,
        "risk$": round(EQUITY*RISK_PER_TRADE,2),
    }

def main():
    if not market_hours_now():
        print("Outside market scan window; skipping.")
        return
    hits = 0
    for t in TICKERS:
        try:
            row = decide_trade(t)
            if row:
                log_row(row); hits += 1
                print("TRADE:", row)
        except Exception as e:
            print("ERR", t, e)
    print(f"done; trades={hits}")

if __name__ == "__main__":
    main()
