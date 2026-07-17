import numpy as np
import pandas as pd

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_features(df, horizon=5):
    x = df.copy()
    c, v = x["close"], x["volume"]
    x["ret_1"] = c.pct_change()
    for n in [5, 10, 20, 60, 120]:
        x[f"ret_{n}"] = c.pct_change(n)
        x[f"ma_{n}"] = c.rolling(n).mean()
        x[f"price_ma_{n}"] = c / x[f"ma_{n}"] - 1
    x["ema12"] = c.ewm(span=12, adjust=False).mean()
    x["ema26"] = c.ewm(span=26, adjust=False).mean()
    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    x["rsi14"] = rsi(c)
    x["atr14"] = atr(x)
    x["atr_pct"] = x["atr14"] / c
    x["vol20"] = x["ret_1"].rolling(20).std() * np.sqrt(252)
    x["vol60"] = x["ret_1"].rolling(60).std() * np.sqrt(252)
    x["volume_ratio"] = v.rolling(5).mean() / v.rolling(20).mean()
    x["volume_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    x["breakout20"] = c / x["high"].rolling(20).max().shift(1) - 1
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    x["bb_position"] = (c - lower) / (upper - lower)
    x["drawdown60"] = c / c.rolling(60).max() - 1
    x["future_return"] = c.shift(-horizon) / c - 1
    x["target"] = (x["future_return"] > 0).astype(float)
    return x
