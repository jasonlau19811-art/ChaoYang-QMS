from __future__ import annotations
import numpy as np
import pandas as pd

ALIASES = {
    "日期": "date", "交易日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"
}

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("行情数据为空")
    x = df.copy()
    x.columns = [ALIASES.get(str(c).strip(), str(c).strip().lower()) for c in x.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"缺少字段: {sorted(missing)}")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date", "open", "high", "low", "close"])
    x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(x) < 80:
        raise ValueError("至少需要80个交易日数据，建议300日以上")
    return x

def fetch_history(symbol: str) -> pd.DataFrame:
    import akshare as ak
    symbol = str(symbol).zfill(6)
    errors = []
    try:
        return normalize_ohlcv(ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date="20180101",
            end_date="20991231", adjust="qfq"))
    except Exception as exc:
        errors.append(str(exc))
    try:
        return normalize_ohlcv(ak.fund_etf_hist_em(
            symbol=symbol, period="daily", start_date="20180101",
            end_date="20991231", adjust="qfq"))
    except Exception as exc:
        errors.append(str(exc))
    raise RuntimeError("在线数据获取失败: " + " | ".join(errors))

def load_upload(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="gbk")
    else:
        df = pd.read_excel(file)
    return normalize_ohlcv(df)

def demo_data(rows=520, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, rows)))
    open_ = close * (1 + rng.normal(0, 0.004, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(.001, .018, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(.001, .018, rows))
    volume = rng.lognormal(15.5, .45, rows)
    return normalize_ohlcv(pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }))
