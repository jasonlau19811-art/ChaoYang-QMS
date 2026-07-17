from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

def clip(x): return float(np.clip(x, 0, 100))

def trend(row):
    s = 50
    s += 10 if row["close"] > row["ma_20"] else -10
    s += 12 if row["ma_20"] > row["ma_60"] else -12
    s += 8 if row["ma_60"] > row["ma_120"] else -8
    s += 10 if row["macd_hist"] > 0 else -10
    s += 7 if row["breakout20"] > 0 else 0
    s += 5 if row["volume_ratio"] > 1.15 else 0
    return clip(s)

def mean_reversion(row):
    s, r = 50, row["rsi14"]
    if r < 28: s += 28
    elif r < 38: s += 15
    elif r > 75: s -= 28
    elif r > 65: s -= 12
    if row["bb_position"] < .05: s += 18
    elif row["bb_position"] > .95: s -= 18
    return clip(s)

def momentum(row):
    s = 50
    for col, w in [("ret_20", 16), ("ret_60", 18), ("ret_120", 12)]:
        s += w if row[col] > 0 else -w
    s += 8 if row["volume_ratio"] > 1 else -4
    return clip(s)

def risk(row):
    s = 80
    s -= max(0, (row["vol20"] - .22) * 100)
    s += row["drawdown60"] * 80
    s -= max(0, (row["atr_pct"] - .025) * 500)
    return clip(s)

FEATURES = [
    "ret_1","ret_5","ret_10","ret_20","ret_60","ret_120",
    "price_ma_5","price_ma_20","price_ma_60","price_ma_120",
    "macd_hist","rsi14","atr_pct","vol20","vol60","volume_ratio",
    "volume_z","breakout20","bb_position","drawdown60"
]

@dataclass
class MLResult:
    probability: float
    score: float
    auc: float
    accuracy: float
    available: bool

def ml_ensemble(f):
    data = f.dropna(subset=["target"]).copy()
    if len(data) < 220:
        return MLResult(.5, 50, np.nan, np.nan, False)
    split = int(len(data) * .78)
    train, valid = data.iloc[:split], data.iloc[split:]
    Xtr, ytr = train[FEATURES], train["target"].astype(int)
    Xv, yv = valid[FEATURES], valid["target"].astype(int)
    latest = f[FEATURES].iloc[[-1]]
    models = [
        Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("m", RandomForestClassifier(n_estimators=80, max_depth=5,
                   min_samples_leaf=8, class_weight="balanced",
                   random_state=42, n_jobs=1))]),
        Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("m", LogisticRegression(max_iter=300, class_weight="balanced",
                   random_state=42))])
    ]
    probs, aucs, accs, ws = [], [], [], []
    for model in models:
        model.fit(Xtr, ytr)
        pv = model.predict_proba(Xv)[:,1]
        pl = float(model.predict_proba(latest)[:,1][0])
        try: a = roc_auc_score(yv, pv)
        except ValueError: a = .5
        probs.append(pl); aucs.append(a)
        accs.append(accuracy_score(yv, pv >= .5))
        ws.append(max(.15, a - .45))
    p = float(np.average(probs, weights=ws))
    return MLResult(p, clip(p * 100), float(np.mean(aucs)),
                    float(np.mean(accs)), True)

def fundamental(values):
    if not values: return 50.0, 0.0
    s, n = 50, 0
    if values.get("roe") is not None:
        s += np.clip((values["roe"] - 8) * 1.2, -12, 18); n += 1
    if values.get("profit_growth") is not None:
        s += np.clip(values["profit_growth"] * .35, -18, 20); n += 1
    if values.get("debt_ratio") is not None:
        s += np.clip((55 - values["debt_ratio"]) * .35, -15, 15); n += 1
    if values.get("pe") is not None:
        pe = values["pe"]; s += 10 if 0 < pe < 25 else (-8 if pe > 60 or pe <= 0 else 0); n += 1
    return clip(s), n / 4
