import numpy as np
import pandas as pd
from .config import DEFAULT_MODEL_WEIGHTS
from .features import build_features
from .models import trend, mean_reversion, momentum, risk, ml_ensemble, fundamental

def signal(score):
    if score >= 80: return "强势买入候选"
    if score >= 68: return "分批买入"
    if score >= 52: return "持有/观察"
    if score >= 38: return "减仓"
    return "规避/卖出"

def analyze(df, fundamentals=None, horizon=5):
    f = build_features(df, horizon)
    row = f.iloc[-1]
    ml = ml_ensemble(f)
    fs, completeness = fundamental(fundamentals)
    scores = {
        "trend": trend(row),
        "mean_reversion": mean_reversion(row),
        "momentum": momentum(row),
        "machine_learning": ml.score,
        "risk": risk(row),
        "fundamental": fs,
        "market_regime": 50.0
    }
    avail = {
        "trend":1, "mean_reversion":1, "momentum":1,
        "machine_learning":1 if ml.available else 0,
        "risk":1, "fundamental":completeness, "market_regime":.35
    }
    ew = {k: DEFAULT_MODEL_WEIGHTS[k] * avail[k] for k in scores}
    total = sum(ew.values())
    ew = {k:v/total for k,v in ew.items()}
    composite = sum(scores[k] * ew[k] for k in scores)
    agreement = max(0, 1 - np.std(list(scores.values())) / 35)
    dq = min(1, len(df) / 500)
    mq = .5 if not ml.available or np.isnan(ml.auc) else np.clip((ml.auc-.45)/.25,0,1)
    confidence = 100 * (.45*agreement + .30*dq + .25*mq)
    vol = row["vol20"] if pd.notna(row["vol20"]) else .35
    risk_mult = np.clip(.30/max(vol,.12), .45, 1.15)
    pos = np.clip(max(0,(composite-42)/58) * risk_mult * confidence/100, 0, .85)
    close = float(row["close"])
    a = float(row["atr14"]) if pd.notna(row["atr14"]) else close*.04
    return {
        "score":round(float(composite),2), "signal":signal(composite),
        "confidence":round(float(confidence),1),
        "up_probability":round(float(ml.probability*100),1),
        "position":round(float(pos*100),1), "close":round(close,3),
        "stop":round(close-2.2*a,3), "target":round(close+3.2*a,3),
        "scores":scores, "weights":ew,
        "ml_auc":None if np.isnan(ml.auc) else round(ml.auc,3),
        "ml_accuracy":None if np.isnan(ml.accuracy) else round(ml.accuracy,3)
    }
