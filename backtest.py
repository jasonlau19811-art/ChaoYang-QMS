import numpy as np
from .features import build_features
from .models import trend, momentum, mean_reversion, risk

def backtest(df, fee=.0012):
    f = build_features(df).dropna().copy()
    f["score"] = f.apply(lambda r: .36*trend(r)+.28*momentum(r)+.16*mean_reversion(r)+.20*risk(r), axis=1)
    f["position"] = np.select([f["score"]>=72,f["score"]>=60,f["score"]>=48],[1,.6,.25],default=0)
    f["position"] = f["position"].shift(1).fillna(0)
    f["turnover"] = f["position"].diff().abs().fillna(f["position"])
    f["strategy_return"] = f["position"]*f["ret_1"] - f["turnover"]*fee
    f["equity"] = (1+f["strategy_return"].fillna(0)).cumprod()
    f["benchmark"] = (1+f["ret_1"].fillna(0)).cumprod()
    dd = f["equity"]/f["equity"].cummax()-1
    years=max(len(f)/252,.1)
    ann=f["equity"].iloc[-1]**(1/years)-1
    vol=f["strategy_return"].std()*np.sqrt(252)
    return f, {"累计收益":f["equity"].iloc[-1]-1,"基准收益":f["benchmark"].iloc[-1]-1,
               "年化收益":ann,"最大回撤":dd.min(),"夏普比率":ann/vol if vol>0 else 0}
