import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="HunterTrend Mobile", page_icon="📈", layout="wide")

SYMBOLS = {
    "汇川技术": "300124",
    "特变电工": "600089",
    "创业板ETF": "159915",
}

def normalize(df):
    aliases = {
        "日期":"date","交易日期":"date","开盘":"open","最高":"high",
        "最低":"low","收盘":"close","成交量":"volume"
    }
    x = df.copy()
    x.columns = [aliases.get(str(c).strip(), str(c).strip().lower()) for c in x.columns]
    required = {"date","open","high","low","close","volume"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"缺少字段：{sorted(missing)}")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date","open","high","low","close"])
    x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(x) < 120:
        raise ValueError("至少需要120个交易日数据。")
    return x

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(symbol):
    import akshare as ak
    symbol = str(symbol).zfill(6)
    errors = []
    try:
        return normalize(ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date="20180101",
            end_date="20991231", adjust="qfq"))
    except Exception as exc:
        errors.append(f"股票接口：{exc}")
    try:
        return normalize(ak.fund_etf_hist_em(
            symbol=symbol, period="daily", start_date="20180101",
            end_date="20991231", adjust="qfq"))
    except Exception as exc:
        errors.append(f"ETF接口：{exc}")
    raise RuntimeError("在线行情获取失败。可改用上传文件。 " + " | ".join(errors))

def demo_data(rows=520):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    close = 50*np.exp(np.cumsum(rng.normal(0.0004,0.018,rows)))
    open_ = close*(1+rng.normal(0,0.004,rows))
    high = np.maximum(open_,close)*(1+rng.uniform(.001,.018,rows))
    low = np.minimum(open_,close)*(1-rng.uniform(.001,.018,rows))
    volume = rng.lognormal(15.5,.45,rows)
    return normalize(pd.DataFrame({
        "date":dates,"open":open_,"high":high,"low":low,
        "close":close,"volume":volume
    }))

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1+gain/loss.replace(0,np.nan))

def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def features(df, horizon=5):
    x = df.copy()
    c,v = x["close"],x["volume"]
    x["ret1"] = c.pct_change()
    for n in [5,10,20,60,120]:
        x[f"ret{n}"] = c.pct_change(n)
        x[f"ma{n}"] = c.rolling(n).mean()
        x[f"pma{n}"] = c/x[f"ma{n}"]-1
    x["ema12"] = c.ewm(span=12,adjust=False).mean()
    x["ema26"] = c.ewm(span=26,adjust=False).mean()
    x["macd"] = x["ema12"]-x["ema26"]
    x["signal"] = x["macd"].ewm(span=9,adjust=False).mean()
    x["macdh"] = x["macd"]-x["signal"]
    x["rsi"] = rsi(c)
    x["atr"] = atr(x)
    x["atrp"] = x["atr"]/c
    x["vol20"] = x["ret1"].rolling(20).std()*np.sqrt(252)
    x["vratio"] = v.rolling(5).mean()/v.rolling(20).mean()
    mid,sd = c.rolling(20).mean(),c.rolling(20).std()
    x["bbpos"] = (c-(mid-2*sd))/((mid+2*sd)-(mid-2*sd))
    x["dd60"] = c/c.rolling(60).max()-1
    x["future"] = c.shift(-horizon)/c-1
    x["target"] = (x["future"]>0).astype(float)
    return x

def clip(v):
    return float(np.clip(v,0,100))

def model_scores(row):
    trend = 50
    trend += 10 if row["close"]>row["ma20"] else -10
    trend += 12 if row["ma20"]>row["ma60"] else -12
    trend += 8 if row["ma60"]>row["ma120"] else -8
    trend += 10 if row["macdh"]>0 else -10
    trend += 5 if row["vratio"]>1.15 else 0

    momentum = 50
    for col,w in [("ret20",16),("ret60",18),("ret120",12)]:
        momentum += w if row[col]>0 else -w

    mean_rev = 50
    if row["rsi"]<30: mean_rev += 25
    elif row["rsi"]<40: mean_rev += 12
    elif row["rsi"]>75: mean_rev -= 25
    elif row["rsi"]>65: mean_rev -= 12
    if row["bbpos"]<.05: mean_rev += 15
    elif row["bbpos"]>.95: mean_rev -= 15

    risk = 80
    risk -= max(0,(row["vol20"]-.22)*100)
    risk += row["dd60"]*80
    risk -= max(0,(row["atrp"]-.025)*500)

    return {
        "趋势":clip(trend),
        "动量":clip(momentum),
        "均值回归":clip(mean_rev),
        "风险质量":clip(risk)
    }

ML_COLS = [
    "ret1","ret5","ret10","ret20","ret60","ret120",
    "pma5","pma20","pma60","pma120","macdh","rsi",
    "atrp","vol20","vratio","bbpos","dd60"
]

def ml_probability(f):
    d = f.dropna(subset=["target"]).copy()
    if len(d)<220:
        return 50.0, None
    split = int(len(d)*.8)
    tr,va = d.iloc[:split],d.iloc[split:]
    model = Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("rf",RandomForestClassifier(
            n_estimators=90,max_depth=5,min_samples_leaf=8,
            class_weight="balanced",random_state=42,n_jobs=-1))
    ])
    model.fit(tr[ML_COLS],tr["target"].astype(int))
    pv = model.predict_proba(va[ML_COLS])[:,1]
    latest = float(model.predict_proba(f[ML_COLS].iloc[[-1]])[:,1][0])
    try:
        auc = float(roc_auc_score(va["target"].astype(int),pv))
    except Exception:
        auc = None
    return latest*100, auc

def analyze(df,horizon):
    f = features(df,horizon)
    row = f.iloc[-1]
    scores = model_scores(row)
    ml,auc = ml_probability(f)
    scores["机器学习"] = ml
    weights = {"趋势":.27,"动量":.22,"均值回归":.14,"风险质量":.17,"机器学习":.20}
    total = sum(scores[k]*weights[k] for k in scores)
    agreement = max(0,1-np.std(list(scores.values()))/35)
    confidence = 100*(.65*agreement+.35*min(1,len(df)/500))
    vol = row["vol20"] if pd.notna(row["vol20"]) else .35
    position = np.clip(max(0,(total-42)/58)*np.clip(.30/max(vol,.12),.45,1.1)*confidence/100,0,.8)
    close = float(row["close"])
    a = float(row["atr"]) if pd.notna(row["atr"]) else close*.04
    if total>=80: signal="强势买入候选"
    elif total>=68: signal="分批买入"
    elif total>=52: signal="持有/观察"
    elif total>=38: signal="减仓"
    else: signal="规避/卖出"
    return {
        "score":round(total,1),"signal":signal,"confidence":round(confidence,1),
        "probability":round(ml,1),"position":round(position*100,1),
        "close":round(close,3),"stop":round(close-2.2*a,3),
        "target":round(close+3.2*a,3),"scores":scores,"auc":auc
    }

st.title("📈 HunterTrend Mobile")
st.caption("A股多模型量化辅助决策｜手机与电脑均可访问｜不构成投资建议")

with st.sidebar:
    name = st.selectbox("选择标的",list(SYMBOLS)+["自定义"])
    symbol = SYMBOLS.get(name,"")
    if name=="自定义":
        symbol = st.text_input("股票/ETF代码","300124")
    horizon = st.selectbox("预测周期（交易日）",[3,5,10,20],index=1)
    source = st.radio("数据来源",["在线行情","上传文件","演示数据"])
    uploaded = st.file_uploader("上传 CSV/XLSX",type=["csv","xlsx","xls"]) if source=="上传文件" else None
    run = st.button("运行分析",type="primary",use_container_width=True)

if run:
    try:
        with st.spinner("正在运行多模型分析..."):
            if source=="在线行情":
                df = fetch_history(symbol)
            elif source=="上传文件":
                if uploaded is None:
                    st.warning("请先上传行情文件。")
                    st.stop()
                if uploaded.name.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(uploaded,encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded,encoding="gbk")
                else:
                    df = pd.read_excel(uploaded)
                df = normalize(df)
            else:
                df = demo_data()

            result = analyze(df,horizon)

        cols = st.columns(5)
        values = [
            ("综合评分",f'{result["score"]}/100'),
            ("决策信号",result["signal"]),
            ("上涨概率",f'{result["probability"]}%'),
            ("置信度",f'{result["confidence"]}%'),
            ("建议仓位",f'{result["position"]}%')
        ]
        for c,(title,value) in zip(cols,values):
            c.metric(title,value)

        p1,p2,p3 = st.columns(3)
        p1.metric("最新收盘",result["close"])
        p2.metric("ATR参考止损",result["stop"])
        p3.metric("ATR参考目标",result["target"])

        st.subheader("模型投票")
        vote = pd.DataFrame(
            [{"模型":k,"评分":round(v,1)} for k,v in result["scores"].items()]
        )
        st.dataframe(vote,use_container_width=True,hide_index=True)

        fig = go.Figure(go.Candlestick(
            x=df["date"].tail(180),open=df["open"].tail(180),
            high=df["high"].tail(180),low=df["low"].tail(180),
            close=df["close"].tail(180),name="K线"
        ))
        fig.update_layout(height=460,xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)

        if result["auc"] is not None:
            st.info(f'机器学习验证 AUC：{result["auc"]:.3f}。AUC仅用于评估历史区分能力，不代表未来收益。')
        else:
            st.info("历史样本不足，机器学习模型使用中性评分。")

        st.download_button(
            "下载本次评分",
            vote.to_csv(index=False).encode("utf-8-sig"),
            "huntertrend_result.csv",
            "text/csv",
            use_container_width=True
        )
    except Exception as exc:
        st.exception(exc)
else:
    st.markdown("""
### 使用方法
1. 左侧选择汇川技术、特变电工或创业板ETF。
2. 选择在线行情，点击“运行分析”。
3. 查看综合评分、上涨概率、建议仓位及止损参考。

### 手机部署优势
此版本只有两个必要文件：`app.py` 和 `requirements.txt`，无需创建文件夹。
""")
