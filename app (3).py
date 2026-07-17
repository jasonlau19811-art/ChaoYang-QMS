import json
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="HunterTrend V3.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_WATCHLIST = {
    "汇川技术": "300124",
    "特变电工": "600089",
    "创业板ETF": "159915",
}

ALIASES = {
    "日期": "date", "交易日期": "date",
    "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume", "成交额": "amount",
}

ML_FEATURES = [
    "ret1", "ret5", "ret10", "ret20", "ret60", "ret120",
    "pma5", "pma20", "pma60", "pma120",
    "macdh", "rsi", "atrp", "vol20", "vratio", "bbpos", "dd60",
]

st.markdown("""
<style>
.block-container {padding-top: .8rem; padding-bottom: 3rem; max-width: 1180px;}
h1 {font-size: 1.65rem !important; margin-bottom: .2rem !important;}
h2 {font-size: 1.3rem !important;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 14px;
    padding: 10px 12px;
    background: rgba(128,128,128,.04);
}
div.stButton > button, div.stDownloadButton > button {
    min-height: 44px; border-radius: 12px; font-weight: 650;
}
div[data-testid="stTabs"] button {font-size: .95rem;}
@media (max-width: 640px) {
  .block-container {padding-left: .75rem; padding-right: .75rem;}
  h1 {font-size: 1.42rem !important;}
  div[data-testid="column"] {min-width: 48% !important;}
}
</style>
""", unsafe_allow_html=True)


def init_state():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = pd.DataFrame()
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = {}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("行情数据为空。")
    x = df.copy()
    x.columns = [ALIASES.get(str(c).strip(), str(c).strip().lower()) for c in x.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError("缺少字段：" + "、".join(sorted(missing)))
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = (x.dropna(subset=["date", "open", "high", "low", "close"])
           .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True))
    if len(x) < 120:
        raise ValueError("至少需要120个交易日，建议300日以上。")
    return x


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(symbol: str) -> pd.DataFrame:
    import akshare as ak
    symbol = str(symbol).strip().zfill(6)
    errors = []

    methods = [
        ("A股", lambda: ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date="20180101",
            end_date="20991231", adjust="qfq")),
        ("ETF", lambda: ak.fund_etf_hist_em(
            symbol=symbol, period="daily", start_date="20180101",
            end_date="20991231", adjust="qfq")),
    ]
    for label, fn in methods:
        for attempt in range(3):
            try:
                return normalize(fn())
            except Exception as exc:
                errors.append(f"{label}{attempt+1}: {exc}")
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError("在线数据源暂时断开。请稍后重试，或改用“上传行情文件”。")


def load_uploaded(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="gbk")
    else:
        df = pd.read_excel(uploaded)
    return normalize(df)


def demo_data(rows=600, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    close = 50 * np.exp(np.cumsum(rng.normal(.00035, .018, rows)))
    op = close * (1 + rng.normal(0, .004, rows))
    high = np.maximum(op, close) * (1 + rng.uniform(.001, .018, rows))
    low = np.minimum(op, close) * (1 - rng.uniform(.001, .018, rows))
    vol = rng.lognormal(15.4, .45, rows)
    return normalize(pd.DataFrame(
        {"date": dates, "open": op, "high": high, "low": low, "close": close, "volume": vol}
    ))


def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def features(df, horizon=5):
    x = df.copy()
    c, v = x["close"], x["volume"]
    x["ret1"] = c.pct_change()
    for n in [5, 10, 20, 60, 120]:
        x[f"ret{n}"] = c.pct_change(n)
        x[f"ma{n}"] = c.rolling(n).mean()
        x[f"pma{n}"] = c / x[f"ma{n}"] - 1
    x["ema12"] = c.ewm(span=12, adjust=False).mean()
    x["ema26"] = c.ewm(span=26, adjust=False).mean()
    x["macd"] = x["ema12"] - x["ema26"]
    x["signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macdh"] = x["macd"] - x["signal"]
    x["rsi"] = rsi(c)
    x["atr"] = atr(x)
    x["atrp"] = x["atr"] / c
    x["vol20"] = x["ret1"].rolling(20).std() * np.sqrt(252)
    x["vratio"] = v.rolling(5).mean() / v.rolling(20).mean()
    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    x["bbpos"] = (c - (mid - 2*std)) / (4*std)
    x["dd60"] = c / c.rolling(60).max() - 1
    x["future"] = c.shift(-horizon) / c - 1
    x["target"] = (x["future"] > 0).astype(float)
    return x


def clamp(v): return float(np.clip(v, 0, 100))


def rule_scores(row):
    trend = 50
    trend += 10 if row["close"] > row["ma20"] else -10
    trend += 12 if row["ma20"] > row["ma60"] else -12
    trend += 8 if row["ma60"] > row["ma120"] else -8
    trend += 10 if row["macdh"] > 0 else -10
    trend += 5 if row["vratio"] > 1.15 else 0

    momentum = 50
    for col, w in [("ret20", 16), ("ret60", 18), ("ret120", 12)]:
        momentum += w if row[col] > 0 else -w
    momentum += 6 if row["vratio"] > 1 else -3

    reversal = 50
    rv = row["rsi"]
    reversal += 24 if rv < 30 else 12 if rv < 40 else -24 if rv > 75 else -12 if rv > 65 else 0
    reversal += 14 if row["bbpos"] < .05 else -14 if row["bbpos"] > .95 else 0

    risk = 80
    risk -= max(0, (row["vol20"] - .22) * 100)
    risk += row["dd60"] * 80
    risk -= max(0, (row["atrp"] - .025) * 500)
    return {"趋势": clamp(trend), "动量": clamp(momentum),
            "均值回归": clamp(reversal), "风险质量": clamp(risk)}


def ml_prob(f):
    data = f.dropna(subset=ML_FEATURES + ["target"]).copy()
    if len(data) < 220 or data["target"].nunique() < 2:
        return 50.0, None
    split = int(len(data) * .8)
    train, valid = data.iloc[:split], data.iloc[split:]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=8,
            class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    model.fit(train[ML_FEATURES], train["target"].astype(int))
    latest = float(model.predict_proba(f[ML_FEATURES].iloc[[-1]])[:, 1][0]) * 100
    auc = None
    try:
        auc = float(roc_auc_score(valid["target"].astype(int),
                                  model.predict_proba(valid[ML_FEATURES])[:, 1]))
    except Exception:
        pass
    return latest, auc


def analyze(df, horizon=5):
    f = features(df, horizon)
    row = f.iloc[-1]
    scores = rule_scores(row)
    ml, auc = ml_prob(f)
    scores["机器学习"] = ml
    weights = {"趋势": .29, "动量": .22, "均值回归": .12, "风险质量": .19, "机器学习": .18}
    total = sum(scores[k] * weights[k] for k in scores)
    agreement = max(0, 1 - np.std(list(scores.values())) / 35)
    confidence = 100 * (.65 * agreement + .35 * min(1, len(df)/500))
    vol = float(row["vol20"]) if pd.notna(row["vol20"]) else .35
    risk_mult = np.clip(.30 / max(vol, .12), .45, 1.1)
    position = np.clip(max(0, (total-42)/58) * risk_mult * confidence/100, 0, .80)
    close = float(row["close"])
    av = float(row["atr"]) if pd.notna(row["atr"]) else close*.04
    signal = ("强势候选" if total >= 80 else "分批关注" if total >= 68 else
              "持有观察" if total >= 52 else "谨慎减仓" if total >= 38 else "风险规避")
    return {
        "score": round(total, 1), "signal": signal, "confidence": round(confidence, 1),
        "probability": round(ml, 1), "position": round(position*100, 1),
        "close": round(close, 3), "stop": round(close-2.2*av, 3),
        "target_price": round(close+3.2*av, 3), "rsi": round(float(row["rsi"]), 1),
        "volatility": round(vol*100, 1), "scores": scores, "auc": auc, "f": f,
    }


def backtest(df, horizon=5):
    f = features(df, horizon).dropna().copy()
    f["trend_ok"] = (f["close"] > f["ma20"]) & (f["ma20"] > f["ma60"]) & (f["macdh"] > 0)
    f["strategy_ret"] = np.where(f["trend_ok"].shift(1).fillna(False), f["ret1"], 0)
    f["equity"] = (1 + f["strategy_ret"].fillna(0)).cumprod()
    f["benchmark"] = (1 + f["ret1"].fillna(0)).cumprod()
    total = f["equity"].iloc[-1] - 1
    bench = f["benchmark"].iloc[-1] - 1
    dd = f["equity"] / f["equity"].cummax() - 1
    ann = f["strategy_ret"].mean() * 252
    vol = f["strategy_ret"].std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    active = f.loc[f["trend_ok"].shift(1).fillna(False), "strategy_ret"]
    win = (active > 0).mean() if len(active) else 0
    return f, total, bench, dd.min(), sharpe, win


def market_view(results):
    if results.empty:
        return 50.0, 30.0, "等待数据"
    score = float(results["Hunter分"].mean())
    pos = float(results["建议仓位"].mean())
    mood = "偏积极" if score >= 68 else "中性震荡" if score >= 52 else "谨慎防守"
    return score, min(pos, 80), mood


def get_data(symbol, mode, uploaded=None):
    if mode == "上传行情文件":
        if uploaded is None:
            raise ValueError("请先选择 CSV 或 Excel 文件。")
        return load_uploaded(uploaded), "上传文件"
    if mode == "演示数据":
        return demo_data(seed=int(symbol[-3:]) if symbol.isdigit() else 42), "演示数据"
    return fetch_history(symbol), "在线行情"


def score_color(v):
    return "🟢" if v >= 68 else "🟡" if v >= 52 else "🔴"


init_state()

st.title("🎯 HunterTrend V3.0")
st.caption("AI量化驾驶舱 · 手机稳定版 · 仅供研究，不构成投资建议")

tab_home, tab_stock, tab_watch, tab_backtest, tab_settings = st.tabs(
    ["驾驶舱", "单股诊断", "自选股", "策略回测", "设置"]
)

with tab_home:
    st.subheader("今日投资驾驶舱")
    if st.button("🚀 一键扫描自选股", type="primary", use_container_width=True):
        rows, failures = [], []
        bar = st.progress(0)
        items = list(st.session_state.watchlist.items())
        for i, (name, symbol) in enumerate(items):
            try:
                df = fetch_history(symbol)
                r = analyze(df)
                rows.append({
                    "名称": name, "代码": symbol, "Hunter分": r["score"],
                    "信号": r["signal"], "上涨概率": r["probability"],
                    "建议仓位": r["position"], "现价": r["close"]
                })
            except Exception as exc:
                failures.append(f"{name}：{exc}")
            bar.progress((i+1)/max(1, len(items)))
        st.session_state.batch_results = pd.DataFrame(rows)
        if failures:
            st.warning("部分标的未完成：" + "；".join(failures))
        elif rows:
            st.success("扫描完成。")

    mscore, mpos, mood = market_view(st.session_state.batch_results)
    c1, c2, c3 = st.columns(3)
    c1.metric("市场温度", f"{mscore:.0f} 分")
    c2.metric("建议总仓位", f"{mpos:.0f}%")
    c3.metric("当前状态", mood)

    if st.session_state.batch_results.empty:
        st.info("点击“一键扫描自选股”生成今日驾驶舱。在线接口若暂时断线，可到“单股诊断”上传行情文件。")
        demo = pd.DataFrame([
            {"名称":"汇川技术","代码":"300124","Hunter分":"--","信号":"等待扫描"},
            {"名称":"特变电工","代码":"600089","Hunter分":"--","信号":"等待扫描"},
            {"名称":"创业板ETF","代码":"159915","Hunter分":"--","信号":"等待扫描"},
        ])
        st.dataframe(demo, hide_index=True, use_container_width=True)
    else:
        show = st.session_state.batch_results.sort_values("Hunter分", ascending=False).copy()
        show["状态"] = show["Hunter分"].apply(score_color)
        st.dataframe(show[["状态","名称","代码","Hunter分","信号","上涨概率","建议仓位","现价"]],
                     hide_index=True, use_container_width=True)
        top = show.iloc[0]
        st.success(f"今日重点：{top['名称']}（{top['代码']}），Hunter分 {top['Hunter分']}，信号：{top['信号']}。")

with tab_stock:
    st.subheader("单股 AI 诊断")
    c1, c2 = st.columns(2)
    name = c1.text_input("名称", value="汇川技术")
    symbol = c2.text_input("代码", value="300124", max_chars=6)
    mode = st.radio("数据方式", ["在线行情", "上传行情文件", "演示数据"], horizontal=True)
    uploaded = None
    if mode == "上传行情文件":
        uploaded = st.file_uploader("上传 CSV / Excel", type=["csv","xlsx","xls"])
        st.caption("字段需要包含：日期、开盘、最高、最低、收盘、成交量。")
    horizon = st.select_slider("预测周期（交易日）", options=[3,5,10,20], value=5)

    if st.button("开始诊断", type="primary", use_container_width=True):
        try:
            with st.spinner("正在计算 Hunter Score…"):
                df, source = get_data(symbol.strip().zfill(6), mode, uploaded)
                r = analyze(df, horizon)
            st.session_state["last_df"] = df
            st.session_state["last_result"] = r
            st.session_state["last_label"] = f"{name}（{symbol.strip().zfill(6)}）"
            st.session_state["last_source"] = source
        except Exception as exc:
            st.error(str(exc))

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        df = st.session_state["last_df"]
        st.caption(f"数据来源：{st.session_state['last_source']}｜最后日期：{df['date'].iloc[-1].date()}")
        a,b,c,d = st.columns(4)
        a.metric("Hunter分", f"{r['score']}")
        b.metric("信号", r["signal"])
        c.metric("上涨概率", f"{r['probability']}%")
        d.metric("建议仓位", f"{r['position']}%")

        e,f,g,h = st.columns(4)
        e.metric("现价", r["close"])
        f.metric("参考止损", r["stop"])
        g.metric("参考目标", r["target_price"])
        h.metric("波动率", f"{r['volatility']}%")

        fig = go.Figure()
        tail = r["f"].tail(180)
        fig.add_trace(go.Candlestick(
            x=tail["date"], open=tail["open"], high=tail["high"],
            low=tail["low"], close=tail["close"], name="K线"))
        fig.add_trace(go.Scatter(x=tail["date"], y=tail["ma20"], name="MA20"))
        fig.add_trace(go.Scatter(x=tail["date"], y=tail["ma60"], name="MA60"))
        fig.update_layout(height=430, margin=dict(l=5,r=5,t=30,b=5),
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        score_df = pd.DataFrame({"模型": list(r["scores"].keys()),
                                 "得分": [round(v,1) for v in r["scores"].values()]})
        st.bar_chart(score_df.set_index("模型"))

        positives, risks = [], []
        if r["scores"]["趋势"] >= 65: positives.append("趋势结构偏强")
        else: risks.append("趋势优势尚不明显")
        if r["scores"]["动量"] >= 65: positives.append("中期动量较好")
        else: risks.append("动量偏弱或分化")
        if r["rsi"] > 70: risks.append("RSI偏高，短线可能过热")
        elif r["rsi"] < 35: positives.append("RSI较低，存在修复空间")
        if r["volatility"] > 40: risks.append("近期波动较高")
        else: positives.append("波动相对可控")
        st.markdown(f"""
### Hunter AI 结论
**{st.session_state['last_label']}：{r['signal']}**

积极因素：{"；".join(positives) if positives else "暂无明显优势"}。  
主要风险：{"；".join(risks) if risks else "暂无突出技术风险"}。  
仓位参考上限约 **{r['position']}%**，ATR参考止损 **{r['stop']}**，参考目标 **{r['target_price']}**。
""")

with tab_watch:
    st.subheader("自选股管理")
    c1, c2 = st.columns(2)
    new_name = c1.text_input("新增名称", placeholder="例如：宁德时代")
    new_symbol = c2.text_input("新增代码", placeholder="例如：300750")
    if st.button("➕ 添加到自选", use_container_width=True):
        n, s = new_name.strip(), new_symbol.strip()
        if not n or not s.isdigit():
            st.warning("请填写名称和数字代码。")
        else:
            st.session_state.watchlist[n] = s.zfill(6)
            st.success(f"已添加 {n}。")

    current = pd.DataFrame(
        [{"名称":k, "代码":v} for k,v in st.session_state.watchlist.items()]
    )
    st.dataframe(current, hide_index=True, use_container_width=True)

    remove = st.selectbox("选择要删除的标的", ["不删除"] + list(st.session_state.watchlist.keys()))
    if remove != "不删除" and st.button("删除所选", use_container_width=True):
        del st.session_state.watchlist[remove]
        st.success("已删除。")
        st.rerun()

    config_bytes = json.dumps(st.session_state.watchlist, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ 备份自选股", config_bytes, "huntertrend_watchlist.json",
                       "application/json", use_container_width=True)
    cfg = st.file_uploader("恢复自选股备份", type=["json"], key="cfg")
    if cfg and st.button("恢复备份", use_container_width=True):
        try:
            data = json.load(cfg)
            st.session_state.watchlist = {str(k): str(v).zfill(6) for k,v in data.items()}
            st.success("自选股已恢复。")
            st.rerun()
        except Exception as exc:
            st.error(f"备份文件无效：{exc}")

with tab_backtest:
    st.subheader("趋势策略回测")
    st.caption("规则：收盘价高于MA20、MA20高于MA60且MACD柱为正时持有。")
    c1, c2 = st.columns(2)
    bt_name = c1.text_input("回测名称", value="创业板ETF", key="bt_name")
    bt_symbol = c2.text_input("回测代码", value="159915", key="bt_symbol")
    bt_mode = st.radio("回测数据", ["在线行情", "演示数据"], horizontal=True, key="bt_mode")
    if st.button("运行回测", type="primary", use_container_width=True):
        try:
            df, source = get_data(bt_symbol.strip().zfill(6), bt_mode)
            curve, total, bench, maxdd, sharpe, win = backtest(df)
            a,b,c = st.columns(3)
            a.metric("策略收益", f"{total*100:.1f}%")
            b.metric("买入持有", f"{bench*100:.1f}%")
            c.metric("最大回撤", f"{maxdd*100:.1f}%")
            d,e = st.columns(2)
            d.metric("夏普比率", f"{sharpe:.2f}")
            e.metric("持仓日胜率", f"{win*100:.1f}%")
            chart = curve.set_index("date")[["equity","benchmark"]].rename(
                columns={"equity":"策略净值","benchmark":"买入持有"})
            st.line_chart(chart)
            st.caption(f"数据来源：{source}。历史回测不代表未来收益。")
        except Exception as exc:
            st.error(str(exc))

with tab_settings:
    st.subheader("版本与使用说明")
    st.markdown("""
**当前版本：V3.0.0 Mobile Stable**

手机更新只需要覆盖仓库根目录中的：

1. `app.py`
2. `requirements.txt`

不要进入同名子文件夹，也不要修改 Streamlit 的 Main file path。

**在线接口断线时：**
进入“单股诊断” → 选择“上传行情文件” → 上传 CSV 或 Excel，即可继续分析。
""")
    if st.button("清除行情缓存", use_container_width=True):
        fetch_history.clear()
        st.success("缓存已清除。")
