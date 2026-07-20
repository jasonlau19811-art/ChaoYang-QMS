import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="HunterTrend V7 专业版",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- UI -----------------------------
st.markdown("""
<style>
.block-container {padding-top: .8rem; padding-bottom: 2rem; max-width: 1180px;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
div[data-testid="stTabs"] button {font-size: .95rem;}
.ht-card {border:1px solid rgba(128,128,128,.22); border-radius:14px;
padding:12px 14px; margin:6px 0; background:rgba(128,128,128,.05);}
.ht-title {font-weight:800; font-size:1.55rem;}
.ht-sub {opacity:.72; font-size:.86rem;}
.buy {color:#e53935; font-weight:800;}
.sell {color:#00a86b; font-weight:800;}
.hold {color:#f39c12; font-weight:800;}
@media (max-width: 700px) {
  .block-container {padding-left:.65rem; padding-right:.65rem;}
  [data-testid="column"] {min-width: 0 !important;}
  .ht-title {font-size:1.25rem;}
}
</style>
""", unsafe_allow_html=True)

DEFAULT_NAMES = {
    "600519": "贵州茅台", "300750": "宁德时代", "002594": "比亚迪",
    "601127": "赛力斯", "300124": "汇川技术", "000001": "平安银行",
    "600036": "招商银行", "601318": "中国平安", "000858": "五粮液",
    "600900": "长江电力", "601899": "紫金矿业", "002415": "海康威视",
}

def normalize_code(code: str) -> str:
    return "".join(ch for ch in str(code).strip() if ch.isdigit()).zfill(6)[-6:]

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_stock_names():
    names = DEFAULT_NAMES.copy()
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        code_col = next((c for c in df.columns if "code" in str(c).lower() or "代码" in str(c)), None)
        name_col = next((c for c in df.columns if "name" in str(c).lower() or "简称" in str(c) or "名称" in str(c)), None)
        if code_col and name_col:
            for _, row in df[[code_col, name_col]].dropna().iterrows():
                names[normalize_code(row[code_col])] = str(row[name_col])
    except Exception:
        pass
    return names

@st.cache_data(ttl=1800, max_entries=80, show_spinner=False)
def load_history(code: str, start: str, end: str):
    code = normalize_code(code)
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            raise ValueError("未返回行情")
        rename = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low",
                  "成交量":"volume","成交额":"amount","涨跌幅":"pct"}
        df = df.rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open","close","high","low","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True), "AKShare"
    except Exception as e:
        # 离线演示数据：保证手机端首次部署也能打开，明确标注为演示
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) < 80:
            dates = pd.bdate_range(end=pd.Timestamp(end), periods=260)
        seed = int(code) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        ret = rng.normal(0.00035, 0.018, len(dates))
        close = 20 * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, .004, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(.001, .018, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(.001, .018, len(dates)))
        volume = rng.integers(2_000_000, 30_000_000, len(dates))
        df = pd.DataFrame({"date":dates,"open":open_,"close":close,"high":high,"low":low,"volume":volume})
        return df, f"演示数据（实时接口失败：{type(e).__name__}）"

def indicators(df):
    x = df.copy()
    c = x["close"]
    x["MA5"] = c.rolling(5).mean()
    x["MA10"] = c.rolling(10).mean()
    x["MA20"] = c.rolling(20).mean()
    x["MA60"] = c.rolling(60).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["DIF"] = ema12 - ema26
    x["DEA"] = x["DIF"].ewm(span=9, adjust=False).mean()
    x["MACD"] = 2 * (x["DIF"] - x["DEA"])

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - 100 / (1 + rs)

    low9 = x["low"].rolling(9).min()
    high9 = x["high"].rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    x["K"] = rsv.ewm(com=2, adjust=False).mean()
    x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    x["J"] = 3*x["K"] - 2*x["D"]

    x["VOL_MA5"] = x["volume"].rolling(5).mean()
    x["RET"] = c.pct_change()
    x["VOLATILITY"] = x["RET"].rolling(20).std() * np.sqrt(252)
    x["MDD20"] = c / c.rolling(20).max() - 1
    return x

def signal_engine(df):
    x = indicators(df)
    score = pd.Series(0.0, index=x.index)

    score += np.where(x["close"] > x["MA20"], 18, -18)
    score += np.where(x["MA5"] > x["MA10"], 12, -10)
    score += np.where(x["MA10"] > x["MA20"], 12, -10)
    score += np.where(x["DIF"] > x["DEA"], 16, -14)
    score += np.where((x["RSI"] >= 50) & (x["RSI"] <= 72), 12, 0)
    score += np.where(x["RSI"] > 80, -14, 0)
    score += np.where(x["RSI"] < 30, 6, 0)
    score += np.where(x["K"] > x["D"], 8, -6)
    score += np.where(x["volume"] > x["VOL_MA5"] * 1.15, 8, 0)
    score += np.where(x["MDD20"] < -0.12, -8, 0)

    x["score_raw"] = score
    x["score"] = (50 + score).clip(0, 100).round(0)
    x["signal"] = np.select(
        [
            x["score"] >= 78,
            x["score"] >= 62,
            x["score"] <= 25,
            x["score"] <= 40,
        ],
        ["强力买入", "买入/持有", "强力卖出", "减仓/卖出"],
        default="观望",
    )
    x["position"] = np.select(
        [x["score"] >= 62, x["score"] <= 40],
        [1, 0],
        default=np.nan,
    )
    x["position"] = pd.Series(x["position"], index=x.index).ffill().fillna(0)
    return x

def backtest(df, fee_bps=8, stop_loss=8, take_profit=20):
    x = signal_engine(df)
    pos = x["position"].copy()
    entry = None
    for i in range(1, len(x)):
        if pos.iat[i-1] == 0 and pos.iat[i] == 1:
            entry = x["close"].iat[i]
        if pos.iat[i-1] == 1 and entry:
            pnl = x["close"].iat[i] / entry - 1
            if pnl <= -stop_loss/100 or pnl >= take_profit/100:
                pos.iat[i] = 0
                entry = None
        if pos.iat[i] == 0:
            entry = None
    x["position_bt"] = pos.shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    fee = trades * fee_bps / 10000
    x["strategy_ret"] = x["RET"].fillna(0) * x["position_bt"] - fee
    x["equity"] = (1 + x["strategy_ret"]).cumprod()
    x["benchmark"] = (1 + x["RET"].fillna(0)).cumprod()

    years = max((x["date"].iat[-1] - x["date"].iat[0]).days / 365.25, 1/365.25)
    total = x["equity"].iat[-1] - 1
    annual = x["equity"].iat[-1] ** (1/years) - 1
    vol = x["strategy_ret"].std() * np.sqrt(252)
    sharpe = annual / vol if vol and not np.isnan(vol) else 0
    dd = x["equity"] / x["equity"].cummax() - 1
    max_dd = dd.min()
    completed = int((trades == 1).sum() // 2)
    active_days = x.loc[x["position_bt"] == 1, "strategy_ret"]
    win_rate = float((active_days > 0).mean()) if len(active_days) else 0
    return x, {
        "总收益": total, "年化收益": annual, "最大回撤": max_dd,
        "夏普比率": sharpe, "交易次数": completed, "持仓日胜率": win_rate,
    }

def fmt_pct(v):
    return f"{v*100:.2f}%"

names = get_stock_names()

# ----------------------------- Header -----------------------------
st.markdown('<div class="ht-title">🦅 HunterTrend V7 专业版</div>', unsafe_allow_html=True)
st.markdown('<div class="ht-sub">移动极速版 · 股票名称 · 买卖点 · 方案B回测 · 数据异常自动降级</div>', unsafe_allow_html=True)

with st.expander("⚙️ 股票与数据设置", expanded=True):
    c1, c2 = st.columns([1.15, 1])
    with c1:
        code_input = st.text_input("输入A股代码", value=st.session_state.get("code", "300750"), max_chars=8)
        code = normalize_code(code_input)
        st.session_state["code"] = code
        stock_name = names.get(code, "名称待获取")
        st.caption(f"当前股票：{code} {stock_name}")
    with c2:
        period = st.selectbox("分析周期", ["近1年", "近2年", "近3年", "自定义"], index=1)
    if period == "自定义":
        d1, d2 = st.columns(2)
        start_d = d1.date_input("开始日期", date.today() - timedelta(days=730))
        end_d = d2.date_input("结束日期", date.today())
    else:
        days = {"近1年":365, "近2年":730, "近3年":1095}[period]
        start_d, end_d = date.today() - timedelta(days=days), date.today()

    refresh = st.button("🔄 刷新行情（清除缓存）", use_container_width=True)
    if refresh:
        load_history.clear()
        st.rerun()

with st.spinner("正在读取行情并计算信号…"):
    raw, source = load_history(code, str(start_d), str(end_d))
    data = signal_engine(raw)

latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest
chg = latest["close"]/previous["close"]-1 if previous["close"] else 0
signal = latest["signal"]
signal_class = "buy" if "买" in signal else "sell" if "卖" in signal or "减" in signal else "hold"

st.markdown(
    f'<div class="ht-card"><b>{code} {stock_name}</b>　'
    f'<span class="{signal_class}">{signal}</span><br>'
    f'<span class="ht-sub">数据源：{source}　最近交易日：{latest["date"].date()}</span></div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("最新价", f'{latest["close"]:.2f}', f"{chg*100:.2f}%")
m2.metric("AI技术评分", f'{int(latest["score"])} / 100')
m3.metric("RSI(14)", f'{latest["RSI"]:.1f}' if pd.notna(latest["RSI"]) else "—")
m4.metric("20日波动率", fmt_pct(latest["VOLATILITY"]) if pd.notna(latest["VOLATILITY"]) else "—")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 买卖点", "📈 K线趋势", "🧪 方案B回测", "📋 诊断"])

with tab1:
    buy_rows = data[data["signal"].isin(["强力买入","买入/持有"])].tail(8)
    sell_rows = data[data["signal"].isin(["强力卖出","减仓/卖出"])].tail(8)
    a, b = st.columns(2)
    with a:
        st.subheader("最近买点")
        if buy_rows.empty:
            st.info("本周期暂无明确买点。")
        else:
            show = buy_rows[["date","close","score","signal"]].copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            show.columns = ["日期","价格","评分","信号"]
            st.dataframe(show.iloc[::-1], use_container_width=True, hide_index=True)
    with b:
        st.subheader("最近卖点")
        if sell_rows.empty:
            st.info("本周期暂无明确卖点。")
        else:
            show = sell_rows[["date","close","score","signal"]].copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            show.columns = ["日期","价格","评分","信号"]
            st.dataframe(show.iloc[::-1], use_container_width=True, hide_index=True)

    reasons = []
    reasons.append(("趋势", "多头" if latest["close"] > latest["MA20"] else "弱势"))
    reasons.append(("均线", "MA5 > MA10" if latest["MA5"] > latest["MA10"] else "MA5 ≤ MA10"))
    reasons.append(("MACD", "金叉/强势" if latest["DIF"] > latest["DEA"] else "死叉/弱势"))
    reasons.append(("量能", "放量" if latest["volume"] > latest["VOL_MA5"]*1.15 else "正常/缩量"))
    st.subheader("信号依据")
    st.write("　｜　".join(f"**{k}：{v}**" for k, v in reasons))
    st.warning("技术信号仅用于研究与风险辅助，不构成收益承诺或个股买卖建议。")

with tab2:
    view = data.tail(240)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view["date"], open=view["open"], high=view["high"],
        low=view["low"], close=view["close"], name="K线"
    ))
    fig.add_trace(go.Scatter(x=view["date"], y=view["MA20"], name="MA20", mode="lines"))
    fig.add_trace(go.Scatter(x=view["date"], y=view["MA60"], name="MA60", mode="lines"))
    buys = view[view["signal"]=="强力买入"]
    sells = view[view["signal"]=="强力卖出"]
    fig.add_trace(go.Scatter(x=buys["date"], y=buys["low"]*.98, mode="markers", name="强买",
                             marker=dict(symbol="triangle-up", size=11)))
    fig.add_trace(go.Scatter(x=sells["date"], y=sells["high"]*1.02, mode="markers", name="强卖",
                             marker=dict(symbol="triangle-down", size=11)))
    fig.update_layout(height=520, margin=dict(l=5,r=5,t=20,b=5), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab3:
    s1, s2, s3 = st.columns(3)
    fee_bps = s1.slider("单边费率（基点）", 0, 30, 8)
    stop_loss = s2.slider("止损%", 3, 20, 8)
    take_profit = s3.slider("止盈%", 8, 50, 20)
    bt, stats = backtest(raw, fee_bps, stop_loss, take_profit)

    q1, q2, q3 = st.columns(3)
    q1.metric("策略总收益", fmt_pct(stats["总收益"]))
    q2.metric("年化收益", fmt_pct(stats["年化收益"]))
    q3.metric("最大回撤", fmt_pct(stats["最大回撤"]))
    q4, q5, q6 = st.columns(3)
    q4.metric("夏普比率", f'{stats["夏普比率"]:.2f}')
    q5.metric("完成交易", f'{stats["交易次数"]} 次')
    q6.metric("持仓日胜率", fmt_pct(stats["持仓日胜率"]))

    eq = bt.set_index("date")[["equity","benchmark"]].rename(
        columns={"equity":"策略净值","benchmark":"买入持有"}
    )
    st.line_chart(eq, use_container_width=True)
    st.caption("方案B：综合趋势、MACD、RSI、KDJ与量能评分；评分触发进出场，并加入手续费、止损及止盈。")

with tab4:
    diag = pd.DataFrame({
        "指标":["收盘价","MA5","MA10","MA20","MA60","DIF","DEA","MACD","RSI","K","D","20日回撤"],
        "数值":[latest.get("close"),latest.get("MA5"),latest.get("MA10"),latest.get("MA20"),
              latest.get("MA60"),latest.get("DIF"),latest.get("DEA"),latest.get("MACD"),
              latest.get("RSI"),latest.get("K"),latest.get("D"),latest.get("MDD20")]
    })
    diag["数值"] = pd.to_numeric(diag["数值"], errors="coerce").round(4)
    st.dataframe(diag, use_container_width=True, hide_index=True)
    st.markdown("""
**V7速度设计**
- 行情缓存30分钟，股票名称缓存24小时。
- 单股页面只加载所需历史行情，不扫描全市场。
- 技术指标全部使用 Pandas 向量化计算。
- 实时接口异常时自动进入演示模式，避免页面卡死。
""")

st.divider()
st.caption("HunterTrend V7 Mobile Professional · 研究工具，不构成投资建议。")
