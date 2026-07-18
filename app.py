
import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import ta

st.set_page_config(
    page_title="HunterTrend Professional",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# 基础设置与样式
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.35rem;}
.small-note {font-size: 0.82rem; color: #777;}
.signal-buy {padding: 12px; border-radius: 10px; background: rgba(0,180,90,.12);}
.signal-hold {padding: 12px; border-radius: 10px; background: rgba(255,180,0,.12);}
.signal-sell {padding: 12px; border-radius: 10px; background: rgba(230,60,60,.12);}
</style>
""", unsafe_allow_html=True)

DEFAULT_POOL = "600519,300750,002594,000001,601318,600036"

if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(
        columns=["股票代码", "持仓成本", "持仓股数"]
    )

# -----------------------------
# 数据层
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_hist_akshare(code: str, days: int = 900) -> pd.DataFrame:
    import akshare as ak

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError("AKShare未返回行情")
    rename = {
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume",
        "成交额": "Amount", "涨跌幅": "PctChange",
    }
    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "Close", "High", "Low", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Date").sort_index().dropna(subset=["Close"])

@st.cache_data(ttl=900, show_spinner=False)
def get_hist_yfinance(code: str, period: str = "3y") -> pd.DataFrame:
    import yfinance as yf

    suffix = ".SS" if code.startswith(("5", "6", "9")) else ".SZ"
    raw = yf.download(
        code + suffix,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError("备用行情源未返回数据")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

def get_history(code: str) -> tuple[pd.DataFrame, str]:
    code = str(code).strip().zfill(6)
    try:
        return get_hist_akshare(code), "AKShare"
    except Exception:
        return get_hist_yfinance(code), "yfinance备用源"

# -----------------------------
# 指标与策略引擎
# -----------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c, h, l, v = x["Close"], x["High"], x["Low"], x["Volume"]

    for n in [5, 10, 20, 60, 120]:
        x[f"MA{n}"] = c.rolling(n).mean()

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    x["MACD"] = macd.macd()
    x["MACD_SIGNAL"] = macd.macd_signal()
    x["MACD_HIST"] = macd.macd_diff()

    x["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=9, smooth_window=3)
    x["K"] = stoch.stoch()
    x["D"] = stoch.stoch_signal()

    x["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    x["ADX"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    x["OBV"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    x["BB_UP"] = bb.bollinger_hband()
    x["BB_MID"] = bb.bollinger_mavg()
    x["BB_LOW"] = bb.bollinger_lband()

    x["VOL_MA20"] = v.rolling(20).mean()
    x["RET"] = c.pct_change()
    x["VOLATILITY20"] = x["RET"].rolling(20).std() * np.sqrt(252)
    x["HIGH20"] = h.rolling(20).max()
    x["LOW20"] = l.rolling(20).min()
    x["HIGH60"] = h.rolling(60).max()
    x["LOW60"] = l.rolling(60).min()
    return x

def finite(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def support_resistance(x: pd.DataFrame) -> tuple[float, float]:
    tail = x.tail(120)
    price = finite(tail["Close"].iloc[-1])
    lows = tail["Low"].rolling(5, center=True).min()
    highs = tail["High"].rolling(5, center=True).max()

    support_candidates = tail.loc[tail["Low"] <= lows * 1.002, "Low"]
    resistance_candidates = tail.loc[tail["High"] >= highs * 0.998, "High"]

    supports = support_candidates[support_candidates < price]
    resistances = resistance_candidates[resistance_candidates > price]

    support = finite(supports.tail(10).median(), finite(tail["LOW20"].iloc[-1], price * .93))
    resistance = finite(resistances.tail(10).median(), finite(tail["HIGH20"].iloc[-1], price * 1.08))
    return support, resistance

def score_stock(x: pd.DataFrame, market_score: int = 60, style: str = "稳健") -> dict:
    if len(x) < 130:
        raise ValueError("历史数据不足130个交易日")

    r = x.iloc[-1]
    p = x.iloc[-2]
    price = finite(r["Close"])
    atr = max(finite(r["ATR"], price * .025), price * .008)
    support, resistance = support_resistance(x)

    trend = momentum = volume_score = risk = 0
    buy_reasons, sell_reasons, warnings = [], [], []

    # 趋势 35
    if r["MA5"] > r["MA10"] > r["MA20"]:
        trend += 12
        buy_reasons.append("短期均线多头排列")
    elif r["MA5"] < r["MA10"]:
        sell_reasons.append("短期均线转弱")

    if r["MA20"] > r["MA60"]:
        trend += 10
        buy_reasons.append("20日线位于60日线上方")
    else:
        sell_reasons.append("中期趋势尚未转强")

    if r["Close"] > r["MA20"]:
        trend += 6
    else:
        sell_reasons.append("收盘价跌破20日均线")

    if r["MA60"] > r["MA120"]:
        trend += 4

    if r["ADX"] >= 22:
        trend += 3
        buy_reasons.append("ADX显示趋势具备一定强度")

    # 动量 30
    if r["MACD"] > r["MACD_SIGNAL"]:
        momentum += 8
        if p["MACD"] <= p["MACD_SIGNAL"]:
            momentum += 5
            buy_reasons.append("MACD刚形成金叉")
        else:
            buy_reasons.append("MACD维持多头")
    else:
        sell_reasons.append("MACD位于信号线下方")

    if 50 <= r["RSI"] <= 70:
        momentum += 8
        buy_reasons.append("RSI处于健康强势区")
    elif 35 <= r["RSI"] < 50:
        momentum += 3
    elif r["RSI"] > 80:
        risk += 8
        warnings.append("RSI过热，追高风险较高")
    elif r["RSI"] < 30:
        warnings.append("RSI超卖，但需等待止跌确认")

    if r["K"] > r["D"]:
        momentum += 5
        if p["K"] <= p["D"]:
            momentum += 4
            buy_reasons.append("KDJ形成金叉")
    else:
        sell_reasons.append("KDJ动能走弱")

    # 成交与突破 20
    vol_ratio = finite(r["Volume"] / r["VOL_MA20"], 1)
    if 1.2 <= vol_ratio <= 2.8:
        volume_score += 8
        buy_reasons.append("成交量较20日均量放大")
    elif vol_ratio > 3.5:
        risk += 4
        warnings.append("成交量异常放大，注意冲高回落")

    if r["OBV"] > x["OBV"].rolling(20).mean().iloc[-1]:
        volume_score += 5

    if price >= finite(p["HIGH20"], price) * .995:
        volume_score += 7
        buy_reasons.append("接近或突破20日阶段高点")

    # 市场环境 15
    env = max(0, min(15, market_score * .15))

    raw = trend + momentum + volume_score + env
    risk_penalty = risk

    vol_ann = finite(r["VOLATILITY20"])
    if vol_ann > .55:
        risk_penalty += 8
        warnings.append("近20日年化波动率偏高")
    elif vol_ann > .4:
        risk_penalty += 4

    score = int(max(0, min(100, round(raw - risk_penalty))))

    style_map = {
        "保守": (88, 76, .25),
        "稳健": (82, 68, .35),
        "积极": (76, 62, .45),
    }
    buy_cut, watch_cut, max_position = style_map[style]

    if score >= buy_cut and len(sell_reasons) <= 1:
        signal, css = "🟢 A级买入关注", "signal-buy"
        position = max_position
    elif score >= watch_cut:
        signal, css = "🟡 试仓 / 持有", "signal-hold"
        position = max_position * .55
    elif score >= 52:
        signal, css = "🟠 观望等待", "signal-hold"
        position = max_position * .2
    else:
        signal, css = "🔴 减仓 / 回避", "signal-sell"
        position = 0.0

    # 动态交易区间与风控
    buy_low = max(support, price - .55 * atr)
    buy_high = price + .15 * atr
    stop = max(support - .35 * atr, price - 2.1 * atr)
    if stop >= price:
        stop = price - 1.8 * atr

    unit_risk = max(price - stop, price * .01)
    target1 = price + 2.0 * unit_risk
    target2 = price + 3.2 * unit_risk
    rr1 = (target1 - price) / unit_risk

    # 行为建议
    if signal.startswith("🟢"):
        action = "分两至三次建仓；首次不超过计划仓位的一半。"
    elif signal.startswith("🟡"):
        action = "已有持仓可观察；新仓只允许小仓试错。"
    elif signal.startswith("🟠"):
        action = "等待放量突破或回踩企稳，不追涨。"
    else:
        action = "优先控制风险；跌破止损位应执行纪律。"

    return {
        "price": price, "score": score, "signal": signal, "css": css,
        "trend": trend, "momentum": momentum, "volume_score": volume_score,
        "risk_penalty": risk_penalty, "market_env": env,
        "buy_low": buy_low, "buy_high": buy_high, "stop": stop,
        "target1": target1, "target2": target2, "rr1": rr1,
        "support": support, "resistance": resistance,
        "position": position, "atr": atr, "rsi": finite(r["RSI"]),
        "adx": finite(r["ADX"]), "vol_ratio": vol_ratio,
        "buy_reasons": buy_reasons, "sell_reasons": sell_reasons,
        "warnings": warnings, "action": action,
    }

def position_plan(result: dict, capital: float, risk_pct: float) -> dict:
    price, stop = result["price"], result["stop"]
    risk_per_share = max(price - stop, price * .01)
    risk_budget = capital * risk_pct / 100
    by_risk = math.floor(risk_budget / risk_per_share / 100) * 100
    by_capital = math.floor(capital * result["position"] / price / 100) * 100
    shares = max(0, min(by_risk, by_capital))
    return {
        "shares": shares,
        "amount": shares * price,
        "max_loss": shares * risk_per_share,
        "position_pct": shares * price / capital if capital else 0,
    }

def backtest(x: pd.DataFrame, initial=100000.0, fee_rate=.0008) -> dict:
    d = x.dropna().copy()
    cash, shares = initial, 0
    equity_curve, trades = [], []
    entry_price = None

    for i in range(1, len(d)):
        r, p = d.iloc[i], d.iloc[i-1]
        price = finite(r["Close"])
        buy = (
            r["MA5"] > r["MA20"] and p["MA5"] <= p["MA20"]
            and r["MACD"] > r["MACD_SIGNAL"]
            and 40 < r["RSI"] < 75
        )
        sell = (
            (r["MA5"] < r["MA20"] and p["MA5"] >= p["MA20"])
            or r["RSI"] > 82
        )

        if shares == 0 and buy:
            qty = math.floor(cash * .95 / price / 100) * 100
            if qty > 0:
                cost = qty * price * (1 + fee_rate)
                cash -= cost
                shares = qty
                entry_price = price
                trades.append({"日期": d.index[i], "动作": "买入", "价格": price, "股数": qty})
        elif shares > 0 and sell:
            cash += shares * price * (1 - fee_rate)
            trades.append({"日期": d.index[i], "动作": "卖出", "价格": price, "股数": shares})
            shares = 0
            entry_price = None

        equity_curve.append((d.index[i], cash + shares * price))

    if shares > 0:
        cash += shares * finite(d["Close"].iloc[-1]) * (1 - fee_rate)
        shares = 0

    curve = pd.Series(dict(equity_curve), dtype=float)
    total_return = cash / initial - 1
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    max_dd = finite(drawdown.min())
    years = max((d.index[-1] - d.index[0]).days / 365.25, 0.1)
    annual = (cash / initial) ** (1 / years) - 1
    ret = curve.pct_change().dropna()
    sharpe = finite(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0

    closed = []
    current = None
    for t in trades:
        if t["动作"] == "买入":
            current = t
        elif t["动作"] == "卖出" and current:
            pnl = (t["价格"] - current["价格"]) / current["价格"]
            closed.append(pnl)
            current = None
    win_rate = sum(p > 0 for p in closed) / len(closed) if closed else 0

    return {
        "curve": curve, "trades": pd.DataFrame(trades),
        "total_return": total_return, "annual": annual,
        "max_dd": max_dd, "sharpe": sharpe,
        "win_rate": win_rate, "closed_trades": len(closed),
    }

def csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# -----------------------------
# 侧边栏
# -----------------------------
with st.sidebar:
    st.title("🦅 HunterTrend")
    st.caption("Professional Final")
    capital = st.number_input("账户资金（元）", 1000.0, 100000000.0, 100000.0, 1000.0)
    risk_pct = st.slider("单笔最大亏损预算", .5, 5.0, 2.0, .5)
    style = st.selectbox("交易风格", ["保守", "稳健", "积极"], index=1)
    market_score = st.slider("市场环境评分", 0, 100, 60)
    st.caption("市场环境可结合指数趋势、成交额与涨跌家数人工调整。")

st.title("🦅 HunterTrend Professional Final")
st.caption("买卖点｜全自选扫描｜交易计划｜回测｜持仓管理｜晨报")

tabs = st.tabs(["单股决策", "自选股雷达", "策略回测", "持仓管理", "AI晨报", "使用说明"])

# -----------------------------
# 单股决策
# -----------------------------
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    code = c1.text_input("A股代码", "600519", key="single_code")
    analyze = c2.button("生成专业交易计划", type="primary", use_container_width=True)

    if analyze:
        try:
            with st.spinner("正在获取行情并计算专业指标…"):
                raw, source = get_history(code)
                x = enrich(raw)
                result = score_stock(x, market_score, style)
                plan = position_plan(result, capital, risk_pct)

            st.caption(f"数据源：{source}｜最后交易日：{x.index[-1].date()}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("当前价", f"{result['price']:.2f}")
            m2.metric("综合评分", f"{result['score']}/100")
            m3.metric("建议仓位", f"{plan['position_pct']:.1%}")
            m4.metric("建议股数", f"{plan['shares']}")

            st.markdown(
                f"<div class='{result['css']}'><b>{result['signal']}</b><br>{result['action']}</div>",
                unsafe_allow_html=True,
            )

            plan_df = pd.DataFrame({
                "交易项目": ["理想买入区间", "关键支撑", "阶段压力", "纪律止损",
                           "第一目标", "第二目标", "第一目标盈亏比", "预计投入", "止损最大亏损"],
                "专业结果": [
                    f"{result['buy_low']:.2f} - {result['buy_high']:.2f}",
                    f"{result['support']:.2f}", f"{result['resistance']:.2f}",
                    f"{result['stop']:.2f}", f"{result['target1']:.2f}",
                    f"{result['target2']:.2f}", f"1 : {result['rr1']:.2f}",
                    f"{plan['amount']:,.0f} 元", f"{plan['max_loss']:,.0f} 元",
                ]
            })
            st.dataframe(plan_df, hide_index=True, use_container_width=True)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("趋势分", result["trend"])
            s2.metric("动量分", result["momentum"])
            s3.metric("量价分", result["volume_score"])
            s4.metric("风险扣分", result["risk_penalty"])

            st.subheader("价格与趋势")
            chart = x[["Close", "MA5", "MA20", "MA60"]].tail(180).rename(
                columns={"Close": "收盘价"}
            )
            st.line_chart(chart)

            left, right = st.columns(2)
            with left:
                st.markdown("#### 买入依据")
                if result["buy_reasons"]:
                    for item in result["buy_reasons"]:
                        st.write("✅", item)
                else:
                    st.write("暂无足够的多信号共振。")
            with right:
                st.markdown("#### 卖出与风险")
                items = result["sell_reasons"] + result["warnings"]
                if items:
                    for item in items:
                        st.write("⚠️", item)
                else:
                    st.write("暂未发现明显趋势破坏。")

            report = pd.DataFrame([{
                "股票代码": code.zfill(6), "日期": str(x.index[-1].date()),
                "当前价": result["price"], "评分": result["score"],
                "信号": result["signal"], "买入下限": result["buy_low"],
                "买入上限": result["buy_high"], "止损": result["stop"],
                "目标一": result["target1"], "目标二": result["target2"],
                "建议股数": plan["shares"], "预计投入": plan["amount"],
            }])
            st.download_button(
                "下载本次交易计划 CSV",
                data=csv_download(report),
                file_name=f"HunterTrend_{code.zfill(6)}_trade_plan.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"分析失败：{e}")

# -----------------------------
# 自选股雷达
# -----------------------------
with tabs[1]:
    pool_text = st.text_area("自选股代码（逗号、空格或换行分隔）", DEFAULT_POOL, height=100)
    if st.button("扫描自选股", type="primary"):
        codes = [x.strip().zfill(6) for x in pool_text.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
        rows = []
        progress = st.progress(0)
        for idx, c in enumerate(codes):
            try:
                raw, source = get_history(c)
                x = enrich(raw)
                r = score_stock(x, market_score, style)
                rows.append({
                    "股票代码": c, "现价": round(r["price"], 2), "评分": r["score"],
                    "信号": r["signal"], "买入下限": round(r["buy_low"], 2),
                    "买入上限": round(r["buy_high"], 2), "止损": round(r["stop"], 2),
                    "目标一": round(r["target1"], 2), "趋势分": r["trend"],
                    "动量分": r["momentum"], "量价分": r["volume_score"],
                    "风险扣分": r["risk_penalty"], "数据源": source,
                })
            except Exception as e:
                rows.append({"股票代码": c, "信号": f"数据失败：{str(e)[:25]}"})
            progress.progress((idx + 1) / len(codes))
        radar = pd.DataFrame(rows)
        if "评分" in radar:
            radar = radar.sort_values("评分", ascending=False, na_position="last")
        st.dataframe(radar, hide_index=True, use_container_width=True)
        st.download_button(
            "下载自选股扫描结果",
            data=csv_download(radar),
            file_name="HunterTrend_watchlist_radar.csv",
            mime="text/csv",
        )

# -----------------------------
# 回测
# -----------------------------
with tabs[2]:
    bt_code = st.text_input("回测股票代码", "300750", key="bt_code")
    initial = st.number_input("回测初始资金", 10000.0, 10000000.0, 100000.0, 10000.0)
    st.caption("策略：MA5上穿MA20 + MACD多头 + RSI过滤；MA5下穿MA20或RSI过热退出。")
    if st.button("运行历史回测", type="primary"):
        try:
            raw, source = get_history(bt_code)
            x = enrich(raw)
            bt = backtest(x, initial=initial)

            a, b, c, d, e = st.columns(5)
            a.metric("累计收益", f"{bt['total_return']:.1%}")
            b.metric("年化收益", f"{bt['annual']:.1%}")
            c.metric("最大回撤", f"{bt['max_dd']:.1%}")
            d.metric("夏普比率", f"{bt['sharpe']:.2f}")
            e.metric("胜率", f"{bt['win_rate']:.1%}")

            st.line_chart(bt["curve"].rename("策略净值"))
            st.write(f"已闭合交易：{bt['closed_trades']} 次｜数据源：{source}")
            if not bt["trades"].empty:
                st.dataframe(bt["trades"], hide_index=True, use_container_width=True)
                st.download_button(
                    "下载交易明细",
                    data=csv_download(bt["trades"]),
                    file_name=f"HunterTrend_{bt_code}_backtest_trades.csv",
                    mime="text/csv",
                )
            st.warning("历史回测不代表未来收益；本回测未完整模拟涨跌停、滑点、停牌、分红税与成交冲击。")
        except Exception as e:
            st.error(f"回测失败：{e}")

# -----------------------------
# 持仓管理
# -----------------------------
with tabs[3]:
    st.subheader("持仓录入")
    upload = st.file_uploader("上传持仓CSV（列名：股票代码、持仓成本、持仓股数）", type=["csv"])
    if upload is not None:
        try:
            st.session_state.portfolio = pd.read_csv(upload, dtype={"股票代码": str})
        except Exception as e:
            st.error(f"持仓文件读取失败：{e}")

    with st.form("add_holding"):
        p1, p2, p3 = st.columns(3)
        h_code = p1.text_input("股票代码", "600519")
        h_cost = p2.number_input("持仓成本", 0.01, 100000.0, 100.0)
        h_shares = p3.number_input("持仓股数", 100, 10000000, 100, 100)
        submitted = st.form_submit_button("加入持仓")
        if submitted:
            new = pd.DataFrame([{
                "股票代码": h_code.zfill(6), "持仓成本": h_cost, "持仓股数": h_shares
            }])
            st.session_state.portfolio = pd.concat(
                [st.session_state.portfolio, new], ignore_index=True
            )

    if not st.session_state.portfolio.empty:
        st.dataframe(st.session_state.portfolio, hide_index=True, use_container_width=True)
        if st.button("评估全部持仓", type="primary"):
            rows = []
            for _, h in st.session_state.portfolio.iterrows():
                c = str(h["股票代码"]).zfill(6)
                try:
                    raw, source = get_history(c)
                    x = enrich(raw)
                    r = score_stock(x, market_score, style)
                    cost = finite(h["持仓成本"])
                    shares = int(h["持仓股数"])
                    pnl = (r["price"] - cost) * shares
                    pnl_pct = r["price"] / cost - 1 if cost else 0
                    if r["price"] < r["stop"] or r["score"] < 50:
                        holding_action = "减仓/执行止损"
                    elif pnl_pct > .20 and r["score"] < 70:
                        holding_action = "分批止盈"
                    elif r["score"] >= 75:
                        holding_action = "继续持有"
                    else:
                        holding_action = "观察"
                    rows.append({
                        "股票代码": c, "成本": cost, "现价": round(r["price"], 2),
                        "持仓股数": shares, "浮动盈亏": round(pnl, 2),
                        "收益率": f"{pnl_pct:.1%}", "AI评分": r["score"],
                        "建议": holding_action, "防守位": round(r["stop"], 2),
                    })
                except Exception as e:
                    rows.append({"股票代码": c, "建议": f"数据失败：{str(e)[:20]}"})
            evaluation = pd.DataFrame(rows)
            st.dataframe(evaluation, hide_index=True, use_container_width=True)
            st.download_button(
                "下载持仓诊断",
                data=csv_download(evaluation),
                file_name="HunterTrend_portfolio_review.csv",
                mime="text/csv",
            )
    else:
        st.info("尚未录入持仓。")

# -----------------------------
# AI晨报
# -----------------------------
with tabs[4]:
    morning_pool = st.text_area("晨报股票池", DEFAULT_POOL, key="morning_pool")
    if st.button("生成今日AI晨报", type="primary"):
        codes = [x.strip().zfill(6) for x in morning_pool.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]
        rows = []
        for c in codes:
            try:
                raw, source = get_history(c)
                x = enrich(raw)
                r = score_stock(x, market_score, style)
                rows.append({
                    "股票代码": c, "评分": r["score"], "信号": r["signal"],
                    "现价": round(r["price"], 2),
                    "关注区间": f"{r['buy_low']:.2f}-{r['buy_high']:.2f}",
                    "止损": round(r["stop"], 2), "第一目标": round(r["target1"], 2),
                    "核心逻辑": "；".join(r["buy_reasons"][:3]) or "等待信号",
                    "主要风险": "；".join((r["sell_reasons"] + r["warnings"])[:2]) or "暂无明显风险",
                })
            except Exception as e:
                rows.append({"股票代码": c, "信号": f"数据失败：{str(e)[:20]}"})

        brief = pd.DataFrame(rows)
        if "评分" in brief:
            brief = brief.sort_values("评分", ascending=False, na_position="last")
        st.subheader(f"HunterTrend AI晨报｜{datetime.now().strftime('%Y-%m-%d')}")
        st.write(f"市场环境评分：**{market_score}/100**｜交易风格：**{style}**")
        st.dataframe(brief, hide_index=True, use_container_width=True)
        if not brief.empty and "评分" in brief and brief["评分"].notna().any():
            top = brief.dropna(subset=["评分"]).iloc[0]
            st.success(f"今日股票池首选：{top['股票代码']}｜评分 {int(top['评分'])}")
        st.download_button(
            "下载AI晨报",
            data=csv_download(brief),
            file_name=f"HunterTrend_morning_brief_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# -----------------------------
# 说明
# -----------------------------
with tabs[5]:
    st.markdown("""
### 专业版包含
- 多指标买卖点：均线、MACD、KDJ、RSI、ADX、ATR、OBV、布林带与量价突破
- 支撑压力、动态止损、两级止盈、盈亏比
- 按账户资金与单笔最大亏损预算计算股数
- 多股票自选雷达、CSV导出
- 历史策略回测、收益/回撤/夏普/胜率
- 持仓盈亏与减仓、止盈、继续持有建议
- 每日AI晨报

### 重要限制
1. 本系统是研究和决策辅助工具，不是收益保证，也不替代持牌投资顾问。
2. 免费行情源可能延迟、限流或临时失效；实盘前必须与券商行情核对。
3. 当前版本不会自动下单。自动交易必须另接券商 QMT、PTrade 或合规交易接口。
4. 回测没有完整覆盖涨跌停、停牌、滑点、成交冲击和所有交易费用。
5. 买卖信号应结合公告、财务、行业、政策和个人风险承受能力。
    """)
