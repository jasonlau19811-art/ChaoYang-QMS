import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huntertrend.config import DEFAULT_SYMBOLS
from huntertrend.data import fetch_history, load_upload, demo_data
from huntertrend.engine import analyze
from huntertrend.backtest import backtest

st.set_page_config(page_title="HunterTrend Phase 1", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1250px;}
[data-testid="stMetric"] {background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08); padding: 12px; border-radius: 14px;}
[data-testid="stSidebar"] {min-width: 280px;}
.hero {padding: 18px 20px; border-radius: 18px; background: linear-gradient(135deg,#0f172a,#123524); border:1px solid rgba(255,255,255,.09); margin-bottom:14px;}
.hero h1 {margin:0;font-size:2rem}.hero p{margin:.4rem 0 0;color:#cbd5e1}
.signal {font-size:1.15rem;font-weight:700;padding:10px 14px;border-radius:12px;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3)}
@media (max-width: 700px) {
 .block-container {padding-left:.65rem;padding-right:.65rem;padding-top:.5rem;}
 .hero {padding:14px}.hero h1{font-size:1.45rem}
 [data-testid="stMetricValue"] {font-size:1.35rem;}
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>HunterTrend Pro · 第一阶段</h1><p>A股多模型量化决策平台｜安卓、电脑浏览器均可使用</p></div>', unsafe_allow_html=True)

@st.cache_data(ttl=1800, show_spinner=False)
def get_online(symbol: str):
    return fetch_history(symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def run_model_cached(df_json: str, fundamentals_tuple, horizon: int):
    df = pd.read_json(df_json, orient="split")
    fundamentals = dict(fundamentals_tuple) if fundamentals_tuple else None
    return analyze(df, fundamentals, horizon)

with st.sidebar:
    st.header("分析控制台")
    name = st.selectbox("关注标的", list(DEFAULT_SYMBOLS) + ["自定义"])
    symbol = DEFAULT_SYMBOLS.get(name, "")
    if name == "自定义":
        symbol = st.text_input("股票或ETF代码", "300124", max_chars=6)
    horizon = st.selectbox("预测周期（交易日）", [3, 5, 10, 20], index=1)
    source = st.radio("行情来源", ["AKShare在线", "上传CSV/XLSX", "演示数据"])
    upload = st.file_uploader("上传行情文件", type=["csv", "xlsx", "xls"]) if source == "上传CSV/XLSX" else None
    with st.expander("基本面数据（可选）"):
        use_f = st.checkbox("加入基本面评分")
        fundamentals = None
        if use_f:
            fundamentals = {
                "roe": st.number_input("ROE（%）", value=12.0),
                "profit_growth": st.number_input("净利润同比（%）", value=10.0),
                "debt_ratio": st.number_input("资产负债率（%）", value=45.0),
                "pe": st.number_input("PE(TTM)", value=25.0),
            }
    run = st.button("开始多模型分析", type="primary", use_container_width=True)
    st.caption("默认自选：汇川技术、特变电工、创业板ETF")

intro, analysis_tab, guide_tab = st.tabs(["自选概览", "量化分析", "使用说明"])

with intro:
    st.subheader("第一阶段已交付功能")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("默认自选", "3只")
    c2.metric("决策模型", "7类")
    c3.metric("预测周期", "3–20日")
    c4.metric("终端支持", "手机 + 电脑")
    st.markdown("""
- **汇川技术 300124**：工业自动化与高端制造方向
- **特变电工 600089**：电网设备、能源与周期属性
- **创业板ETF 159915**：分散持有创业板核心资产

系统最终输出综合评分、上涨概率、置信度、建议仓位、止损位、目标位、模型投票和历史回测。在线数据失败时，可切换上传文件或演示数据继续使用。
""")

if run:
    with analysis_tab:
        try:
            with st.spinner("正在读取数据并运行多模型…"):
                if source == "AKShare在线":
                    df = get_online(symbol)
                elif source == "上传CSV/XLSX":
                    if upload is None:
                        st.error("请先上传行情文件。")
                        st.stop()
                    df = load_upload(upload)
                else:
                    seed = int(symbol[-3:]) if symbol.isdigit() else 42
                    df = demo_data(seed=seed)
                fundamentals_tuple = tuple(fundamentals.items()) if fundamentals else None
                result = run_model_cached(df.to_json(orient="split", date_format="iso"), fundamentals_tuple, horizon)
                bt, metrics = backtest(df)

            st.subheader(f"{name}（{symbol or '演示'}）决策结果")
            st.markdown(f'<div class="signal">当前信号：{result["signal"]}</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            values = [
                ("综合评分", f'{result["score"]}/100'),
                ("上涨概率", f'{result["up_probability"]}%'),
                ("置信度", f'{result["confidence"]}%'),
                ("建议仓位上限", f'{result["position"]}%'),
                ("最新收盘", result["close"]),
            ]
            for col, (label, value) in zip(cols, values):
                col.metric(label, value)

            p1, p2 = st.columns(2)
            p1.metric("ATR参考止损", result["stop"], help="最新收盘价减2.2倍ATR")
            p2.metric("ATR参考目标", result["target"], help="最新收盘价加3.2倍ATR")

            labels = {
                "trend": "趋势", "mean_reversion": "均值回归", "momentum": "动量",
                "machine_learning": "机器学习", "risk": "风险质量",
                "fundamental": "基本面", "market_regime": "市场环境"
            }
            votes = pd.DataFrame([
                {"模型": labels[k], "评分": round(v, 2), "有效权重": round(result["weights"][k] * 100, 1)}
                for k, v in result["scores"].items()
            ])
            left, right = st.columns([1, 1.6])
            with left:
                st.subheader("模型投票")
                st.dataframe(votes, use_container_width=True, hide_index=True)
                st.caption(f'机器学习验证 AUC：{result["ml_auc"]}｜准确率：{result["ml_accuracy"]}')
            with right:
                st.subheader("近180日K线")
                fig = go.Figure(go.Candlestick(
                    x=df["date"].tail(180), open=df["open"].tail(180), high=df["high"].tail(180),
                    low=df["low"].tail(180), close=df["close"].tail(180), name="K线"
                ))
                fig.update_layout(height=430, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.subheader("规则层历史回测")
            mc = st.columns(len(metrics))
            for col, (key, value) in zip(mc, metrics.items()):
                col.metric(key, f"{value:.2f}" if key == "夏普比率" else f"{value:.1%}")
            eq = go.Figure()
            eq.add_trace(go.Scatter(x=bt["date"], y=bt["equity"], name="策略净值"))
            eq.add_trace(go.Scatter(x=bt["date"], y=bt["benchmark"], name="买入持有"))
            eq.update_layout(height=390, margin=dict(l=5, r=5, t=10, b=5), yaxis_title="净值")
            st.plotly_chart(eq, use_container_width=True, config={"displayModeBar": False})
            st.download_button("导出本次模型评分", votes.to_csv(index=False).encode("utf-8-sig"),
                               f"huntertrend_{symbol}_scores.csv", "text/csv", use_container_width=True)
            st.warning("量化结果用于辅助决策，不代表确定收益。实盘前应结合持仓成本、资金规模和走样本外验证。")
        except Exception as exc:
            st.error(f"运行失败：{exc}")
            st.info("可切换到“演示数据”，或上传包含日期、开盘、最高、最低、收盘、成交量字段的行情文件。")
else:
    with analysis_tab:
        st.info("请在左侧选择标的和数据源，然后点击“开始多模型分析”。")

with guide_tab:
    st.subheader("安卓手机打开方式")
    st.markdown("""
1. 将本项目部署至 Streamlit Community Cloud、Render 或其他云服务器。
2. 部署完成后会得到一个 HTTPS 网页地址。
3. 使用安卓 Chrome 浏览器打开该网址，并选择“添加到主屏幕”，即可像应用一样启动。

### 第一阶段边界
已完成移动网页、行情读取、多模型决策、自选标的、K线、回测和部署配置。账号登录、自动收盘更新、消息推送和全市场自动选股属于后续阶段。
""")
