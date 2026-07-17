import json
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
    page_title="HunterTrend V2.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_WATCHLIST = {
    "汇川技术": "300124",
    "特变电工": "600089",
    "创业板ETF": "159915",
}

ALIASES = {
    "日期": "date",
    "交易日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}

ML_FEATURES = [
    "ret1", "ret5", "ret10", "ret20", "ret60", "ret120",
    "pma5", "pma20", "pma60", "pma120",
    "macdh", "rsi", "atrp", "vol20", "vratio", "bbpos", "dd60",
]


def init_state():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("行情数据为空。")
    x = df.copy()
    x.columns = [ALIASES.get(str(c).strip(), str(c).strip().lower()) for c in x.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"缺少字段：{sorted(missing)}")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = (
        x.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if len(x) < 120:
        raise ValueError("至少需要120个交易日数据，建议300日以上。")
    return x


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(symbol: str) -> pd.DataFrame:
    import akshare as ak

    symbol = str(symbol).strip().zfill(6)
    errors = []

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date="20180101",
            end_date="20991231",
            adjust="qfq",
        )
        return normalize(df)
    except Exception as exc:
        errors.append(f"股票接口：{exc}")

    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date="20180101",
            end_date="20991231",
            adjust="qfq",
        )
        return normalize(df)
    except Exception as exc:
        errors.append(f"ETF接口：{exc}")

    raise RuntimeError("在线行情获取失败：" + " | ".join(errors))


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def build_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    x = df.copy()
    close = x["close"]
    volume = x["volume"]

    x["ret1"] = close.pct_change()
    for n in [5, 10, 20, 60, 120]:
        x[f"ret{n}"] = close.pct_change(n)
        x[f"ma{n}"] = close.rolling(n).mean()
        x[f"pma{n}"] = close / x[f"ma{n}"] - 1

    x["ema12"] = close.ewm(span=12, adjust=False).mean()
    x["ema26"] = close.ewm(span=26, adjust=False).mean()
    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macdh"] = x["macd"] - x["macd_signal"]
    x["rsi"] = rsi(close)
    x["atr"] = atr(x)
    x["atrp"] = x["atr"] / close
    x["vol20"] = x["ret1"].rolling(20).std() * np.sqrt(252)
    x["vratio"] = volume.rolling(5).mean() / volume.rolling(20).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    x["bbpos"] = (close - lower) / (upper - lower)
    x["dd60"] = close / close.rolling(60).max() - 1

    x["future"] = close.shift(-horizon) / close - 1
    x["target"] = (x["future"] > 0).astype(float)
    return x


def clip(value: float) -> float:
    return float(np.clip(value, 0, 100))


def rule_scores(row: pd.Series) -> dict:
    trend = 50
    trend += 10 if row["close"] > row["ma20"] else -10
    trend += 12 if row["ma20"] > row["ma60"] else -12
    trend += 8 if row["ma60"] > row["ma120"] else -8
    trend += 10 if row["macdh"] > 0 else -10
    trend += 5 if row["vratio"] > 1.15 else 0

    momentum = 50
    for col, weight in [("ret20", 16), ("ret60", 18), ("ret120", 12)]:
        momentum += weight if row[col] > 0 else -weight
    momentum += 6 if row["vratio"] > 1 else -3

    mean_reversion = 50
    if row["rsi"] < 30:
        mean_reversion += 25
    elif row["rsi"] < 40:
        mean_reversion += 12
    elif row["rsi"] > 75:
        mean_reversion -= 25
    elif row["rsi"] > 65:
        mean_reversion -= 12

    if row["bbpos"] < 0.05:
        mean_reversion += 15
    elif row["bbpos"] > 0.95:
        mean_reversion -= 15

    risk_quality = 80
    risk_quality -= max(0, (row["vol20"] - 0.22) * 100)
    risk_quality += row["dd60"] * 80
    risk_quality -= max(0, (row["atrp"] - 0.025) * 500)

    return {
        "趋势": clip(trend),
        "动量": clip(momentum),
        "均值回归": clip(mean_reversion),
        "风险质量": clip(risk_quality),
    }


def ml_probability(features: pd.DataFrame) -> tuple[float, float | None]:
    data = features.dropna(subset=["target"]).copy()
    if len(data) < 220:
        return 50.0, None

    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    valid = data.iloc[split:]

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=5,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[ML_FEATURES], train["target"].astype(int))

    valid_prob = model.predict_proba(valid[ML_FEATURES])[:, 1]
    latest_prob = float(
        model.predict_proba(features[ML_FEATURES].iloc[[-1]])[:, 1][0]
    )

    try:
        auc = float(roc_auc_score(valid["target"].astype(int), valid_prob))
    except Exception:
        auc = None

    return latest_prob * 100, auc


def analyze(df: pd.DataFrame, horizon: int = 5) -> dict:
    feature_df = build_features(df, horizon)
    row = feature_df.iloc[-1]

    scores = rule_scores(row)
    ml_score, auc = ml_probability(feature_df)
    scores["机器学习"] = ml_score

    weights = {
        "趋势": 0.27,
        "动量": 0.22,
        "均值回归": 0.14,
        "风险质量": 0.17,
        "机器学习": 0.20,
    }
    composite = sum(scores[k] * weights[k] for k in scores)

    agreement = max(0, 1 - np.std(list(scores.values())) / 35)
    data_quality = min(1, len(df) / 500)
    confidence = 100 * (0.65 * agreement + 0.35 * data_quality)

    volatility = row["vol20"] if pd.notna(row["vol20"]) else 0.35
    risk_multiplier = np.clip(0.30 / max(volatility, 0.12), 0.45, 1.10)
    position = np.clip(
        max(0, (composite - 42) / 58)
        * risk_multiplier
        * confidence
        / 100,
        0,
        0.80,
    )

    close = float(row["close"])
    atr_value = float(row["atr"]) if pd.notna(row["atr"]) else close * 0.04

    if composite >= 80:
        signal = "强势买入候选"
    elif composite >= 68:
        signal = "分批买入"
    elif composite >= 52:
        signal = "持有/观察"
    elif composite >= 38:
        signal = "减仓"
    else:
        signal = "规避/卖出"

    return {
        "score": round(float(composite), 1),
        "signal": signal,
        "confidence": round(float(confidence), 1),
        "probability": round(float(ml_score), 1),
        "position": round(float(position * 100), 1),
        "close": round(close, 3),
        "stop": round(close - 2.2 * atr_value, 3),
        "target_price": round(close + 3.2 * atr_value, 3),
        "rsi": round(float(row["rsi"]), 1),
        "volatility": round(float(volatility * 100), 1),
        "scores": scores,
        "auc": auc,
        "feature_df": feature_df,
    }


def diagnosis_text(name: str, symbol: str, result: dict) -> str:
    positives = []
    risks = []

    if result["scores"]["趋势"] >= 65:
        positives.append("中期趋势结构偏强")
    else:
        risks.append("趋势尚未形成明显优势")

    if result["scores"]["动量"] >= 65:
        positives.append("20至120日动量较好")
    else:
        risks.append("价格动量偏弱或分化")

    if result["rsi"] >= 70:
        risks.append("RSI偏高，短线存在过热风险")
    elif result["rsi"] <= 35:
        positives.append("RSI处于相对低位，存在修复空间")

    if result["volatility"] >= 40:
        risks.append("近20日波动率较高")
    else:
        positives.append("近期波动处于可控范围")

    positive_text = "；".join(positives) if positives else "暂未出现明显优势信号"
    risk_text = "；".join(risks) if risks else "暂未发现突出的技术风险"

    return f"""
### {name}（{symbol}）诊断结论

**综合评分：{result["score"]}/100　｜　信号：{result["signal"]}**

**上涨概率：{result["probability"]}%　｜　置信度：{result["confidence"]}%**

**积极因素：** {positive_text}。

**主要风险：** {risk_text}。

**仓位参考：** 模型建议上限约为 **{result["position"]}%**。  
**价格纪律：** ATR参考止损 **{result["stop"]}**，参考目标 **{result["target_price"]}**。

该结论来自历史行情与技术模型，不代表未来必然表现。
"""


def watchlist_editor():
    st.subheader("管理自选股")

    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        new_name = st.text_input("股票名称", placeholder="例如：宁德时代")
    with c2:
        new_symbol = st.text_input("股票/ETF代码", placeholder="例如：300750")
    with c3:
        st.write("")
        st.write("")
        if st.button("添加自选", use_container_width=True):
            name = new_name.strip()
            symbol = new_symbol.strip()
            if not name or not symbol:
                st.warning("请填写名称和代码。")
            elif not symbol.isdigit() or len(symbol) > 6:
                st.warning("代码应为不超过6位的数字。")
            else:
                st.session_state.watchlist[name] = symbol.zfill(6)
                st.success(f"已添加：{name} {symbol.zfill(6)}")
                st.rerun()

    if st.session_state.watchlist:
        remove_name = st.selectbox(
            "选择要删除的自选股", list(st.session_state.watchlist.keys())
        )
        if st.button("删除所选", type="secondary"):
            if remove_name in DEFAULT_WATCHLIST:
                st.warning("默认标的也可以删除；再次点击确认。")
                confirm_key = f"confirm_{remove_name}"
                if st.session_state.get(confirm_key):
                    del st.session_state.watchlist[remove_name]
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                st.session_state[confirm_key] = True
            else:
                del st.session_state.watchlist[remove_name]
                st.rerun()

    watchlist_df = pd.DataFrame(
        [{"股票": name, "代码": symbol} for name, symbol in st.session_state.watchlist.items()]
    )
    st.dataframe(watchlist_df, use_container_width=True, hide_index=True)

    st.download_button(
        "导出自选股 CSV",
        watchlist_df.to_csv(index=False).encode("utf-8-sig"),
        "huntertrend_watchlist.csv",
        "text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("导入自选股 CSV", type=["csv"])
    if uploaded is not None:
        try:
            imported = pd.read_csv(uploaded)
            if not {"股票", "代码"}.issubset(imported.columns):
                st.error("CSV必须包含“股票”和“代码”两列。")
            elif st.button("确认导入"):
                for _, row in imported.iterrows():
                    name = str(row["股票"]).strip()
                    symbol = str(row["代码"]).split(".")[0].strip().zfill(6)
                    if name and symbol.isdigit():
                        st.session_state.watchlist[name] = symbol
                st.success("自选股已导入。")
                st.rerun()
        except Exception as exc:
            st.error(f"导入失败：{exc}")


init_state()

st.title("📈 HunterTrend V2.1｜自选股智能分析")
st.caption("支持自选股管理、批量评分、排行榜与单股诊断。仅用于研究和辅助决策。")

page = st.sidebar.radio(
    "功能模块",
    ["自选股总览", "单股深度诊断", "管理自选股", "模型说明"],
)
horizon = st.sidebar.selectbox("预测周期（交易日）", [3, 5, 10, 20], index=1)

if page == "自选股总览":
    st.subheader("自选股批量分析")

    if st.button("分析全部自选股", type="primary", use_container_width=True):
        rows = []
        progress = st.progress(0)
        names = list(st.session_state.watchlist.items())

        for index, (name, symbol) in enumerate(names, start=1):
            try:
                df = fetch_history(symbol)
                result = analyze(df, horizon)
                rows.append(
                    {
                        "股票": name,
                        "代码": symbol,
                        "最新价": result["close"],
                        "综合评分": result["score"],
                        "上涨概率%": result["probability"],
                        "置信度%": result["confidence"],
                        "建议仓位%": result["position"],
                        "信号": result["signal"],
                        "RSI": result["rsi"],
                        "波动率%": result["volatility"],
                        "参考止损": result["stop"],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "股票": name,
                        "代码": symbol,
                        "最新价": np.nan,
                        "综合评分": np.nan,
                        "上涨概率%": np.nan,
                        "置信度%": np.nan,
                        "建议仓位%": np.nan,
                        "信号": f"数据失败：{str(exc)[:40]}",
                        "RSI": np.nan,
                        "波动率%": np.nan,
                        "参考止损": np.nan,
                    }
                )
            progress.progress(index / max(len(names), 1))

        st.session_state.batch_results = (
            pd.DataFrame(rows)
            .sort_values("综合评分", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    if not st.session_state.batch_results.empty:
        results = st.session_state.batch_results

        valid = results.dropna(subset=["综合评分"])
        c1, c2, c3 = st.columns(3)
        c1.metric("自选股数量", len(results))
        c2.metric(
            "最高评分",
            "-" if valid.empty else f'{valid["综合评分"].max():.1f}',
        )
        c3.metric(
            "平均建议仓位",
            "-" if valid.empty else f'{valid["建议仓位%"].mean():.1f}%',
        )

        st.dataframe(results, use_container_width=True, hide_index=True)

        st.download_button(
            "下载批量分析结果",
            results.to_csv(index=False).encode("utf-8-sig"),
            "huntertrend_watchlist_analysis.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.info("点击“分析全部自选股”，系统会逐只获取行情并生成排行榜。")

elif page == "单股深度诊断":
    if not st.session_state.watchlist:
        st.warning("自选股为空，请先添加股票。")
        st.stop()

    selected_name = st.selectbox("选择自选股", list(st.session_state.watchlist.keys()))
    selected_symbol = st.session_state.watchlist[selected_name]

    if st.button("运行单股诊断", type="primary", use_container_width=True):
        try:
            with st.spinner("正在获取行情并运行模型..."):
                stock_df = fetch_history(selected_symbol)
                result = analyze(stock_df, horizon)

            cols = st.columns(5)
            metrics = [
                ("综合评分", f'{result["score"]}/100'),
                ("信号", result["signal"]),
                ("上涨概率", f'{result["probability"]}%'),
                ("置信度", f'{result["confidence"]}%'),
                ("建议仓位", f'{result["position"]}%'),
            ]
            for col, (label, value) in zip(cols, metrics):
                col.metric(label, value)

            st.markdown(diagnosis_text(selected_name, selected_symbol, result))

            vote_df = pd.DataFrame(
                [
                    {"模型": model, "评分": round(score, 1)}
                    for model, score in result["scores"].items()
                ]
            )
            st.subheader("模型投票")
            st.dataframe(vote_df, use_container_width=True, hide_index=True)

            fig = go.Figure(
                go.Candlestick(
                    x=stock_df["date"].tail(180),
                    open=stock_df["open"].tail(180),
                    high=stock_df["high"].tail(180),
                    low=stock_df["low"].tail(180),
                    close=stock_df["close"].tail(180),
                    name="K线",
                )
            )
            fig.update_layout(height=460, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            if result["auc"] is not None:
                st.info(f'机器学习历史验证 AUC：{result["auc"]:.3f}')
            else:
                st.info("有效样本不足，机器学习模型采用中性评分。")

        except Exception as exc:
            st.exception(exc)

elif page == "管理自选股":
    watchlist_editor()

else:
    st.subheader("模型说明")
    st.markdown(
        """
### 当前模型构成

- **趋势模型**：MA20、MA60、MA120、MACD与成交量确认
- **动量模型**：20日、60日、120日收益趋势
- **均值回归模型**：RSI与布林带位置
- **风险模型**：波动率、60日回撤与ATR
- **机器学习模型**：随机森林，使用时间顺序切分进行历史验证

### 自选股分析流程

1. 从 AKShare 获取前复权历史行情。
2. 对每只股票计算模型评分。
3. 输出综合评分、上涨概率、置信度、建议仓位和ATR止损。
4. 按综合评分生成自选股排行榜。

应用不会自动下单，所有结果仅供研究参考。
"""
    )
