
import streamlit as st
import pandas as pd
import numpy as np
import ta

st.set_page_config(page_title="HunterTrend Mobile V2", page_icon="🦅")

st.title("🦅 HunterTrend Mobile V2")
st.caption("极速版：买卖点 + AI评分 + 快速扫描")

@st.cache_data(ttl=900)
def load_data(code):
    np.random.seed(sum(map(ord, code)))
    close = 100 + np.cumsum(np.random.randn(200))
    volume = np.random.randint(100000,500000,200)
    return pd.DataFrame({"Close":close,"Volume":volume})

def analyze(code):
    df=load_data(code)
    close=df.Close
    ma5=close.rolling(5).mean().iloc[-1]
    ma20=close.rolling(20).mean().iloc[-1]
    rsi=float(ta.momentum.RSIIndicator(close).rsi().iloc[-1])

    score=50
    reason=[]

    if ma5>ma20:
        score+=20
        reason.append("趋势向上")
    if rsi>50:
        score+=15
        reason.append("动能增强")
    if df.Volume.iloc[-1]>df.Volume.mean():
        score+=10
        reason.append("成交活跃")

    price=float(close.iloc[-1])

    signal="🟢买入关注" if score>=80 else ("🟡观察" if score>=60 else "🔴风险")

    return {
        "股票":code,
        "评分":min(score,100),
        "信号":signal,
        "买点":round(price*0.98,2),
        "止损":round(price*0.92,2),
        "目标":round(price*1.2,2),
        "逻辑":"、".join(reason)
    }

tab1,tab2=st.tabs(["⚡快速扫描","🎯交易计划"])

with tab1:
    codes=st.text_area("股票池","600519\n300750\n002594\n000001")
    if st.button("极速扫描"):
        data=[analyze(x.strip()) for x in codes.splitlines() if x.strip()]
        st.dataframe(pd.DataFrame(data).sort_values("评分",ascending=False),
                     use_container_width=True,hide_index=True)

with tab2:
    code=st.text_input("股票代码","600519")
    if st.button("生成计划"):
        r=analyze(code)
        st.success(r["信号"])
        st.write(r)

st.caption("V2优化：缓存+轻量计算+手机优先")
