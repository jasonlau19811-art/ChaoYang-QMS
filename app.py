import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import ta

st.set_page_config(page_title='HunterTrend Final Optimized', page_icon='🦅', layout='centered', initial_sidebar_state='collapsed')
st.title('🦅 HunterTrend Final Optimized')
st.caption('专业功能保留｜手机极速架构｜股票名称自动识别')

@st.cache_data(ttl=86400, show_spinner=False)
def get_name_map():
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        df['code'] = df['code'].astype(str).str.zfill(6)
        return dict(zip(df['code'], df['name']))
    except Exception:
        return {}

def stock_name(code):
    code = str(code).strip().zfill(6)
    return get_name_map().get(code, '名称暂缺')

@st.cache_data(ttl=1800, show_spinner=False, max_entries=300)
def get_history(code):
    code = str(code).strip().zfill(6)
    try:
        import akshare as ak
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=800)).strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start_date, end_date=end_date, adjust='qfq')
        if df is not None and not df.empty:
            df = df.rename(columns={'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ['Open','High','Low','Close','Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.set_index('Date')[['Open','High','Low','Close','Volume']].dropna()
            if len(df) >= 130:
                return df, 'AKShare'
    except Exception:
        pass

    import yfinance as yf
    suffix = '.SS' if code.startswith(('5','6','9')) else '.SZ'
    df = yf.download(code + suffix, period='3y', auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    if len(df) < 130:
        raise ValueError('历史行情不足')
    return df, '备用行情源'

def f(v, default=0.0):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def enrich(df):
    x = df.copy()
    c,h,l,v = x['Close'],x['High'],x['Low'],x['Volume']
    for n in [5,10,20,60,120]:
        x[f'MA{n}'] = c.rolling(n).mean()
    macd = ta.trend.MACD(c)
    x['MACD'] = macd.macd(); x['MACD_SIGNAL'] = macd.macd_signal()
    x['RSI'] = ta.momentum.RSIIndicator(c,14).rsi()
    stoch = ta.momentum.StochasticOscillator(h,l,c,9,3)
    x['K'] = stoch.stoch(); x['D'] = stoch.stoch_signal()
    x['ATR'] = ta.volatility.AverageTrueRange(h,l,c,14).average_true_range()
    x['ADX'] = ta.trend.ADXIndicator(h,l,c,14).adx()
    x['VOL_MA20'] = v.rolling(20).mean()
    x['HIGH20'] = h.rolling(20).max()
    x['VOLATILITY20'] = c.pct_change().rolling(20).std() * np.sqrt(252)
    return x

def score_stock(x, market_score, style):
    r,p = x.iloc[-1],x.iloc[-2]
    price = f(r['Close']); atr = max(f(r['ATR'], price*.025), price*.008)
    recent = x.tail(60)
    support = f(recent['Low'].min(), price*.92); resistance = f(recent['High'].max(), price*1.08)
    score = 35; reasons=[]; risks=[]
    if r['MA5'] > r['MA10'] > r['MA20']:
        score += 15; reasons.append('短期均线多头')
    elif r['MA5'] < r['MA10']:
        score -= 8; risks.append('短期趋势转弱')
    if r['MA20'] > r['MA60']:
        score += 12; reasons.append('中期趋势向上')
    else:
        risks.append('中期趋势未确认')
    if r['MACD'] > r['MACD_SIGNAL']:
        score += 10; reasons.append('MACD维持多头')
        if p['MACD'] <= p['MACD_SIGNAL']:
            score += 5; reasons.append('MACD刚形成金叉')
    else:
        score -= 8; risks.append('MACD处于弱势')
    if 50 <= r['RSI'] <= 70:
        score += 10; reasons.append('RSI健康强势')
    elif r['RSI'] > 80:
        score -= 10; risks.append('RSI过热')
    score += 6 if r['K'] > r['D'] else -3
    vol_ratio = f(r['Volume']/r['VOL_MA20'],1)
    if 1.2 <= vol_ratio <= 3:
        score += 8; reasons.append('成交量放大')
    elif vol_ratio > 3.5:
        score -= 4; risks.append('成交异常放大')
    if r['Close'] >= p['HIGH20']*.995:
        score += 8; reasons.append('接近阶段突破')
    if f(r['VOLATILITY20']) > .55:
        score -= 8; risks.append('波动率偏高')
    score += market_score*.12
    score = int(max(0,min(100,round(score))))
    buy_cut,hold_cut,max_pos = {'保守':(88,74,.25),'稳健':(82,68,.35),'积极':(76,62,.45)}[style]
    if score >= buy_cut and len(risks) <= 1:
        signal,position='🟢 A级买入关注',max_pos
    elif score >= hold_cut:
        signal,position='🟡 试仓 / 持有',max_pos*.55
    elif score >= 52:
        signal,position='🟠 观望等待',max_pos*.2
    else:
        signal,position='🔴 减仓 / 回避',0.0
    buy_low=max(support,price-.55*atr); buy_high=price+.15*atr
    stop=max(support-.3*atr,price-2*atr)
    if stop >= price: stop=price-1.8*atr
    risk_per_share=max(price-stop,price*.01)
    return {'price':price,'score':score,'signal':signal,'position':position,'buy_low':buy_low,'buy_high':buy_high,'stop':stop,'target1':price+2*risk_per_share,'target2':price+3.2*risk_per_share,'support':support,'resistance':resistance,'reasons':reasons,'risks':risks}

@st.cache_data(ttl=1800, show_spinner=False, max_entries=300)
def analyze_cached(code, market_score, style):
    code = str(code).strip().zfill(6)
    raw,source = get_history(code)
    result = score_stock(enrich(raw), market_score, style)
    result.update({'code':code,'name':stock_name(code),'source':source,'last_date':str(raw.index[-1].date())})
    return result

def parse_codes(text):
    text=text.replace('\n',',').replace(' ', ',').replace('，', ',')
    return list(dict.fromkeys(x.strip().zfill(6) for x in text.split(',') if x.strip()))

def position_plan(r,capital,risk_pct):
    risk_per_share=max(r['price']-r['stop'],r['price']*.01)
    by_risk=math.floor((capital*risk_pct/100)/risk_per_share/100)*100
    by_cap=math.floor((capital*r['position'])/r['price']/100)*100
    shares=max(0,min(by_risk,by_cap))
    return shares,shares*r['price'],shares*risk_per_share

def scan_codes(codes,market_score,style):
    rows=[]
    workers=min(4,max(1,len(codes)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(analyze_cached,c,market_score,style):c for c in codes}
        for future in as_completed(futures):
            code=futures[future]
            try:
                r=future.result()
                rows.append({'股票代码':code,'股票名称':r['name'],'评分':r['score'],'信号':r['signal'],'现价':round(r['price'],2),'买入区间':f"{r['buy_low']:.2f}-{r['buy_high']:.2f}",'止损':round(r['stop'],2),'目标一':round(r['target1'],2),'数据源':r['source']})
            except Exception as exc:
                rows.append({'股票代码':code,'股票名称':stock_name(code),'信号':f'失败：{str(exc)[:18]}'})
    frame=pd.DataFrame(rows)
    if '评分' in frame.columns:
        frame=frame.sort_values('评分',ascending=False,na_position='last')
    return frame

with st.expander('账户与策略设置', expanded=False):
    capital=st.number_input('账户资金（元）',1000.0,100000000.0,100000.0,1000.0)
    risk_pct=st.slider('单笔最大亏损预算',.5,5.0,2.0,.5)
    style=st.selectbox('交易风格',['保守','稳健','积极'],index=1)
    market_score=st.slider('市场环境评分',0,100,60)

page=st.radio('功能',['单股决策','自选股雷达','持仓诊断','AI晨报'],horizontal=True)

if page=='单股决策':
    code=st.text_input('股票代码','600519').strip().zfill(6)
    if st.button('生成专业交易计划',type='primary',use_container_width=True):
        try:
            with st.spinner('首次查询稍慢，30分钟内再次查询将直接使用缓存…'):
                r=analyze_cached(code,market_score,style)
                raw,_=get_history(code); x=enrich(raw)
            st.subheader(f"{r['name']}（{code}）")
            st.caption(f"数据源：{r['source']}｜最后交易日：{r['last_date']}")
            c1,c2,c3=st.columns(3)
            c1.metric('当前价',f"{r['price']:.2f}"); c2.metric('AI评分',f"{r['score']}/100"); c3.metric('操作信号',r['signal'])
            shares,amount,max_loss=position_plan(r,capital,risk_pct)
            plan=pd.DataFrame({'项目':['股票名称','股票代码','买入区间','支撑位','压力位','止损位','第一目标','第二目标','建议股数','预计投入','最大计划亏损'],'结果':[r['name'],code,f"{r['buy_low']:.2f}-{r['buy_high']:.2f}",f"{r['support']:.2f}",f"{r['resistance']:.2f}",f"{r['stop']:.2f}",f"{r['target1']:.2f}",f"{r['target2']:.2f}",f'{shares} 股',f'{amount:,.0f} 元',f'{max_loss:,.0f} 元']})
            st.dataframe(plan,hide_index=True,use_container_width=True)
            st.line_chart(x[['Close','MA5','MA20','MA60']].tail(160).rename(columns={'Close':'收盘价'}))
            left,right=st.columns(2)
            with left:
                st.markdown('#### 买入依据')
                for item in r['reasons'] or ['暂无明显多信号共振']: st.write('✅',item)
            with right:
                st.markdown('#### 风险与卖出依据')
                for item in r['risks'] or ['暂未发现明显趋势破坏']: st.write('⚠️',item)
        except Exception as exc:
            st.error(f'分析失败：{exc}')

elif page=='自选股雷达':
    pool=st.text_area('自选股代码','600519\n300750\n002594\n000001\n601318\n600036',height=130)
    if st.button('并发扫描自选股',type='primary',use_container_width=True):
        codes=parse_codes(pool)
        if len(codes)>20:
            st.warning('手机端建议一次扫描不超过20只。'); codes=codes[:20]
        with st.spinner('正在并发扫描，后续30分钟内会明显更快…'):
            result=scan_codes(codes,market_score,style)
        st.dataframe(result,hide_index=True,use_container_width=True)
        st.download_button('下载扫描结果',result.to_csv(index=False).encode('utf-8-sig'),'HunterTrend_watchlist.csv','text/csv')

elif page=='持仓诊断':
    uploaded=st.file_uploader('上传持仓CSV，列名：股票代码、持仓成本、持仓股数',type=['csv'])
    if uploaded is not None:
        holdings=pd.read_csv(uploaded,dtype={'股票代码':str}); holdings['股票代码']=holdings['股票代码'].astype(str).str.zfill(6)
        if st.button('分析全部持仓',type='primary',use_container_width=True):
            radar=scan_codes(holdings['股票代码'].tolist(),market_score,style)
            merged=holdings.merge(radar,on='股票代码',how='left')
            merged['浮动盈亏']=((merged['现价']-merged['持仓成本'])*merged['持仓股数']).round(2)
            merged['收益率']=(merged['现价']/merged['持仓成本']-1).map(lambda v:f'{v:.1%}' if pd.notna(v) else '')
            st.dataframe(merged,hide_index=True,use_container_width=True)
    else:
        st.info('请先上传持仓CSV。')

else:
    pool=st.text_area('晨报股票池','600519\n300750\n002594\n000001\n601318\n600036',height=130)
    if st.button('生成AI晨报',type='primary',use_container_width=True):
        codes=parse_codes(pool)[:20]
        with st.spinner('正在生成晨报…'):
            report=scan_codes(codes,market_score,style)
        st.subheader(f"AI晨报｜{datetime.now().strftime('%Y-%m-%d')}")
        st.write(f'市场环境：**{market_score}/100**｜风格：**{style}**')
        st.dataframe(report,hide_index=True,use_container_width=True)
        if '评分' in report.columns and report['评分'].notna().any():
            top=report.dropna(subset=['评分']).iloc[0]
            st.success(f"股票池首选：{top['股票名称']}（{top['股票代码']}），评分 {int(top['评分'])}")

st.caption('优化：行情缓存30分钟、股票名称缓存24小时、按需加载、最多4线程并发。')
st.warning('仅供研究和辅助决策，不构成投资建议；实盘前请与券商行情核对。')
