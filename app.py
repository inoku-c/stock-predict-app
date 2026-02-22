import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

st.title('🚀 AI株価予測アプリ (プロトタイプ)')

# 1. 銘柄選択（例: Apple, NVIDIA, トヨタなど）
ticker = st.text_input('ティッカーシンボルを入力してください (例: AAPL, NVDA, 7203.T)', 'AAPL')

# 2. データ取得
data = yf.download(ticker, start="2020-01-01")

st.subheader(f'{ticker} の過去チャート')
st.line_chart(data['Close'])

# 3. AI予測 (Prophet)
if st.button('将来を予測する'):
    df_train = data.reset_index()[['Date', 'Close']]
    df_train.columns = ['ds', 'y'] # Prophet指定の列名
    
    model = Prophet()
    model.fit(df_train)
    
    # 今後30日間を予測
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    st.subheader('30日後の予測結果')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='予測値'))
    st.plotly_chart(fig)
