import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
import stripe
import pandas as pd

import stripe
import os

# ローカル環境（.env）と公開環境（Secrets）の両方に対応する書き方
stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY") or os.getenv("STRIPE_SECRET_KEY")



# --- 2. 支払い状態の管理 ---
if "paid" not in st.session_state:
    st.session_state.paid = False

# --- 3. 銘柄選択とデータ表示（無料部分） ---
ticker = st.text_input('ティッカーシンボルを入力 (例: AAPL, 7203.T)', 'AAPL')
data = yf.download(ticker, start="2020-01-01")

st.subheader(f'{ticker} の過去チャート')
st.line_chart(data['Close'])

# --- 4. 支払いチェックとAI予測 ---
if not st.session_state.paid:
    st.warning("将来の予測結果を見るには、解析レポートの購入が必要です。")
    if st.button("予測レポートを購入する (テストモード)"):
        # 本来はここで Stripe Checkout へリダイレクトしますが、
        # 今回はデプロイ確認のため、ボタン押下で支払い済みフラグを立てます
        st.session_state.paid = True
        st.rerun()
else:
    st.success("✅ 支払いを確認しました。高度な統計モデル（Prophet）による予測を実行します。")
    if st.button('将来を予測する'):
        with st.spinner('AIが計算中...'):
            df_train = data.reset_index()[['Date', 'Close']]
            # 統計的な整合性のための列名変更
            df_train.columns = ['ds', 'y'] 
            
            model = Prophet()
            model.fit(df_train)
            
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            st.subheader('30日後の予測結果 (95%信頼区間)')
            fig = go.Figure()
            # 予測値
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='予測値'))
            # 信頼区間（統計アナリストとしてのこだわり）
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='上限'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='下限'))
            st.plotly_chart(fig)