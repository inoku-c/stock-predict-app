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
    # --- AI予測 (Prophet) ---
    if st.button('将来を予測する'):
        # 翌日（1日分）だけを予測
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        
        # 翌日のデータだけを抽出
        tomorrow = forecast.iloc[-1]
        
        st.subheader(f"📅 {tomorrow['ds'].strftime('%Y-%m-%d')} の予測結果")
        
        # メトリクス（大きな数字）で表示
        col1, col2, col3 = st.columns(3)
        col1.metric("予測価格", f"${tomorrow['yhat']:.2f}")
        col2.metric("95%下限 (CI)", f"${tomorrow['yhat_lower']:.2f}")
        col3.metric("95%上限 (CI)", f"${tomorrow['yhat_upper']:.2f}")
        
        # チャートを直近30日に絞って表示（視認性向上）
        fig = model.plot(forecast)
        plt.xlim(forecast['ds'].iloc[-30], forecast['ds'].iloc[-1]) 
        st.pyplot(fig)