import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import stripe
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Predictor Pro", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY") or os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = st.secrets.get("STRIPE_PRICE_ID") or os.getenv("STRIPE_PRICE_ID")
APP_URL = st.secrets.get("APP_URL") or os.getenv("APP_URL", "http://localhost:8501")

if "paid" not in st.session_state:
    st.session_state.paid = False
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "subscription_id" not in st.session_state:
    st.session_state.subscription_id = None

def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode="subscription",
            success_url=f"{APP_URL}?session_id={{CHECKOUT_SESSION_ID}}&status=success",
            cancel_url=f"{APP_URL}?status=cancel",
        )
        return session
    except Exception as e:
        st.error(f"決済セッション作成エラー: {e}")
        return None

def check_payment_status():
    params = st.query_params
    if params.get("status") == "success" and params.get("session_id"):
        try:
            session = stripe.checkout.Session.retrieve(params["session_id"])
            if session.payment_status == "paid" or session.status == "complete":
                st.session_state.paid = True
                st.session_state.customer_id = session.customer
                st.session_state.subscription_id = session.subscription
                return True
        except Exception:
            pass
    return st.session_state.paid

def create_portal_session(customer_id):
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id, return_url=APP_URL)
        return session.url
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_data(ticker, start_date="2015-01-01"):
    try:
        data = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if data is None or data.empty:
            data = yf.download(ticker, start=start_date, progress=False)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if "Adj Close" in data.columns and "Close" not in data.columns:
            data["Close"] = data["Adj Close"]
        return data.dropna()
    except Exception as e:
        st.warning(f"データ取得リトライ中... ({e})")
        try:
            import time
            time.sleep(2)
            data = yf.download(ticker, start=start_date, progress=False)
            if data is None or data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data.dropna()
        except Exception:
            return None

def calc_technical_indicators(data):
    df = data.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

def prepare_lstm_data(data, look_back=60):
    from sklearn.preprocessing import MinMaxScaler
    close = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y, scaler

def build_and_train_lstm(X_train, y_train, look_back=60):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              callbacks=[EarlyStopping(monitor='loss', patience=5,
                                       restore_best_weights=True)], verbose=0)
    return model

def predict_lstm(data, forecast_days=30, look_back=60):
    X, y, scaler = prepare_lstm_data(data, look_back)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = build_and_train_lstm(X_train, y_train, look_back)
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    close = data["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(close)
    last_seq = scaled[-look_back:]
    future_preds = []
    current_seq = last_seq.copy()
    for _ in range(forecast_days):
        pred = model.predict(current_seq.reshape(1, look_back, 1), verbose=0)
        future_preds.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred.reshape(1, 1), axis=0)
    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((test_pred.flatten() - y_test_actual.flatten()) ** 2))
    mape = np.mean(np.abs(
        (y_test_actual.flatten() - test_pred.flatten()) / y_test_actual.flatten())) * 100
    return {"future_preds": future_preds, "test_pred": test_pred.flatten(),
            "test_actual": y_test_actual.flatten(), "rmse": rmse,
            "mape": mape, "test_dates": data.index[split + look_back:],
            "model_name": "LSTM"}

def predict_rf(data, forecast_days=30, look_back=60):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    close = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = RandomForestRegressor(n_estimators=200, max_depth=20,
                                   min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    test_pred = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    last_seq = scaled[-look_back:, 0].copy()
    future_preds = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, -1))
        future_preds.append(pred[0])
        last_seq = np.append(last_seq[1:], pred)
    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((test_pred - y_test_actual) ** 2))
    mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100
    return {"future_preds": future_preds, "test_pred": test_pred,
            "test_actual": y_test_actual, "rmse": rmse, "mape": mape,
            "test_dates": data.index[split + look_back:], "model_name": "RandomForest"}

def predict_gb(data, forecast_days=30, look_back=60):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler
    close = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    test_pred = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    last_seq = scaled[-look_back:, 0].copy()
    future_preds = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, -1))
        future_preds.append(pred[0])
        last_seq = np.append(last_seq[1:], pred)
    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((test_pred - y_test_actual) ** 2))
    mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100
    return {"future_preds": future_preds, "test_pred": test_pred,
            "test_actual": y_test_actual, "rmse": rmse, "mape": mape,
            "test_dates": data.index[split + look_back:], "model_name": "GradientBoosting"}

def predict_ensemble(data, forecast_days=30, look_back=60, models=None):
    if models is None:
        models = ["LSTM", "RandomForest", "GradientBoosting"]
    results = {}
    progress = st.progress(0, text="モデルを構築中...")
    for i, name in enumerate(models):
        progress.progress(i / len(models), text=f"{name} を訓練中...")
        if name == "LSTM":
            results[name] = predict_lstm(data, forecast_days, look_back)
        elif name == "RandomForest":
            results[name] = predict_rf(data, forecast_days, look_back)
        elif name == "GradientBoosting":
            results[name] = predict_gb(data, forecast_days, look_back)
    progress.progress(1.0, text="完了！")
    total_inv = sum(1.0 / r["rmse"] for r in results.values())
    weights = {k: (1.0 / v["rmse"]) / total_inv for k, v in results.items()}
    ensemble_preds = np.zeros(forecast_days)
    for name, result in results.items():
        ensemble_preds += weights[name] * result["future_preds"]
    all_preds = np.array([r["future_preds"] for r in results.values()])
    pred_std = np.std(all_preds, axis=0)
    avg_rmse = np.mean([r["rmse"] for r in results.values()])
    combined_unc = np.sqrt(pred_std**2 + avg_rmse**2)
    return {"ensemble_preds": ensemble_preds,
            "upper_95": ensemble_preds + 1.96 * combined_unc,
            "lower_95": ensemble_preds - 1.96 * combined_unc,
            "weights": weights, "individual_results": results, "pred_std": pred_std}

def plot_free_chart(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("株価チャート", "RSI", "MACD"))
    recent = df.tail(252)
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"], name="OHLC"), row=1, col=1)
    for col, color, name in [("SMA_20", "#FF6B6B", "SMA 20"),
                              ("SMA_50", "#4ECDC4", "SMA 50"),
                              ("SMA_200", "#45B7D1", "SMA 200")]:
        if col in recent.columns:
            fig.add_trace(go.Scatter(x=recent.index, y=recent[col],
                                     line=dict(color=color, width=1.5),
                                     name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["BB_upper"],
                              line=dict(color="rgba(150,150,150,0.3)"),
                              name="BB Upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["BB_lower"],
                              line=dict(color="rgba(150,150,150,0.3)"),
                              fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
                              name="BB Lower", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["RSI"],
                              line=dict(color="#9B59B6", width=1.5),
                              name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD"],
                              line=dict(color="#3498DB", width=1.5),
                              name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD_signal"],
                              line=dict(color="#E74C3C", width=1.5),
                              name="Signal"), row=3, col=1)
    colors = ["green" if v >= 0 else "red" for v in recent["MACD_hist"]]
    fig.add_trace(go.Bar(x=recent.index, y=recent["MACD_hist"],
                          marker_color=colors, name="Histogram",
                          opacity=0.5), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                      template="plotly_dark",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=60, r=30, t=40, b=30))
    return fig

def plot_prediction_chart(data, ens, forecast_days):
    fig = go.Figure()
    recent = data.tail(90)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"],
                              line=dict(color="#4ECDC4", width=2), name="実績値"))
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                   periods=forecast_days)
    clrs = {"LSTM": "#FF6B6B", "RandomForest": "#45B7D1", "GradientBoosting": "#FFA07A"}
    for name, result in ens["individual_results"].items():
        w = ens["weights"][name]
        fig.add_trace(go.Scatter(
            x=future_dates, y=result["future_preds"],
            line=dict(color=clrs.get(name, "gray"), width=1, dash="dot"),
            name=f"{name} ({w:.1%})", opacity=0.6))
    fig.add_trace(go.Scatter(x=future_dates, y=ens["ensemble_preds"],
                              line=dict(color="#FFD700", width=3),
                              name="🎯 アンサンブル予測"))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(ens["upper_95"]) + list(ens["lower_95"][::-1]),
        fill="toself", fillcolor="rgba(255,215,0,0.15)",
        line=dict(color="rgba(255,215,0,0)"), name="95% 信頼区間"))
    fig.add_trace(go.Scatter(
        x=[last_date, future_dates[0]],
        y=[data["Close"].iloc[-1], ens["ensemble_preds"][0]],
        line=dict(color="#FFD700", width=2, dash="dash"), showlegend=False))
    fig.update_layout(title="📈 AI 株価予測（アンサンブル）",
                      xaxis_title="日付", yaxis_title="株価",
                      height=600, template="plotly_dark",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=60, r=30, t=60, b=30))
    return fig

is_paid = check_payment_status()

with st.sidebar:
    st.title("📈 Stock Predictor Pro")
    st.divider()
    if is_paid:
        st.success("✅ Pro プラン有効")
        if st.session_state.customer_id:
            portal_url = create_portal_session(st.session_state.customer_id)
            if portal_url:
                st.link_button("⚙️ サブスク管理", portal_url)
    else:
        st.info("🆓 無料プラン")
    st.divider()
    st.subheader("銘柄設定")
    presets = {"S&P 500": "^GSPC", "日経平均": "^N225", "NASDAQ": "^IXIC",
               "Apple": "AAPL", "Tesla": "TSLA", "Microsoft": "MSFT",
               "Google": "GOOGL", "NVIDIA": "NVDA",
               "トヨタ (7203.T)": "7203.T", "ソニー (6758.T)": "6758.T",
               "カスタム入力": "custom"}
    selected = st.selectbox("銘柄を選択", list(presets.keys()))
    if presets[selected] == "custom":
        ticker = st.text_input("ティッカーシンボル", placeholder="例: AAPL, 7203.T")
    else:
        ticker = presets[selected]
    st.divider()
    if is_paid:
        st.subheader("予測設定")
        forecast_days = st.slider("予測日数", 5, 60, 30)
        look_back = st.slider("参照期間（日）", 30, 120, 60)
        models_to_use = st.multiselect(
            "使用モデル", ["LSTM", "RandomForest", "GradientBoosting"],
            default=["LSTM", "RandomForest", "GradientBoosting"])
    st.divider()
    st.caption("⚠️ 投資助言ではありません。投資判断はご自身の責任で。")

if not ticker:
    st.info("👈 サイドバーで銘柄を選択してください")
    st.stop()

with st.spinner("📡 データを取得中..."):
    data = load_data(ticker)
if data is None or data.empty:
    st.error(f"'{ticker}' のデータを取得できませんでした。")
    st.stop()

df = calc_technical_indicators(data)
c1, c2, c3, c4 = st.columns(4)
latest = df["Close"].iloc[-1]
prev = df["Close"].iloc[-2]
chg = latest - prev
chg_pct = (chg / prev) * 100
c1.metric("最新終値", f"{latest:,.2f}", f"{chg:+,.2f} ({chg_pct:+.2f}%)")
c2.metric("52週高値", f"{df['High'].tail(252).max():,.2f}")
c3.metric("52週安値", f"{df['Low'].tail(252).min():,.2f}")
c4.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")

st.subheader("📊 テクニカル分析")
st.plotly_chart(plot_free_chart(df), use_container_width=True)
st.divider()

st.subheader("🤖 AI 株価予測")
if not is_paid:
    st.markdown("---")
    st.markdown("### 🔒 Pro プランで AI 予測をアンロック")
    st.markdown("LSTM・RandomForest・GradientBoosting の3モデルによるアンサンブル予測")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("🧠 **LSTM**")
        st.caption("深層学習による時系列予測")
    with m2:
        st.markdown("🌲 **RandomForest**")
        st.caption("決定木アンサンブル予測")
    with m3:
        st.markdown("🚀 **GradientBoosting**")
        st.caption("勾配ブースティング予測")
    st.markdown("✅ 95%信頼区間　✅ モデル精度比較　✅ 最大60日先予測　✅ 全銘柄対応")
    st.markdown("---")
    _, cb, _ = st.columns([1, 2, 1])
    with cb:
        if st.button("🚀 Pro プランに登録する（月額 ¥380）", type="primary",
                      use_container_width=True):
            sess = create_checkout_session()
            if sess:
                st.link_button("💳 決済ページへ進む", sess.url,
                               use_container_width=True)
else:
    if st.button("🎯 AI 予測を実行", type="primary", use_container_width=True):
        if len(models_to_use) == 0:
            st.warning("サイドバーで1つ以上のモデルを選択してください。")
        else:
            with st.spinner("🧠 AI モデルを訓練中...（1〜3分）"):
                ens = predict_ensemble(data, forecast_days, look_back, models_to_use)
            st.success("✅ 予測完了！")
            last_p = data["Close"].iloc[-1]
            pred_p = ens["ensemble_preds"][-1]
            pred_chg = ((pred_p - last_p) / last_p) * 100
            r1, r2, r3 = st.columns(3)
            r1.metric(f"{forecast_days}日後の予測価格",
                      f"{pred_p:,.2f}", f"{pred_chg:+.2f}%")
            r2.metric("95%信頼区間 上限", f"{ens['upper_95'][-1]:,.2f}")
            r3.metric("95%信頼区間 下限", f"{ens['lower_95'][-1]:,.2f}")
            st.plotly_chart(plot_prediction_chart(data, ens, forecast_days),
                           use_container_width=True)
            st.subheader("📋 モデル別パフォーマンス")
            perf = []
            for name, r in ens["individual_results"].items():
                perf.append({"モデル": name, "RMSE": f"{r['rmse']:.2f}",
                             "MAPE": f"{r['mape']:.2f}%",
                             "重み": f"{ens['weights'][name]:.1%}",
                             f"{forecast_days}日後予測": f"{r['future_preds'][-1]:,.2f}"})
            st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)
            future_dates = pd.bdate_range(
                start=data.index[-1] + timedelta(days=1), periods=forecast_days)
            export = pd.DataFrame({"日付": future_dates,
                                    "アンサンブル予測": ens["ensemble_preds"],
                                    "95%上限": ens["upper_95"],
                                    "95%下限": ens["lower_95"]})
            for name, r in ens["individual_results"].items():
                export[f"{name}予測"] = r["future_preds"]
            st.download_button("📥 予測データをCSVでダウンロード",
                export.to_csv(index=False).encode("utf-8-sig"),
                f"prediction_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

st.divider()
st.caption("⚠️ **免責事項**: 本サービスは教育・情報提供目的のみです。"
           "投資助言ではありません。過去の実績は将来の結果を保証しません。"
           "投資判断はご自身の責任で行ってください。")
