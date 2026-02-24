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

# ============================================================
# 0. ページ設定
# ============================================================
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 1. Stripe設定
# ============================================================
stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY") or os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = st.secrets.get("STRIPE_PRICE_ID") or os.getenv("STRIPE_PRICE_ID")
APP_URL = st.secrets.get("APP_URL") or os.getenv("APP_URL", "http://localhost:8501")

# ============================================================
# 2. セッション状態の初期化
# ============================================================
if "paid" not in st.session_state:
    st.session_state.paid = False
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "subscription_id" not in st.session_state:
    st.session_state.subscription_id = None

# ============================================================
# 3. Stripe ヘルパー関数
# ============================================================
def create_checkout_session():
    """Stripeチェックアウトセッションを作成"""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": STRIPE_PRICE_ID,
                "quantity": 1,
            }],
            mode="subscription",
            success_url=f"{APP_URL}?session_id={{CHECKOUT_SESSION_ID}}&status=success",
            cancel_url=f"{APP_URL}?status=cancel",
        )
        return session
    except Exception as e:
        st.error(f"決済セッション作成エラー: {e}")
        return None

def check_payment_status():
    """URLパラメータから支払い状態を確認"""
    params = st.query_params
    if params.get("status") == "success" and params.get("session_id"):
        try:
            session = stripe.checkout.Session.retrieve(params["session_id"])
            if session.payment_status == "paid" or session.status == "complete":
                st.session_state.paid = True
                st.session_state.customer_id = session.customer
                st.session_state.subscription_id = session.subscription
                return True
        except Exception as e:
            st.error(f"支払い確認エラー: {e}")
    return st.session_state.paid

def create_portal_session(customer_id):
    """Stripeカスタマーポータルセッションを作成"""
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=APP_URL,
        )
        return session.url
    except Exception as e:
        st.error(f"ポータル作成エラー: {e}")
        return None

# ============================================================
# 4. データ取得
# ============================================================
@st.cache_data(ttl=3600)
def load_data(ticker, start_date="2015-01-01"):
    """株価データを取得"""
    try:
        data = yf.download(ticker, start=start_date, auto_adjust=True)
        if data.empty:
            return None
        # MultiIndex対策
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None

# ============================================================
# 5. テクニカル指標計算（無料機能）
# ============================================================
def calc_technical_indicators(data):
    """テクニカル指標を計算"""
    df = data.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    # ボリンジャーバンド
    df["BB_mid"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

# ============================================================
# 6. LSTM予測モデル（有料機能）
# ============================================================
def prepare_lstm_data(data, look_back=60):
    """LSTM用データを準備"""
    from sklearn.preprocessing import MinMaxScaler
    
    close_prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def build_and_train_lstm(X_train, y_train, look_back=60):
    """LSTMモデルを構築・訓練"""
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
    
    early_stop = EarlyStopping(
        monitor='loss', patience=5, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model

def predict_lstm(data, forecast_days=30, look_back=60):
    """LSTMで将来の株価を予測"""
    X, y, scaler = prepare_lstm_data(data, look_back)
    
    # 訓練データ（最後の20%をテスト用に）
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_and_train_lstm(X_train, y_train, look_back)
    
    # テストデータでの予測
    test_pred = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 将来予測
    close_prices = data["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)
    last_seq = scaled[-look_back:]
    
    future_preds = []
    current_seq = last_seq.copy()
    
    for _ in range(forecast_days):
        pred = model.predict(current_seq.reshape(1, look_back, 1), verbose=0)
        future_preds.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred.reshape(1, 1), axis=0)
    
    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()
    
    # 信頼区間（テストデータのRMSEベース）
    rmse = np.sqrt(np.mean((test_pred.flatten() - y_test_actual.flatten()) ** 2))
    
    # MAPE計算
    mape = np.mean(np.abs(
        (y_test_actual.flatten() - test_pred.flatten()) / y_test_actual.flatten()
    )) * 100
    
    return {
        "future_preds": future_preds,
        "test_pred": test_pred.flatten(),
        "test_actual": y_test_actual.flatten(),
        "rmse": rmse,
        "mape": mape,
        "test_dates": data.index[split + look_back:],
        "model_name": "LSTM"
    }

# ============================================================
# 7. RandomForest予測モデル（有料機能）
# ============================================================
def predict_rf(data, forecast_days=30, look_back=60):
    """RandomForestで将来の株価を予測"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    
    close_prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    test_pred = scaler.inverse_transform(
        model.predict(X_test).reshape(-1, 1)
    ).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 将来予測
    last_seq = scaled[-look_back:, 0].copy()
    future_preds = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, -1))
        future_preds.append(pred[0])
        last_seq = np.append(last_seq[1:], pred)
    
    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()
    
    rmse = np.sqrt(np.mean((test_pred - y_test_actual) ** 2))
    mape = np.mean(np.abs(
        (y_test_actual - test_pred) / y_test_actual
    )) * 100
    
    return {
        "future_preds": future_preds,
        "test_pred": test_pred,
        "test_actual": y_test_actual,
        "rmse": rmse,
        "mape": mape,
        "test_dates": data.index[split + look_back:],
        "model_name": "RandomForest"
    }

# ============================================================
# 8. GradientBoosting予測モデル（有料機能）
# ============================================================
def predict_gb(data, forecast_days=30, look_back=60):
    """GradientBoostingで将来の株価を予測"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler
    
    close_prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    
    test_pred = scaler.inverse_transform(
        model.predict(X_test).reshape(-1, 1)
    ).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 将来予測
    last_seq = scaled[-look_back:, 0].copy()
    future_preds = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, -1))
        future_preds.append(pred[0])
        last_seq = np.append(last_seq[1:], pred)
    
    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()
    
    rmse = np.sqrt(np.mean((test_pred - y_test_actual) ** 2))
    mape = np.mean(np.abs(
        (y_test_actual - test_pred) / y_test_actual
    )) * 100
    
    return {
        "future_preds": future_preds,
        "test_pred": test_pred,
        "test_actual": y_test_actual,
        "rmse": rmse,
        "mape": mape,
        "test_dates": data.index[split + look_back:],
        "model_name": "GradientBoosting"
    }

# ============================================================
# 9. アンサンブル予測（有料機能）
# ============================================================
def predict_ensemble(data, forecast_days=30, look_back=60, models=None):
    """アンサンブル（加重平均）で将来の株価を予測"""
    if models is None:
        models = ["LSTM", "RandomForest", "GradientBoosting"]
    
    results = {}
    progress = st.progress(0, text="モデルを構築中...")
    
    for i, model_name in enumerate(models):
        progress.progress(
            (i) / len(models),
            text=f"{model_name} を訓練中..."
        )
        if model_name == "LSTM":
            results[model_name] = predict_lstm(data, forecast_days, look_back)
        elif model_name == "RandomForest":
            results[model_name] = predict_rf(data, forecast_days, look_back)
        elif model_name == "GradientBoosting":
            results[model_name] = predict_gb(data, forecast_days, look_back)
    
    progress.progress(1.0, text="完了！")
    
    # RMSE逆数で重み付け
    total_inv_rmse = sum(1.0 / r["rmse"] for r in results.values())
    weights = {k: (1.0 / v["rmse"]) / total_inv_rmse for k, v in results.items()}
    
    # アンサンブル予測
    ensemble_preds = np.zeros(forecast_days)
    for model_name, result in results.items():
        ensemble_preds += weights[model_name] * result["future_preds"]
    
    # 信頼区間
    all_preds = np.array([r["future_preds"] for r in results.values()])
    pred_std = np.std(all_preds, axis=0)
    avg_rmse = np.mean([r["rmse"] for r in results.values()])
    combined_uncertainty = np.sqrt(pred_std**2 + avg_rmse**2)
    
    upper_95 = ensemble_preds + 1.96 * combined_uncertainty
    lower_95 = ensemble_preds - 1.96 * combined_uncertainty
    
    return {
        "ensemble_preds": ensemble_preds,
        "upper_95": upper_95,
        "lower_95": lower_95,
        "weights": weights,
        "individual_results": results,
        "pred_std": pred_std,
    }

# ============================================================
# 10. 可視化
# ============================================================
def plot_free_chart(df):
    """無料版チャート：ローソク足 + 移動平均"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("株価チャート", "RSI", "MACD")
    )
    
    # 直近1年分
    recent = df.tail(252)
    
    # ローソク足
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"], name="OHLC"
    ), row=1, col=1)
    
    # 移動平均
    for col, color, name in [
        ("SMA_20", "#FF6B6B", "SMA 20"),
        ("SMA_50", "#4ECDC4", "SMA 50"),
        ("SMA_200", "#45B7D1", "SMA 200"),
    ]:
        if col in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent.index, y=recent[col],
                line=dict(color=color, width=1.5),
                name=name
            ), row=1, col=1)
    
    # ボリンジャーバンド
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["BB_upper"],
        line=dict(color="rgba(150,150,150,0.3)"), name="BB Upper",
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["BB_lower"],
        line=dict(color="rgba(150,150,150,0.3)"), name="BB Lower",
        fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
        showlegend=False
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["RSI"],
        line=dict(color="#9B59B6", width=1.5), name="RSI"
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",
                  opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  opacity=0.5, row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["MACD"],
        line=dict(color="#3498DB", width=1.5), name="MACD"
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["MACD_signal"],
        line=dict(color="#E74C3C", width=1.5), name="Signal"
    ), row=3, col=1)
    colors = ["green" if v >= 0 else "red" for v in recent["MACD_hist"]]
    fig.add_trace(go.Bar(
        x=recent.index, y=recent["MACD_hist"],
        marker_color=colors, name="Histogram", opacity=0.5
    ), row=3, col=1)
    
    fig.update_layout(
        height=800, xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=40, b=30),
    )
    
    return fig

def plot_prediction_chart(data, ensemble_result, forecast_days):
    """有料版チャート：予測結果の可視化"""
    fig = go.Figure()
    
    # 過去90日の実績
    recent = data.tail(90)
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["Close"],
        line=dict(color="#4ECDC4", width=2),
        name="実績値"
    ))
    
    # 将来の日付生成
    last_date = data.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=forecast_days
    )
    
    # 個別モデル予測
    colors = {"LSTM": "#FF6B6B", "RandomForest": "#45B7D1",
              "GradientBoosting": "#FFA07A"}
    for name, result in ensemble_result["individual_results"].items():
        weight = ensemble_result["weights"][name]
        fig.add_trace(go.Scatter(
            x=future_dates, y=result["future_preds"],
            line=dict(color=colors.get(name, "gray"), width=1, dash="dot"),
            name=f"{name} (重み: {weight:.1%})",
            opacity=0.6
        ))
    
    # アンサンブル予測
    fig.add_trace(go.Scatter(
        x=future_dates, y=ensemble_result["ensemble_preds"],
        line=dict(color="#FFD700", width=3),
        name="🎯 アンサンブル予測"
    ))
    
    # 95%信頼区間
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(ensemble_result["upper_95"]) + list(ensemble_result["lower_95"][::-1]),
        fill="toself", fillcolor="rgba(255,215,0,0.15)",
        line=dict(color="rgba(255,215,0,0)"),
        name="95% 信頼区間"
    ))
    
    # 実績→予測の接続線
    fig.add_trace(go.Scatter(
        x=[last_date, future_dates[0]],
        y=[data["Close"].iloc[-1], ensemble_result["ensemble_preds"][0]],
        line=dict(color="#FFD700", width=2, dash="dash"),
        showlegend=False
    ))
    
    fig.update_layout(
        title="📈 AI 株価予測（アンサンブル）",
        xaxis_title="日付", yaxis_title="株価",
        height=600, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=60, b=30),
    )
    
    return fig

# ============================================================
# 11. UI構成
# ============================================================

# --- 支払い状態チェック ---
is_paid = check_payment_status()

# --- サイドバー ---
with st.sidebar:
    st.title("📈 Stock Predictor Pro")
    st.divider()
    
    # プラン表示
    if is_paid:
        st.success("✅ Pro プラン有効")
        if st.session_state.customer_id:
            portal_url = create_portal_session(st.session_state.customer_id)
            if portal_url:
                st.link_button("⚙️ サブスク管理", portal_url)
    else:
        st.info("🆓 無料プラン")
    
    st.divider()
    
    # 銘柄選択
    st.subheader("銘柄設定")
    ticker_presets = {
        "S&P 500": "^GSPC",
        "日経平均": "^N225",
        "NASDAQ": "^IXIC",
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "NVIDIA": "NVDA",
        "トヨタ (7203.T)": "7203.T",
        "ソニー (6758.T)": "6758.T",
        "カスタム入力": "custom",
    }
    
    selected = st.selectbox("銘柄を選択", list(ticker_presets.keys()))
    
    if ticker_presets[selected] == "custom":
        ticker = st.text_input(
            "ティッカーシンボルを入力",
            placeholder="例: AAPL, 7203.T"
        )
    else:
        ticker = ticker_presets[selected]
    
    st.divider()
    
    # 予測設定（有料のみ）
    if is_paid:
        st.subheader("予測設定")
        forecast_days = st.slider("予測日数", 5, 60, 30)
        look_back = st.slider("参照期間（日）", 30, 120, 60)
        models_to_use = st.multiselect(
            "使用モデル",
            ["LSTM", "RandomForest", "GradientBoosting"],
            default=["LSTM", "RandomForest", "GradientBoosting"]
        )
    
    st.divider()
    st.caption("⚠️ 本サービスは投資助言ではありません。投資判断はご自身の責任で行ってください。")

# --- メインコンテンツ ---
if not ticker:
    st.info("👈 サイドバーで銘柄を選択してください")
    st.stop()

# データ取得
with st.spinner("📡 データを取得中..."):
    data = load_data(ticker)

if data is None or data.empty:
    st.error(f"'{ticker}' のデータを取得できませんでした。ティッカーを確認してください。")
    st.stop()

# テクニカル指標計算
df = calc_technical_indicators(data)

# ヘッダー情報
col1, col2, col3, col4 = st.columns(4)
latest_close = df["Close"].iloc[-1]
prev_close = df["Close"].iloc[-2]
change = latest_close - prev_close
change_pct = (change / prev_close) * 100

with col1:
    st.metric("最新終値", f"{latest_close:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
with col2:
    st.metric("52週高値", f"{df['High'].tail(252).max():,.2f}")
with col3:
    st.metric("52週安値", f"{df['Low'].tail(252).min():,.2f}")
with col4:
    st.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")

# === 無料機能：テクニカルチャート ===
st.subheader("📊 テクニカル分析")
fig_free = plot_free_chart(df)
st.plotly_chart(fig_free, use_container_width=True)

st.divider()

# === 有料機能：AI予測 ===
st.subheader("🤖 AI 株価予測")

if not is_paid:
    # --- 無料ユーザー向けのアップグレードUI ---
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 2px solid #e94560;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    ">
        <h2 style="color: #e94560; margin-bottom: 10px;">🔒 Pro プランで AI 予測をアンロック</h2>
        <p style="color: #ccc; font-size: 18px; margin-bottom: 30px;">
            LSTM・RandomForest・GradientBoosting の3モデルによるアンサンブル予測
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 30px;">
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; width: 200px;">
                <div style="font-size: 28px;">🧠</div>
                <div style="color: #fff; font-weight: bold;">LSTM</div>
                <div style="color: #aaa; font-size: 13px;">深層学習による時系列予測</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; width: 200px;">
                <div style="font-size: 28px;">🌲</div>
                <div style="color: #fff; font-weight: bold;">RandomForest</div>
                <div style="color: #aaa; font-size: 13px;">決定木アンサンブル予測</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; width: 200px;">
                <div style="font-size: 28px;">🚀</div>
                <div style="color: #fff; font-weight: bold;">GradientBoosting</div>
                <div style="color: #aaa; font-size: 13px;">勾配ブースティング予測</div>
            </div>
        </div>
        <p style="color: #aaa;">✅ 95%信頼区間 ✅ モデル精度比較 ✅ 最大60日先予測 ✅ 全銘柄対応</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Pro プランに登録する（月額）", type="primary", use_container_width=True):
            session = create_checkout_session()
            if session:
                st.link_button(
                    "💳 決済ページへ進む",
                    session.url,
                    use_container_width=True
                )

else:
    # --- 有料ユーザー向け：AI予測実行 ---
    if st.button("🎯 AI 予測を実行", type="primary", use_container_width=True):
        if len(models_to_use) == 0:
            st.warning("サイドバーで1つ以上のモデルを選択してください。")
        else:
            with st.spinner("🧠 AI モデルを訓練中... （1〜3分かかります）"):
                ensemble_result = predict_ensemble(
                    data, forecast_days, look_back, models_to_use
                )
            
            st.success("✅ 予測完了！")
            
            # 予測結果のサマリー
            last_price = data["Close"].iloc[-1]
            pred_price = ensemble_result["ensemble_preds"][-1]
            pred_change = ((pred_price - last_price) / last_price) * 100
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(
                    f"{forecast_days}日後の予測価格",
                    f"{pred_price:,.2f}",
                    f"{pred_change:+.2f}%"
                )
            with col_r2:
                st.metric(
                    "95%信頼区間 上限",
                    f"{ensemble_result['upper_95'][-1]:,.2f}"
                )
            with col_r3:
                st.metric(
                    "95%信頼区間 下限",
                    f"{ensemble_result['lower_95'][-1]:,.2f}"
                )
            
            # 予測チャート
            fig_pred = plot_prediction_chart(data, ensemble_result, forecast_days)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # モデル詳細
            st.subheader("📋 モデル別パフォーマンス")
            perf_data = []
            for name, result in ensemble_result["individual_results"].items():
                perf_data.append({
                    "モデル": name,
                    "RMSE": f"{result['rmse']:.2f}",
                    "MAPE": f"{result['mape']:.2f}%",
                    "重み": f"{ensemble_result['weights'][name]:.1%}",
                    f"{forecast_days}日後予測": f"{result['future_preds'][-1]:,.2f}",
                })
            
            st.dataframe(
                pd.DataFrame(perf_data),
                use_container_width=True,
                hide_index=True
            )
            
            # 予測データのダウンロード
            future_dates = pd.bdate_range(
                start=data.index[-1] + timedelta(days=1),
                periods=forecast_days
            )
            export_df = pd.DataFrame({
                "日付": future_dates,
                "アンサンブル予測": ensemble_result["ensemble_preds"],
                "95%上限": ensemble_result["upper_95"],
                "95%下限": ensemble_result["lower_95"],
            })
            for name, result in ensemble_result["individual_results"].items():
                export_df[f"{name}予測"] = result["future_preds"]
            
            csv = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 予測データをCSVでダウンロード",
                csv,
                f"prediction_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

# --- フッター ---
st.divider()
st.caption(
    "⚠️ **免責事項**: 本サービスは教育・情報提供目的のみです。"
    "投資助言ではありません。過去の実績は将来の結果を保証しません。"
    "投資判断はご自身の責任で行ってください。"
)