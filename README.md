# 📈 Stock Predictor Pro

LSTM / RandomForest / GradientBoosting アンサンブルによるAI株価予測アプリ。  
Stripe サブスクリプション（月額課金）で有料機能を提供。

## 機能

### 🆓 無料プラン
- ローソク足チャート
- テクニカル指標（SMA, ボリンジャーバンド, RSI, MACD）
- 主要銘柄プリセット（S&P500, 日経平均, NASDAQ, 個別株）

### 💎 Pro プラン（月額サブスクリプション）
- **LSTM**（深層学習）による時系列予測
- **RandomForest** アンサンブル予測
- **GradientBoosting** 予測
- 3モデルの **加重アンサンブル予測**（RMSE逆数重み付け）
- **95%信頼区間**
- モデル精度比較レポート
- 予測データCSVダウンロード
- 予測日数カスタマイズ（5〜60日）

---

## セットアップ手順

### 1. Stripe の設定

#### 1-1. サブスクリプション商品を作成

1. [Stripe Dashboard](https://dashboard.stripe.com/) にログイン
2. **商品カタログ** → **+ 商品を追加**
3. 以下を入力：
   - 商品名: `Stock Predictor Pro`
   - 料金: 月額 `¥980`（お好みの金額）
   - 請求間隔: **月次**
4. 保存後、作成された **Price ID** (`price_xxxx...`) をメモ

#### 1-2. カスタマーポータルを有効化

1. **設定** → **Billing** → **カスタマーポータル**
2. **「リンクを有効にする」** をON
3. 「サブスクリプションのキャンセル」を許可

#### 1-3. APIキーを確認

- **開発者** → **APIキー** で `sk_test_xxxx...` を確認
- 本番運用時は **テストモード** を解除して本番キーを使用

### 2. GitHub にコードをプッシュ

```bash
# Codespaces or ローカルで
git add app.py requirements.txt .gitignore
git commit -m "Complete stock predictor with LSTM ensemble + Stripe subscription"
git push origin main
```

### 3. Streamlit Community Cloud にデプロイ

1. [share.streamlit.io](https://share.streamlit.io/) にアクセス
2. **New app** → GitHub リポジトリ `inoku-c/stock-predict-app` を選択
3. Branch: `main`, Main file: `app.py`
4. **Advanced settings** → **Secrets** に以下を入力：

```toml
STRIPE_SECRET_KEY = "sk_test_あなたのキー"
STRIPE_PRICE_ID = "price_あなたのプライスID"
APP_URL = "https://your-app-name.streamlit.app"
```

5. **Deploy** をクリック

> ⚠️ デプロイ後にURLが確定したら、APP_URL を正しい値に更新してください。

### 4. Stripe Webhook（任意・推奨）

本番運用では Webhook でサブスク状態をリアルタイム管理することを推奨：

1. Stripe Dashboard → **開発者** → **Webhook**
2. エンドポイントURL: `https://your-app.streamlit.app`（Streamlitでは直接受けられないため、別途APIサーバーが必要）

> 現在の実装では、Checkout Session のリダイレクトパラメータで支払い確認を行っています。

---

## 技術スタック

| 項目 | 技術 |
|------|------|
| フロントエンド | Streamlit |
| 可視化 | Plotly |
| データ取得 | yfinance |
| 深層学習 | TensorFlow / Keras (LSTM) |
| 機械学習 | scikit-learn (RF, GB) |
| 決済 | Stripe (Subscriptions) |

---

## 免責事項

⚠️ 本サービスは教育・情報提供目的のみです。投資助言ではありません。  
過去の実績は将来の結果を保証しません。投資判断はご自身の責任で行ってください。
