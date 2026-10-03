# ============================================================
# SPY 15m AI Monitor - Alpaca Real-Time + GitHub Actions
# ============================================================

import os
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# ===== الإعدادات من GitHub Secrets =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ALPACA_KEY = os.environ.get("ALPACA_KEY_ID")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY")
SYMBOL = os.environ.get("SYMBOL", "SPY")
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "75"))

DATA_URL = "https://data.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET
}

# ===== 1) جلب شموع 15 دقيقة من Alpaca (لحظي حقيقي) =====
def fetch_bars(symbol, timeframe="15Min", limit=500):
    print(f"📥 Fetching {symbol} ({timeframe}) from Alpaca...")
    url = f"{DATA_URL}/v2/stocks/{symbol}/bars?timeframe={timeframe}&limit={limit}&adjustment=all"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code == 403:
        raise RuntimeError("403: فعّل حزمة Free IEX من لوحة Alpaca (Trading → Market Data)")
    if r.status_code == 401:
        raise RuntimeError("401: مفاتيح Alpaca خاطئة")
    r.raise_for_status()
    bars = r.json().get("bars", [])
    if not bars:
        raise ValueError("No bars returned")
    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "ts", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    df.set_index("ts", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    print(f"✅ Got {len(df)} candles (latest: {df.index[-1]})")
    return df

# ===== 2) المؤشرات الفنية =====
def add_indicators(df):
    df = df.copy()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    df["RSI_7"] = RSIIndicator(df["Close"], 7).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Hist"] = macd.macd_diff()
    df["EMA_9"] = EMAIndicator(df["Close"], 9).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    bb = BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    df["Stoch_K"] = (df["Close"] - low_min) / (high_max - low_min + 1e-9) * 100
    vol_sma = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / vol_sma
    df.dropna(inplace=True)
    return df

# ===== 3) كشف الإشارات الفنية =====
def detect_signals(df):
    if len(df) < 2:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    if last["Close"] > last["BB_Upper"] and prev["Close"] <= prev["BB_Upper"]:
        signals.append("🔥 كسر Bollinger Upper")
    if last["Close"] < last["BB_Lower"] and prev["Close"] >= prev["BB_Lower"]:
        signals.append("🔥 كسر Bollinger Lower")
    if last["EMA_9"] > last["EMA_20"] and prev["EMA_9"] <= prev["EMA_20"]:
        signals.append("⚡ تقاطع EMA صاعد")
    if last["EMA_9"] < last["EMA_20"] and prev["EMA_9"] >= prev["EMA_20"]:
        signals.append("⚡ تقاطع EMA هابط")
    if last["RSI"] < 30:
        signals.append("📉 RSI تشبع بيعي")
    elif last["RSI"] > 70:
        signals.append("📈 RSI تشبع شرائي")
    if last["Volume_Ratio"] > 2.5:
        signals.append(f"💥 حجم غير عادي ({last['Volume_Ratio']:.1f}x)")
    return signals

# ===== 4) نموذج AI =====
def train_and_predict(df):
    features = ["RSI", "RSI_7", "MACD", "MACD_Hist", "EMA_9", "EMA_20",
                "BB_Upper", "BB_Lower", "Stoch_K", "Volume_Ratio"]
    X = df[features].copy()
    for c in X.columns:
        X[c] = (X[c] - X[c].mean()) / (X[c].std() + 1e-9)

    future_ret = df["Close"].shift(-8) / df["Close"] - 1
    y = np.where(future_ret > 0.003, 1, np.where(future_ret < -0.003, -1, 0))

    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    last_X = X.iloc[-1:].values
    pred = int(model.predict(last_X)[0])
    proba = model.predict_proba(last_X)[0]
    classes = model.classes_
    proba_dict = {int(c): round(p * 100, 1) for c, p in zip(classes, proba)}
    confidence = max(proba) * 100
    return pred, proba_dict, confidence, acc

# ===== 5) إرسال تيليجرام =====
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram credentials")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

# ===== 6) التنفيذ الرئيسي =====
def main():
    print("=" * 50)
    print(f"🚀 {SYMBOL} 15m AI Monitor (Alpaca Real-Time)")
    print("=" * 50)

    if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ALPACA_KEY, ALPACA_SECRET]):
        print("❌ Missing one or more secrets. Check Settings → Secrets.")
        return

    try:
        df = fetch_bars(SYMBOL, "15Min", 500)
        df = add_indicators(df)
        if len(df) < 50:
            print(f"⚠️ Insufficient data: {len(df)} rows")
            return

        pred, proba, confidence, acc = train_and_predict(df)
        signals = detect_signals(df)
        signal_map = {1: "🟢 شراء", -1: "🔴 بيع", 0: "⚪ انتظار"}
        signal = signal_map.get(pred, "⚪ انتظار")
        last_price = float(df["Close"].iloc[-1])
        candle_time = df.index[-1].strftime("%Y-%m-%d %H:%M")

        print(f"📊 Signal: {signal} | Conf: {confidence:.1f}% | Acc: {acc*100:.1f}% | Signals: {len(signals)}")

        should_alert = (confidence >= MIN_CONFIDENCE and pred != 0) or len(signals) >= 2

        if should_alert:
            signals_text = "\n".join([f"• {s}" for s in signals]) if signals else "• لا توجد"
            urgency = "🚨" if confidence >= 85 or len(signals) >= 3 else "⚡"
            msg = f"""
{urgency} <b>{SYMBOL} 15m ALERT (LIVE)</b> {urgency}

<b>⏰ آخر شمعة:</b> {candle_time}
<b>💰 السعر:</b> ${last_price:.2f}
<b>🎯 الإشارة:</b> {signal}
<b>🧠 الثقة:</b> <b>{confidence:.1f}%</b>
<b>📊 دقة النموذج:</b> {acc*100:.1f}%

<b>━━━━━━━━━━━━━━━</b>
<b>🔥 الإشارات النشطة:</b>
{signals_text}

<b>━━━━━━━━━━━━━━━</b>
<b>📈 المؤشرات:</b>
• RSI(14): {df['RSI'].iloc[-1]:.2f}
• Stoch: {df['Stoch_K'].iloc[-1]:.1f}
• MACD Hist: {df['MACD_Hist'].iloc[-1]:.4f}
• Volume Ratio: {df['Volume_Ratio'].iloc[-1]:.1f}x

<b>━━━━━━━━━━━━━━━</b>
<b>📊 الاحتمالات:</b>
• شراء: {proba.get(1, 0)}%
• انتظار: {proba.get(0, 0)}%
• بيع: {proba.get(-1, 0)}%

<i>⚡ بيانات لحظية من Alpaca - فريم 15 دقيقة</i>
"""
            if send_telegram(msg):
                print("✅ Alert sent!")
            else:
                print("❌ Failed to send alert")
        else:
            print(f"⏸️ No alert (conf={confidence:.1f}%, signals={len(signals)})")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
