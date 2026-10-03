import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# ===== الإعدادات من GitHub Secrets =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "75"))

# ===== جلب بيانات 15 دقيقة =====
def fetch_data(symbol="SPY", period="60d"):
    df = yf.download(symbol, period=period, interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ===== المؤشرات الفنية =====
def add_indicators(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(df['Close'], 14).rsi()
    df['RSI_7'] = RSIIndicator(df['Close'], 7).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Hist'] = macd.macd_diff()
    df['EMA_9'] = EMAIndicator(df['Close'], 9).ema_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], 20).ema_indicator()
    bb = BollingerBands(df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    stoch_k = (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-9) * 100
    df['Stoch_K'] = stoch_k
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    df.dropna(inplace=True)
    return df

# ===== كشف الإشارات القوية =====
def detect_signals(df):
    if len(df) < 2:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    
    if last['Close'] > last['BB_Upper'] and prev['Close'] <= prev['BB_Upper']:
        signals.append(" كسر BB Upper")
    if last['Close'] < last['BB_Lower'] and prev['Close'] >= prev['BB_Lower']:
        signals.append("🔥 كسر BB Lower")
    if last['EMA_9'] > last['EMA_20'] and prev['EMA_9'] <= prev['EMA_20']:
        signals.append(" تقاطع EMA صاعد")
    if last['EMA_9'] < last['EMA_20'] and prev['EMA_9'] >= prev['EMA_20']:
        signals.append("⚡ تقاطع EMA هابط")
    if last['RSI'] < 30:
        signals.append("📉 RSI تشبع بيعي")
    elif last['RSI'] > 70:
        signals.append("📈 RSI تشبع شرائي")
    if last['Volume_Ratio'] > 2.5:
        signals.append(f"💥 حجم غير عادي ({last['Volume_Ratio']:.1f}x)")
    
    return signals

# ===== نموذج AI =====
def train_and_predict(df):
    features = ['RSI', 'RSI_7', 'MACD', 'MACD_Hist', 'EMA_9', 'EMA_20',
                'BB_Upper', 'BB_Lower', 'Stoch_K', 'Volume_Ratio']
    X = df[features].copy()
    for c in X.columns:
        X[c] = (X[c] - X[c].mean()) / (X[c].std() + 1e-9)
    
    future_ret = df['Close'].shift(-8) / df['Close'] - 1
    y = np.where(future_ret > 0.003, 1, np.where(future_ret < -0.003, -1, 0))
    
    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    last_X = X.iloc[-1:].values
    pred = int(model.predict(last_X)[0])
    proba = model.predict_proba(last_X)[0]
    classes = model.classes_
    proba_dict = {int(c): round(p*100, 1) for c, p in zip(classes, proba)}
    confidence = max(proba) * 100
    
    return pred, proba_dict, confidence, acc

# ===== إرسال لتيليجرام =====
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ لم يتم ضبط TELEGRAM_TOKEN أو TELEGRAM_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    r = requests.post(url, data=data, timeout=15)
    return r.status_code == 200

# ===== التنفيذ الرئيسي =====
def main():
    print("🚀 SPY 15m Monitor - GitHub Actions")
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing Telegram credentials")
        return
    
    df = fetch_data("SPY", "60d")
    df = add_indicators(df)
    
    if len(df) < 50:
        print("️ بيانات غير كافية")
        return
    
    pred, proba, confidence, acc = train_and_predict(df)
    signals = detect_signals(df)
    
    signal_map = {1: "🟢 شراء", -1: "🔴 بيع", 0: "⚪ انتظار"}
    signal = signal_map.get(pred, "⚪ انتظار")
    
    print(f"📊 Signal: {signal} | Confidence: {confidence:.1f}% | Signals: {len(signals)}")
    
    # إرسال إذا: ثقة عالية أو إشارات قوية
    should_alert = (confidence >= MIN_CONFIDENCE and pred != 0) or len(signals) >= 2
    
    if should_alert:
        last_price = df['Close'].iloc[-1]
        signals_text = "\n".join([f"• {s}" for s in signals]) if signals else "• لا توجد"
        urgency = "🚨" if confidence >= 85 or len(signals) >= 3 else "⚡"
        
        msg = f"""
{urgency} <b>SPY 15m ALERT</b> {urgency}

<b>⏰ الوقت:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
<b>💰 السعر:</b> ${last_price:.2f}
<b> الإشارة:</b> {signal}
<b>🧠 الثقة:</b> <b>{confidence:.1f}%</b>
<b>📊 دقة النموذج:</b> {acc*100:.1f}%

<b>━━━━━━━━━━━━━━━</b>
<b>🔥 الإشارات:</b>
{signals_text}

<b>━━━━━━━━━━━━━━━</b>
<b>📈 المؤشرات:</b>
• RSI(14): {df['RSI'].iloc[-1]:.2f}
• Stoch: {df['Stoch_K'].iloc[-1]:.1f}
• MACD: {df['MACD_Hist'].iloc[-1]:.4f}
• Volume: {df['Volume_Ratio'].iloc[-1]:.1f}x

<b>━━━━━━━━━━━━━━━</b>
<b>📊 الاحتمالات:</b>
• شراء: {proba.get(1, 0)}%
• انتظار: {proba.get(0, 0)}%
• بيع: {proba.get(-1, 0)}%

<i>⚡ فريم 15 دقيقة - قرار سريع!</i>
"""
        if send_telegram(msg):
            print("✅ Alert sent!")
        else:
            print("❌ Failed to send")
    else:
        print(f"⏸️ No alert (conf={confidence:.1f}%, signals={len(signals)})")

if __name__ == "__main__":
    main()
