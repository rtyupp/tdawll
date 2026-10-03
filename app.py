import os
import time
import threading
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ===== المتغيرات من Render =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")   # شاتك الشخصي للتنبيهات
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SYMBOL = os.environ.get("SYMBOL", "SPY")
RESOLUTION = os.environ.get("RESOLUTION", "15")
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "75"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))  # 5 دقائق

app = Flask(__name__)

# ===== إرسال تيليجرام =====
def tg_send(text, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ===== الذكاء الاصطناعي (Groq - مجاني وسريع) =====
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None

def ai_answer(user_text):
    if not groq_client:
        return "⚠️ الذكاء الاصطناعي غير مهيأ (GROQ_API_KEY مفقود في Render)."
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "أنت مساعد تداول مالي. أجب باختصار ودقة بالعربية. قدّم تحليلاً تعليمياً فقط، وليست نصيحة مالية ملزمة."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ خطأ: {e}"

# ===== تحليل السوق =====
def fetch_bars(symbol, resolution="15", count=300):
    to_ts = int(time.time())
    from_ts = to_ts - count * int(resolution) * 60
    url = (f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}"
           f"&resolution={resolution}&from={from_ts}&to={to_ts}&token={FINNHUB_KEY}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok" or not j.get("c"):
        return None
    return pd.DataFrame(
        {"Open": j["o"], "High": j["h"], "Low": j["l"], "Close": j["c"], "Volume": j["v"]},
        index=pd.to_datetime(j["t"], unit="s")
    )

def analyze():
    if not FINNHUB_KEY:
        return None
    try:
        df = fetch_bars(SYMBOL, RESOLUTION, 300)
        if df is None or len(df) < 50:
            return None
        df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
        df["MACD_Hist"] = MACD(df["Close"]).macd_diff()
        df["EMA_9"] = EMAIndicator(df["Close"], 9).ema_indicator()
        df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
        bb = BollingerBands(df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df.dropna(inplace=True)
        if len(df) < 50:
            return None
        features = ["RSI", "MACD_Hist", "EMA_9", "EMA_20", "BB_Upper", "BB_Lower"]
        X = df[features].copy()
        for c in X.columns:
            X[c] = (X[c] - X[c].mean()) / (X[c].std() + 1e-9)
        future_ret = df["Close"].shift(-8) / df["Close"] - 1
        y = np.where(future_ret > 0.003, 1, np.where(future_ret < -0.003, -1, 0))
        split = int(len(X) * 0.85)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
        model = VotingClassifier(estimators=[("rf", rf)], voting="soft")
        model.fit(X.iloc[:split], y[:split])
        last_X = X.iloc[-1:].values
        pred = int(model.predict(last_X)[0])
        proba = model.predict_proba(last_X)[0]
        confidence = max(proba) * 100
        signal_map = {1: "🟢 شراء", -1: "🔴 بيع", 0: "⚪ انتظار"}
        return {
            "signal": signal_map.get(pred, "⚪ انتظار"),
            "pred": pred,
            "confidence": confidence,
            "price": float(df["Close"].iloc[-1]),
            "rsi": float(df["RSI"].iloc[-1]),
            "time": df.index[-1].strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        print(f"Analyze error: {e}")
        return None

# ===== Thread المراقبة الخلفي =====
def monitor_loop():
    last_alert = 0
    # رسالة بداية عند التشغيل
    time.sleep(5)
    tg_send(f"✅ <b>Render Monitor started</b>\n📡 {SYMBOL} {RESOLUTION}m | فحص كل {CHECK_INTERVAL//60} دقيقة")
    while True:
        try:
            result = analyze()
            now = time.time()
            if result and result["pred"] != 0 and result["confidence"] >= MIN_CONFIDENCE:
                if now - last_alert > 1800:  # منع التكرار خلال 30 دقيقة
                    msg = (f"🚨 <b>{SYMBOL} {RESOLUTION}m ALERT</b>\n\n"
                           f"⏰ {result['time']}\n"
                           f"💰 ${result['price']:.2f}\n"
                           f"🎯 {result['signal']}\n"
                           f"🧠 الثقة: {result['confidence']:.1f}%\n"
                           f"📉 RSI: {result['rsi']:.2f}\n\n"
                           f"<i>Render 24/7 • Finnhub Live</i>")
                    if tg_send(msg):
                        last_alert = now
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(CHECK_INTERVAL)

# ===== المسارات =====
@app.route("/health")
def health():
    return jsonify({"status": "ok", "symbol": SYMBOL, "ai": bool(groq_client), "time": time.strftime("%H:%M:%S")})

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    msg = data.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    text = msg.get("text", "")
    if not chat_id or not text:
        return jsonify({"ok": True})
    if text.startswith("/analyze"):
        result = analyze()
        reply = (f"📊 {SYMBOL}: {result['signal']} | ثقة {result['confidence']:.1f}% | ${result['price']:.2f}"
                 if result else "⚠️ لا توجد بيانات (السوق مغلق؟)")
    elif text.startswith("/start"):
        reply = "✅ البوت جاهز! اسألني عن التداول، أو اكتب /analyze للتحليل الفوري."
    elif text.startswith("/help"):
        reply = "الأوامر:\n/analyze → تحليل فوري\n/start → تهيئة\nأي سؤال → ذكاء اصطناعي"
    else:
        reply = ai_answer(text)
    tg_send(reply, chat_id=chat_id)
    return jsonify({"ok": True})

if __name__ == "__main__":
    threading.Thread(target=monitor_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
