import os
import io
import time
import threading
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib
matplotlib.use('Agg')  # مهم جداَ للسيرفرات بدون شاشة
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

# ===== إنشاء تطبيق Flask (هذا هو السطر الذي كان ناقصاً!) =====
app = Flask(__name__)

# ===== المتغيرات من Render Secrets =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SYMBOL = os.environ.get("SYMBOL", "SPY")
RESOLUTION = os.environ.get("RESOLUTION", "15")
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "70"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))

print(f"✅ Config loaded: SYMBOL={SYMBOL}, RESOLUTION={RESOLUTION}")
print(f"🔑 Keys present: Telegram={bool(TELEGRAM_TOKEN)}, Finnhub={bool(FINNHUB_KEY)}, Gemini={bool(GEMINI_API_KEY)}")

# ===== ذاكرة المحادثة =====
chat_memory = {}

SYSTEM_PROMPT = """أنت خبير تداول مالي عالمي بمستوى مؤسساتي (Hedge Fund Level). 
معلوماتك تشمل: التحليل الفني المتقدم، إدارة المخاطر، السيولة، Order Flow، Price Action، 
الأنماط الكلاسيكية والحديثة، الاقتصاد الكلي، تأثير الأخبار على الأسواق، وعلم نفس التداول.

أسلوبك: دقيق، مختصر، احترافي، بالعربية الفصحى المبسطة. تستخدم المصطلحات الإنجليزية بين قوسين عند الحاجة.
تقدم تحليلاً تعليمياً وليس نصيحة مالية ملزمة. دائماً تذكر المستخدم بإدارة المخاطر."""


def tg_send_text(text, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


def tg_send_photo(photo_bytes, caption, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        files = {"photo": ("chart.png", photo_bytes, "image/png")}
        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram photo error: {e}")
        return False


# ===== جلب البيانات الحية من Finnhub =====
def fetch_bars_safe(symbol, resolution="15", count=200):
    try:
        to_ts = int(time.time())
        from_ts = to_ts - count * int(resolution) * 60
        url = (f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}"
               f"&resolution={resolution}&from={from_ts}&to={to_ts}&token={FINNHUB_KEY}")
        r = requests.get(url, timeout=20)
        j = r.json()
        if j.get("s") != "ok" or not j.get("c"):
            return None
        
        df = pd.DataFrame({
            "Open": j["o"], "High": j["h"], "Low": j["l"],
            "Close": j["c"], "Volume": j["v"]
        }, index=pd.to_datetime(j["t"], unit="s"))
        return df
    except Exception as e:
        print(f"Fetch bars error: {e}")
        return None


# ===== حساب المؤشرات الفنية =====
def compute_indicators(df):
    if df is None or len(df) < 20:
        return None
    d = df.copy()
    d["RSI"] = RSIIndicator(d["Close"], 14).rsi()
    macd = MACD(d["Close"])
    d["MACD_Hist"] = macd.macd_diff()
    d["EMA_9"] = EMAIndicator(d["Close"], 9).ema_indicator()
    d["EMA_21"] = EMAIndicator(d["Close"], 21).ema_indicator()
    bb = BollingerBands(d["Close"])
    d["BB_Upper"] = bb.bollinger_hband()
    d["BB_Lower"] = bb.bollinger_lband()
    stoch = StochasticOscillator(d["High"], d["Low"], d["Close"])
    d["Stoch_K"] = stoch.stoch()
    d["ATR"] = AverageTrueRange(d["High"], d["Low"], d["Close"]).average_true_range()
    return d.dropna()


# ===== التنبؤ بالذكاء الاصطناعي المحلي =====
def predict(df):
    if df is None or len(df) < 30: return 0, {}, 0.0
    features = ["RSI", "MACD_Hist", "EMA_9", "EMA_21", "BB_Upper", "BB_Lower", "Stoch_K", "ATR"]
    X = df[features].copy()
    for c in X.columns:
        std_val = X[c].std()
        mean_val = X[c].mean()
        X[c] = (X[c] - mean_val) / (std_val + 1e-9)
    
    future_ret = df["Close"].shift(-8) / df["Close"] - 1
    y = np.where(future_ret > 0.003, 1, np.where(future_ret < -0.003, -1, 0))
    
    split = int(len(X) * 0.85)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    model = VotingClassifier(estimators=[("rf", rf)], voting="soft")
    model.fit(X.iloc[:split], y[:split])
    
    last_X = X.iloc[-1:].values
    pred = int(model.predict(last_X)[0])
    proba = model.predict_proba(last_X)[0]
    classes = model.classes_
    probs = {int(c): round(p*100, 1) for c, p in zip(classes, proba)}
    conf = max(proba) * 100
    return pred, probs, conf


# ===== توليد نص التحليل الاحترافي =====
def build_analysis_text(symbol, df, pred, probs, conf):
    if df is None or len(df)==0: return "⚠️ لا توجد بيانات كافية للتحليل."
    last = df.iloc[-1]
    signal_map = {1: "🟢 شراء (Long)", -1: "🔴 بيع (Short)", 0: "⚪ انتظار (Neutral)"}
    atr = last["ATR"]; price = last["Close"]
    sl_buy = price - 1.5 * atr; tp_buy = price + 2.5 * atr
    sl_sell = price + 1.5 * atr; tp_sell = price - 2.5 * atr
    
    trend = "صاعد 📈" if last["EMA_9"] > last["EMA_21"] else "هابط 📉"
    rsi_zone = "تشبع شرائي ⚠️" if last["RSI"] > 70 else "تشبع بيعي 💡" if last["RSI"] < 30 else "محايد ➖"
    
    txt = f"""<b>🧠 تحليل {symbol} الاحترافي</b>

<b>السعر:</b> ${price:.2f} | <b>الاتجاه:</b> {trend}
<b>الإشارة:</b> {signal_map.get(pred, '⚪')} | <b>الثقة:</b> {conf:.1f}%

<b>📊 المؤشرات الرئيسية:</b>
• RSI(14): {last['RSI']:.1f} ({rsi_zone})
• MACD Hist: {last['MACD_Hist']:.4f}
• ATR: {atr:.2f}

<b>🎯 خطة التداول المقترحة:</b>"""
    if pred == 1:
        txt += f"\n• Entry: ${price:.2f}\n• Stop Loss: ${sl_buy:.2f}\n• Take Profit: ${tp_buy:.2f}"
    elif pred == -1:
        txt += f"\n• Entry: ${price:.2f}\n• Stop Loss: ${sl_sell:.2f}\n• Take Profit: ${tp_sell:.2f}"
    else:
        txt += "\n• لا توجد إشارة واضحة. انتظر تأكيداَ فنياَ."
        
    txt += f"\n\n<b>الاحتمالات:</b>\n• شراء: {probs.get(1,0)}% | انتظار: {probs.get(0,0)}% | بيع: {probs.get(-1,0)}%"
    txt += "\n\n<i>⚠️ تحليل تعليمي - ليس نصيحة مالية.</i>"
    return txt


# ===== رسم الشارت الاحترافي =====
def generate_chart(symbol, df):
    if df is None or len(df) < 20: return None
    plot_df = df.tail(60).copy()
    ap = [
        mpf.make_addplot(plot_df["EMA_9"], color="#00ffff", width=1.2),
        mpf.make_addplot(plot_df["EMA_21"], color="#ff8c00", width=1.2),
        mpf.make_addplot(plot_df["BB_Upper"], color="#888888", width=0.6, linestyle="--"),
        mpf.make_addplot(plot_df["BB_Lower"], color="#888888", width=0.6, linestyle="--"),
    ]
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"figure.facecolor": "#0a0a0a", "axes.facecolor": "#141414"})
    buf = io.BytesIO()
    fig, axes = mpf.plot(plot_df, type="candle", style=style, addplot=ap, volume=True,
                         figsize=(12, 8), title=f"\n{symbol} - Real-Time Chart", returnfig=True)
    
    ax_rsi = fig.add_subplot(3, 1, 3)
    ax_rsi.plot(plot_df.index, plot_df["RSI"], color="#00ff88", linewidth=1.5)
    ax_rsi.axhline(70, color="#ff3366", linestyle="--", alpha=0.6)
    ax_rsi.axhline(30, color="#00ff88", linestyle="--", alpha=0.6)
    ax_rsi.set_ylabel("RSI", color="white"); ax_rsi.tick_params(colors="white"); ax_rsi.set_facecolor("#141414")
    
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig); buf.seek(0)
    return buf.read()


# ===== رد الذكاء الاصطناعي العام عبر Google Gemini =====
def ai_general_reply(user_text, chat_id):
    if not GEMINI_API_KEY:
        return "⚠️ محرك الذكاء الاصطناعي غير مهيأ. تأكد من وجود GEMINI_API_KEY في إعدادات Render."
    
    history = chat_memory.get(chat_id, [])
    
    context_parts = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Model"
        context_parts.append(f"{role}: {msg['content']}")
    
    full_prompt = f"""{SYSTEM_PROMPT}

Context from previous conversation:
{'\n'.join(context_parts)}

Current User Question: {user_text}

Please provide a professional trading analysis response in Arabic."""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 800,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if r.status_code != 200:
            error_detail = r.json().get("error", {}).get("message", str(r.text))[:100]
            return f"❌ خطأ من Gemini: {error_detail}"
            
        data = r.json()
        
        candidates = data.get("candidates", [])
        if not candidates or not candidates[0].get("content"):
            return "⚠️ لم يرجع Gemini رداً واضحاً. حاول مرة أخرى."
            
        parts = candidates[0]["content"].get("parts", [])
        reply = "".join([p.get("text", "") for p in parts]).strip()
        
        if not reply:
             return "⚠️ رد فارغ من الذكاء الاصطناعي."

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        chat_memory[chat_id] = history[-20:]
        
        return reply
        
    except Exception as e:
        return f"❌ فشل الاتصال بـ Gemini: {str(e)[:80]}"


# ===== معالجة الأوامر الذكية =====
def handle_command(text, chat_id):
    t = text.lower().strip()
    
    if t.startswith("/start") or t == "/help":
        return ("<b>👋 أهلاَ بك! أنا بوت التداول الخبير.</b>\n\n"
                "<b>الأوامر المتاحة:</b>\n"
                "• /analyze → تحليل فوري كامل مع خطة دخول ووقف\n"
                "• /chart → إرسال الشارت كصورة احترافية\n"
                "• /news → آخر أخبار الشركة/السوق (عبر Finnhub)\n"
                "• /rsi • /macd • /bb → قراءة مؤشر محدد\n"
                "• /price → السعر الحالي اللحظي\n"
                "• أي سؤال عادي → إجابة من خبير ذكي (Google Gemini AI)\n\n"
                "<i>اسألني عن أي شيء يتعلق بالتداول!</i>")
    
    raw_df = fetch_bars_safe(SYMBOL, RESOLUTION, 200)
    
    if t.startswith("/price"):
        if raw_df is None: return "⚠️ لا توجد بيانات (السوق قد يكون مغلقاَ)."
        return f"💰 <b>{SYMBOL}</b>: ${raw_df['Close'].iloc[-1]:.2f}"
    
    if t.startswith("/chart"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات لرسم الشارت."
        img = generate_chart(SYMBOL, ind_df)
        if img is None: return "⚠️ تعذر توليد الصورة."
        cap = f"📊 <b>{SYMBOL} Chart</b>\n${ind_df['Close'].iloc[-1]:.2f} | RSI {ind_df['RSI'].iloc[-1]:.1f}"
        tg_send_photo(img, cap, chat_id=chat_id)
        return None 
    
    if t.startswith("/analyze"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات للتحليل."
        pred, probs, conf = predict(ind_df)
        return build_analysis_text(SYMBOL, ind_df, pred, probs, conf)
    
    if t.startswith("/rsi"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات."
        v = ind_df["RSI"].iloc[-1]
        zone = "تشبع شرائي ⚠️" if v > 70 else "تشبع بيعي 💡" if v < 30 else "محايد ➖"
        return f"📉 RSI(14) = {v:.1f} → {zone}"
    
    if t.startswith("/macd"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات."
        h = ind_df["MACD_Hist"].iloc[-1]
        state = "إيجابي صاعد 🟢" if h > 0 else "سلبي هابط 🔴"
        return f"〽️ MACD Histogram = {h:.4f} → {state}"
    
    if t.startswith("/bb"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات."
        l = ind_df.iloc[-1]
        return (f"🎯 Bollinger Bands:\n• Upper: ${l['BB_Upper']:.2f}\n"
                f"• Mid: ${l['BB_Mid']:.2f}\n• Lower: ${l['BB_Lower']:.2f}")
    
    if t.startswith("/news"):
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            past_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            url = f"https://finnhub.io/api/v1/company-news?symbol={SYMBOL}&from={past_month}&to={today}&token={FINNHUB_KEY}"
            r = requests.get(url, timeout=10)
            news = r.json()
            if not isinstance(news, list) or not news:
                return "📰 لا توجد أخبار حديثة لهذا الرمز."
            top_news = news[:3]
            msg = f"<b>📰 آخر أخبار {SYMBOL}:</b>\n\n"
            for n in top_news:
                msg += f"• <a href='{n['url']}'>{n['headline']}</a>\n"
            return msg
        except:
            return "⚠️ تعذر جلب الأخبار حالياً."
    
    return ai_general_reply(text, chat_id)


# ===== Webhook لتيليجرام =====
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    msg = data.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    text = msg.get("text", "")
    if not chat_id or not text: return jsonify({"ok": True})
    try:
        reply = handle_command(text, chat_id)
        if reply: tg_send_text(reply, chat_id=chat_id)
    except Exception as e:
        print(f"Webhook Crash: {e}")
        tg_send_text(f"❌ حدث خطأ داخلي: {str(e)[:50]}", chat_id=chat_id)
    return jsonify({"ok": True})


# ===== Health Check =====
@app.route("/")
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "bot": "expert-trading-v5-final",
        "symbol": SYMBOL,
        "ai_ready": bool(GEMINI_API_KEY),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ===== حلقة المراقبة التلقائية للخلفية =====
def monitor_loop():
    last_alert = 0
    time.sleep(8)
    tg_send_text(f"✅ <b>Expert Bot Online (v5 Final)</b>\n📡 يراقب {SYMBOL} كل {CHECK_INTERVAL//60} دقيقة\nاكتب /help للأوامر")
    while True:
        try:
            raw_df = fetch_bars_safe(SYMBOL, RESOLUTION, 200)
            if raw_df is not None:
                ind_df = compute_indicators(raw_df)
                if ind_df is not None and len(ind_df) >= 30:
                    pred, probs, conf = predict(ind_df)
                    now = time.time()
                    if pred != 0 and conf >= MIN_CONFIDENCE and (now - last_alert) > 1800:
                        txt = build_analysis_text(SYMBOL, ind_df, pred, probs, conf)
                        if tg_send_text(txt): last_alert = now
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    threading.Thread(target=monitor_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
