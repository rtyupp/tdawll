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
matplotlib.use('Agg')  # إلزامي للسيرفرات بدون شاشة رسومية
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

# ===== إنشاء تطبيق Flask (مطلوب لأمر gunicorn app:app) =====
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

# ===== ذاكرة المحادثة (آخر 10 رسائل لكل شات لمنع النسيان) =====
chat_memory = {}

SYSTEM_PROMPT = """أنت خبير تداول مالي عالمي بمستوى مؤسساتي (Hedge Fund Level). 
معلوماتك تشمل: التحليل الفني المتقدم، إدارة المخاطر، السيولة، Order Flow، Price Action، 
الأنماط الكلاسيكية والحديثة، الاقتصاد الكلي، تأثير الأخبار على الأسواق، وعلم نفس التداول.

أسلوبك: دقيق، مختصر، احترافي، بالعربية الفصحى المبسطة. تستخدم المصطلحات الإنجليزية بين قوسين عند الحاجة.
تقدم تحليلاً تعليمياً وليس نصيحة مالية ملزمة. دائماً تذكر المستخدم بإدارة المخاطر."""


def tg_send_text(text, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


def tg_send_photo(photo_bytes, caption, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        files = {"photo": ("chart.png", photo_bytes, "image/png")}
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          files=files, data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}, timeout=30)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram photo error: {e}")
        return False


# ===== جلب البيانات الحية من Finnhub مع حماية ضد None =====
def fetch_bars_safe(symbol, resolution="15", count=200):
    try:
        to_ts = int(time.time())
        from_ts = to_ts - count * int(resolution) * 60
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={resolution}&from={from_ts}&to={to_ts}&token={FINNHUB_KEY}"
        j = requests.get(url, timeout=20).json()
        if j.get("s") != "ok" or not j.get("c"): return None
        
        df = pd.DataFrame({
            "Open": j["o"], "High": j["h"], "Low": j["l"],
            "Close": j["c"], "Volume": j["v"]
        }, index=pd.to_datetime(j["t"], unit="s"))
        return df
    except Exception as e:
        print(f"Fetch bars error: {e}")
        return None


# ===== حساب المؤشرات الفنية مع فحص مسبق =====
def compute_indicators(df):
    if df is None or len(df) < 20: return None
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


# ===== التنبؤ بالذكاء الاصطناعي المحلي (Random Forest) =====
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
def build_analysis_text(sym, df, pred, probs, conf):
    if df is None or len(df)==0: return "⚠️ لا توجد بيانات كافية."
    l = df.iloc[-1]; atr=l["ATR"]; pr=l["Close"]
    smap={1:"🟢 شراء",-1:"🔴 بيع",0:"⚪ انتظار"}
    trnd="صاعد 📈" if l["EMA_9"]>l["EMA_21"] else "هابط 📉"
    rz="تشبع شرائي ⚠️" if l["RSI"]>70 else "تشبع بيعي 💡" if l["RSI"]<30 else "محايد ➖"
    t=f"<b>🧠 تحليل {sym}</b>\n\n<b>السعر:</b> ${pr:.2f} | <b>الاتجاه:</b> {trnd}\n<b>الإشارة:</b> {smap.get(pred,'⚪')} | <b>الثقة:</b> {conf:.1f}%\n\n<b>📊 المؤشرات:</b>\n• RSI: {l['RSI']:.1f} ({rz})\n• MACD Hist: {l['MACD_Hist']:.4f}\n• ATR: {atr:.2f}\n\n<b>🎯 الخطة:</b>"
    if pred==1: t+=f"\n• Entry: ${pr:.2f}\n• SL: ${pr-1.5*atr:.2f}\n• TP: ${pr+2.5*atr:.2f}"
    elif pred==-1: t+=f"\n• Entry: ${pr:.2f}\n• SL: ${pr+1.5*atr:.2f}\n• TP: ${pr-2.5*atr:.2f}"
    else: t+="\n• انتظر تأكيدا فنيا."
    t+=f"\n\n<b>الاحتمالات:</b>\n• شراء:{probs.get(1,0)}% | انتظار:{probs.get(0,0)}% | بيع:{probs.get(-1,0)}%\n\n<i>⚠️ تحليل تعليمي - ليس نصيحة مالية.</i>"
    return t


# ===== رسم الشارت الاحترافي وإرساله كصورة =====
def generate_chart(sym, df):
    if df is None or len(df)<20: return None
    p=df.tail(60).copy()
    ap=[mpf.make_addplot(p["EMA_9"],color="#00ffff",width=1.2),
        mpf.make_addplot(p["EMA_21"],color="#ff8c00",width=1.2),
        mpf.make_addplot(p["BB_Upper"],color="#888",width=.6,linestyle="--"),
        mpf.make_addplot(p["BB_Lower"],color="#888",width=.6,linestyle="--")]
    st=mpf.make_mpf_style(base_mpf_style="charles",rc={"figure.facecolor":"#0a0a0a","axes.facecolor":"#141414"})
    buf=io.BytesIO()
    fig,_=mpf.plot(p,type="candle",style=st,addplot=ap,volume=True,figsize=(12,8),title=f"\n{sym}",returnfig=True)
    ax=fig.add_subplot(3,1,3)
    ax.plot(p.index,p["RSI"],color="#00ff88",lw=1.5)
    ax.axhline(70,color="#ff3366",ls="--",alpha=.6);ax.axhline(30,color="#00ff88",ls="--",alpha=.6)
    ax.set_ylabel("RSI",color="white");ax.tick_params(colors="white");ax.set_facecolor("#141414")
    plt.tight_layout();fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor="#0a0a0a")
    plt.close(fig);buf.seek(0);return buf.read()


# ===== الدالة الذكية التي تجرّب كل النماذج المتاحة تلقائياً =====
def ai_general_reply(user_text, chat_id):
    if not GEMINI_API_KEY:
        return "⚠️ مفتاح Gemini غير مضبوط في Render Environment Variables."
    
    history = chat_memory.get(chat_id, [])
    ctx = "\n".join([("User: "+m["content"]) if m["role"]=="user" else ("Model: "+m["content"]) for m in history[-6:]])
    prompt = f"{SYSTEM_PROMPT}\n\nPrevious:\n{ctx}\n\nQuestion: {user_text}\nAnswer professionally in Arabic."
    
    # قائمة النماذج المرشحة بالترتيب (الأحدث والأقوى أولاً)
    candidate_models = [
        "gemini-2.5-flash",       # الأحدث والأفضل حالياً
        "gemini-2.0-flash",       # مستقر وقوي
        "gemini-2.0-flash-exp",   # نسخة تجريبية قد تكون متاحة
        "gemini-1.5-flash",       # النسخة السابقة المستقرة
        "gemini-1.5-pro",         # الأقوى لكن أبطأ وأغلى
    ]
    
    last_error = ""
    for model_name in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents":[{"parts":[{"text":prompt}]}],
                       "generationConfig":{"temperature":0.7,"maxOutputTokens":800}}
            r = requests.post(url, json=payload, headers={"Content-Type":"application/json"}, timeout=30)
            
            if r.status_code == 200:
                data = r.json()
                cand = data.get("candidates",[])
                if cand and cand[0].get("content"):
                    parts = cand[0]["content"].get("parts",[])
                    reply = "".join([p.get("text","") for p in parts]).strip()
                    if reply:
                        # حفظ الذاكرة فقط عند النجاح
                        history.append({"role":"user","content":user_text})
                        history.append({"role":"assistant","content":reply})
                        chat_memory[chat_id] = history[-20:]
                        print(f"✅ Used model: {model_name}")
                        return reply
            
            err_body = r.text[:150]
            last_error = f"{model_name}: HTTP {r.status_code} - {err_body}"
            print(f"❌ Tried {model_name} -> {last_error}")
            
        except Exception as e:
            last_error = f"{model_name}: {str(e)[:80]}"
            continue
    
    return f"❌ لم ينجح أي نموذج Gemini.\nآخر خطأ: {last_error[:200]}\n\n💡 تحقق من الرابط التالي لمعرفة النماذج المتاحة لديك:\nhttps://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY"


# ===== معالجة الأوامر الذكية =====
def handle_command(text, chat_id):
    t=text.lower().strip()
    
    if t.startswith("/start") or t=="/help":
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
        "bot": "smart-model-v8-final",
        "symbol": SYMBOL,
        "ai_ready": bool(GEMINI_API_KEY),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ===== حلقة المراقبة التلقائية للخلفية =====
def monitor_loop():
    last_alert = 0
    time.sleep(8)
    tg_send_text(f"✅ <b>Expert Bot Online (Smart Model v8)</b>\n📡 يراقب {SYMBOL} كل {CHECK_INTERVAL//60} دقيقة\nاكتب /help للأوامر")
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
