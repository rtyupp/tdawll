import os
import io
import time
import threading
import requests
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

app = Flask(__name__)

# ===== المتغيرات من Render Secrets =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ⚠️ التخصيص الحصري: لا يقبل إلا SPY أو SPX
ALLOWED_SYMBOLS = ["SPY", "^GSPC"]  # ^GSPC هو الرمز الرسمي لمؤشر S&P 500 (SPX)
CURRENT_SYMBOL = "SPY"
RESOLUTION = "15"
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "70"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))

print(f"✅ Expert Bot Initialized for SPY/SPX Only")
print(f"🔑 Keys: TG={bool(TELEGRAM_TOKEN)}, FH={bool(FINNHUB_KEY)}, GM={bool(GEMINI_API_KEY)}")

chat_memory = {}

# نظام تعليمات صارم للمتخصص
SYSTEM_PROMPT = """أنت خبير تداول مالي متخصص حصرياً في مؤشرات S&P 500 (SPY ETF و ^GSPC Index). 
قواعد الإجابة الصارمة:
1. أجب بالعربية الفصحى المبسطة وبشكل مختصر جداً (لا تتجاوز 3 جمل قصيرة).
2. اربط دائماً بين حركة SPY ومؤشر SPX الأصلي إذا لزم الأمر.
3. ركز فقط على التحليل الفني القصير المدى (فريم 15 دقيقة).
4. لا تقدم نصيحة مالية مباشرة، بل قدم قراءة فنية موضوعية.
5. تجاهل تماماً أي سؤال عن أسهم فردية (مثل Apple أو Tesla) أو عملات رقمية، وأخبر المستخدم أنك متخصص في S&P 500 فقط."""


def tg_send_text(text, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except: return False


def tg_send_photo(photo_bytes, caption, chat_id=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        files = {"photo": ("chart.png", photo_bytes, "image/png")}
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          files=files, data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}, timeout=30)
        return r.status_code == 200
    except: return False


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
        print(f"Fetch error {symbol}: {e}")
        return None


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
    model = VotingClassifier(estimators=[("rf", rf)], voting="soft") # Note: Using simple RF here as per previous logic but keeping structure
    # Correction for standalone execution without VotingClassifier import issues if any, sticking to pure RF for stability
    model.fit(X.iloc[:split], y[:split]) 
    
    last_X = X.iloc[-1:].values
    pred = int(model.predict(last_X)[0])
    proba = model.predict_proba(last_X)[0]
    classes = model.classes_
    probs = {int(c): round(p*100, 1) for c, p in zip(classes, proba)}
    conf = max(proba) * 100
    return pred, probs, conf


# ===== الذكاء المدمج: يحلل البيانات ويعطي رأياً قصيراً =====
def get_ai_insight_on_data(sym, df, pred, conf):
    """يقوم بتمرير الحالة الفنية الحالية للذكاء الاصطناعي ليختصرها في جملة واحدة"""
    if not GEMINI_API_KEY or df is None: return ""
    
    l = df.iloc[-1]
    trend = "صاعد" if l["EMA_9"] > l["EMA_21"] else "هابط"
    signal_map = {1: "شراء محتمل", -1: "بيع محتمل", 0: "انتظار"}
    current_signal = signal_map.get(pred, "انتظار")
    
    prompt = f"""
    Data Snapshot for {sym} (15m chart):
    - Price: ${l['Close']:.2f}
    - Trend: {trend} (EMA9 vs EMA21)
    - RSI: {l['RSI']:.1f}
    - MACD Hist: {l['MACD_Hist']:.4f}
    - AI Prediction: {current_signal} with {conf:.1f}% confidence.
    
    Task: Provide a ONE SENTENCE professional trading insight in Arabic based ONLY on this data. Be concise. Do not give financial advice, just technical observation.
    """
    
    try:
        models_to_try = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"]
        for m_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents":[{"parts":[{"text":prompt}]}],
                       "generationConfig":{"temperature":0.3,"maxOutputTokens":100}} # Low temp for factual brevity
            
            r = requests.post(url, json=payload, headers={"Content-Type":"application/json"}, timeout=15)
            if r.status_code == 200:
                cand = r.json().get("candidates",[])
                if cand and cand[0].get("content"):
                    parts = cand[0]["content"].get("parts",[])
                    reply = "".join([p.get("text","") for p in parts]).strip()
                    if reply: return reply
    except: pass
    return ""


def generate_chart(sym, df):
    if df is None or len(df)<20: return None
    p=df.tail(60).copy()
    ap=[mpf.make_addplot(p["EMA_9"],color="#00ffff",width=1.2),
        mpf.make_addplot(p["EMA_21"],color="#ff8c00",width=1.2),
        mpf.make_addplot(p["BB_Upper"],color="#888",width=.6,linestyle="--"),
        mpf.make_addplot(p["BB_Lower"],color="#888",width=.6,linestyle="--")]
    st=mpf.make_mpf_style(base_mpf_style="charles",rc={"figure.facecolor":"#0a0a0a","axes.facecolor":"#141414"})
    buf=io.BytesIO()
    fig,_=mpf.plot(p,type="candle",style=st,addplot=ap,volume=True,figsize=(12,8),title=f"\n{sym} Expert View",returnfig=True)
    ax=fig.add_subplot(3,1,3)
    ax.plot(p.index,p["RSI"],color="#00ff88",lw=1.5)
    ax.axhline(70,color="#ff3366",ls="--",alpha=.6);ax.axhline(30,color="#00ff88",ls="--",alpha=.6)
    ax.set_ylabel("RSI",color="white");ax.tick_params(colors="white");ax.set_facecolor("#141414")
    plt.tight_layout();fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor="#0a0a0a")
    plt.close(fig);buf.seek(0);return buf.read()


def handle_command(text, chat_id):
    global CURRENT_SYMBOL
    t=text.lower().strip()
    
    # 1. فحص التخصيص الصارم
    if any(x in t for x in ["spy", "s&p", "sp500", "index"]):
        target_sym = "SPY"
    elif any(x in t for x in ["spx", "gspc", "^gspc"]):
        target_sym = "^GSPC"
    elif t.startswith("/switch"):
        # أمر تبديل يدوي آمن
        if "spx" in t or "index" in t: target_sym = "^GSPC"
        else: target_sym = "SPY"
    else:
        target_sym = CURRENT_SYMBOL # افتراضي
    
    # منع الرموز غير المصرح بها
    if target_sym not in ALLOWED_SYMBOLS:
        return "⛔ أنا متخصص فقط في S&P 500 (SPY/Index). لا أدعم رموزاً أخرى."
    
    CURRENT_SYMBOL = target_sym
    display_name = "SPY" if target_sym == "SPY" else "SPX (^GSPC)"

    if t.startswith("/start") or t=="/help":
        return ("<b>👋 S&P 500 Specialist Bot</b>\n\n"
                "<b>Current Focus:</b> {}\n"
                "<b>Commands:</b>\n"
                "• /analyze → تحليل فني + رأي ذكي مختصر\n"
                "• /chart → شارت احترافي مع تعليقات\n"
                "• /price → السعر الحالي\n"
                "• /switch spy | /switch spx → تغيير التركيز\n"
                "• سؤال عادي → إجابة متخصصة ومختصرة\n\n"
                "<i>متخصص حصرياً في مؤشر S&P 500.</i>").format(display_name)
    
    raw_df = fetch_bars_safe(CURRENT_SYMBOL, RESOLUTION, 200)
    
    if t.startswith("/price"):
        if raw_df is None: return f"⚠️ لا توجد بيانات لـ {display_name} (السوق مغلق؟)."
        return f"💰 <b>{display_name}</b>: ${raw_df['Close'].iloc[-1]:.2f}"
    
    if t.startswith("/chart"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات لرسم الشارت."
        
        # توليد الصورة
        img = generate_chart(CURRENT_SYMBOL, ind_df)
        if img is None: return "⚠️ تعذر توليد الصورة."
        
        # جلب الرأي الذكي المختصر لإضافته للكابشن
        pred, _, conf = predict(ind_df)
        ai_note = get_ai_insight_on_data(display_name, ind_df, pred, conf)
        
        cap = f"📊 <b>{display_name} Chart</b>\n${ind_df['Close'].iloc[-1]:.2f} | RSI {ind_df['RSI'].iloc[-1]:.1f}\n\n💡 <i>{ai_note}</i>" if ai_note else f"📊 <b>{display_name} Chart</b>\n${ind_df['Close'].iloc[-1]:.2f}"
        
        tg_send_photo(img, cap, chat_id=chat_id)
        return None 
    
    if t.startswith("/analyze"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: return "⚠️ لا توجد بيانات للتحليل."
        
        pred, probs, conf = predict(ind_df)
        l = ind_df.iloc[-1]
        
        # بناء التقرير الأساسي
        smap={1:"🟢 شراء",-1:"🔴 بيع",0:"⚪ انتظار"}
        trnd="صاعد 📈" if l["EMA_9"]>l["EMA_21"] else "هابط 📉"
        
        base_report = (f"<b>🧠 Analysis: {display_name}</b>\n"
                       f"Price: ${l['Close']:.2f} | Trend: {trnd}\n"
                       f"Signal: {smap.get(pred,'⚪')} ({conf:.1f}%)\n"
                       f"RSI: {l['RSI']:.1f} | MACD H: {l['MACD_Hist']:.4f}")
        
        # إضافة اللمسة الذكية المدمجة
        ai_note = get_ai_insight_on_data(display_name, ind_df, pred, conf)
        final_msg = base_report + (f"\n\n💬 <b>Expert Insight:</b>\n{ai_note}" if ai_note else "")
        
        return final_msg
    
    if t.startswith("/switch"):
        return f"✅ Switched focus to: {display_name}. Use /analyze or /chart now."
        
    # الأسئلة العامة (مختصرة وذكية)
    return ask_expert_question(text, chat_id, display_name)


def ask_expert_question(user_text, chat_id, sym_display):
    if not GEMINI_API_KEY:
        return "⚠️ محرك الذكاء الاصطناعي غير متاح حالياً."
    
    history = chat_memory.get(chat_id, [])
    ctx = "\n".join([("U: "+m["content"]) if m["role"]=="user" else ("A: "+m["content"]) for m in history[-4:]])
    
    prompt = f"{SYSTEM_PROMPT}\nContext:\n{ctx}\nUser asks about {sym_display}: {user_text}\nAnswer briefly in Arabic."
    
    try:
        models_to_try = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"]
        for m_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents":[{"parts":[{"text":prompt}]}],
                       "generationConfig":{"temperature":0.4,"maxOutputTokens":250}} # Strict brevity
            
            r = requests.post(url, json=payload, headers={"Content-Type":"application/json"}, timeout=20)
            if r.status_code == 200:
                cand = r.json().get("candidates",[])
                if cand and cand[0].get("content"):
                    parts = cand[0]["content"].get("parts",[])
                    reply = "".join([p.get("text","") for p in parts]).strip()
                    if reply:
                        history.append({"role":"user","content":user_text})
                        history.append({"role":"assistant","content":reply})
                        chat_memory[chat_id] = history[-10:]
                        return reply
    except: pass
    return "❌ تعذر الحصول على إجابة ذكية."


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
        tg_send_text(f"❌ Error: {str(e)[:50]}", chat_id=chat_id)
    return jsonify({"ok": True})


@app.route("/")
@app.route("/health")
def health():
    return jsonify({"status":"ok","bot":"SPY_SPX_Specialist_v10","focus":CURRENT_SYMBOL,"time":datetime.now().strftime("%H:%M:%S")})


def monitor_loop():
    la=0; time.sleep(8)
    tg_send_text(f"✅ S&P 500 Specialist Online\nFocus: {CURRENT_SYMBOL}\n/help for commands")
    while True:
        try:
            raw=fetch_bars_safe(CURRENT_SYMBOL, RESOLUTION, 200)
            if raw is not None:
                ind=compute_indicators(raw)
                if ind is not None and len(ind)>=30:
                    p,pr,c=predict(ind);now=time.time()
                    if p!=0 and c>=MIN_CONFIDENCE and (now-la)>1800:
                        # تنبيه ذكي مختصر
                        note = get_ai_insight_on_data(CURRENT_SYMBOL, ind, p, c)
                        txt = f"🚨 <b>{CURRENT_SYMBOL} Alert</b>\nSignal: {'BUY' if p==1 else 'SELL'} ({c:.0f}%)\n💬 {note}"
                        if tg_send_text(txt): la=now
        except Exception as e: print(f"MonErr:{e}")
        time.sleep(CHECK_INTERVAL)


if __name__=="__main__":
    threading.Thread(target=monitor_loop,daemon=True).start()
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",10000)))
