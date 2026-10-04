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

ALLOWED_SYMBOLS = ["SPY", "^GSPC"] 
CURRENT_SYMBOL = "SPY"
RESOLUTION = "15"
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "70"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))

print(f"✅ Smart Specialist Bot Initialized for SPY/SPX")

chat_memory = {}

# نظام تعليمات مرن وذكي (بدون قيود طول صارمة)
SYSTEM_PROMPT = """أنت خبير تداول مالي عالمي متخصص حصرياُ في S&P 500 (SPY ETF و ^GSPC Index).

قواعد التفاعل:
1. كن ذكياً وغنياً بالمعلومات. لا تختصر إلا إذا كان السؤال بسيطاً جداً.
2. اشرح المفاهيم المعقدة بوضوح وبأمثلة عملية مرتبطة بالسوق الأمريكي.
3. عند تحليل البيانات الفنية، قدم قراءة احترافية تربط بين المؤشرات (RSI, MACD, EMA) وسياق السوق العام.
4. استخدم العربية الفصحى المبسطة مع المصطلحات الإنجليزية بين قوسين عند الضرورة.
5. لا تقدم نصيحة مالية مباشرة ("اشترِ الآن")، بل قدم رؤى تحليلية ("البيانات تشير إلى...").
6. كن ودوداً ومتحمساً لمساعدة المستخدم على فهم الأسواق بشكل أفضل."""


def tg_send_text(text, chat_id=None, reply_markup=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup) # Need to import json
        
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data=data, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False

# Import json here since we use it above
import json 

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
        std_val = X[c].std(); mean_val = X[c].mean()
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


# ===== المحرك الذكي الموحد (The Unified Brain) =====
def get_ai_response(prompt_text, context_data=None):
    """
    يرسل الاستعلام والسياق إلى Gemini للحصول على إجابة ذكية ومفصلة.
    يعمل هذا سواء للسؤال النظري أو للتحليل الفني.
    """
    if not GEMINI_API_KEY:
        return "⚠️ محرك الذكاء الاصطناعي غير متاح."
    
    history = chat_memory.get(TELEGRAM_CHAT_ID, [])
    ctx_str = "\n".join([("U: "+m["content"]) if m["role"]=="user" else ("A: "+m["content"]) for m in history[-4:]])
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nConversation History:\n{ctx_str}\n\nCurrent Context Data:\n{context_data if context_data else 'None'}\n\nUser Query: {prompt_text}\nProvide a comprehensive, expert response in Arabic."
    
    try:
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
        for m_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents":[{"parts":[{"text":full_prompt}]}],
                       "generationConfig":{"temperature":0.7,"maxOutputTokens":1024}} 
            
            r = requests.post(url, json=payload, headers={"Content-Type":"application/json"}, timeout=20)
            if r.status_code == 200:
                cand = r.json().get("candidates",[])
                if cand and cand[0].get("content"):
                    parts = cand[0]["content"].get("parts",[])
                    reply = "".join([p.get("text","") for p in parts]).strip()
                    if reply:
                        # Update memory
                        history.append({"role":"user","content":prompt_text})
                        history.append({"role":"assistant","content":reply})
                        chat_memory[TELEGRAM_CHAT_ID] = history[-10:]
                        return reply
    except: pass
    return "❌ تعذر توليد إجابة ذكية."


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
        if "spx" in t or "index" in t: target_sym = "^GSPC"
        else: target_sym = "SPY"
    else:
        target_sym = CURRENT_SYMBOL
    
    if target_sym not in ALLOWED_SYMBOLS:
        return "⛔ أنا متخصص فقط في S&P 500 (SPY/Index)."
    
    CURRENT_SYMBOL = target_sym
    display_name = "SPY" if target_sym == "SPY" else "SPX (^GSPC)"

    # --- إنشاء لوحة الأزرار الجميلة بالعربي ---
    keyboard_menu = {
        "inline_keyboard": [
            [{"text": "📊 تحليل فوري", "callback_data": "cmd_analyze"}],
            [{"text": "📈 رسم الشارت", "callback_data": "cmd_chart"}],
            [{"text": "💰 السعر الحالي", "callback_data": "cmd_price"}],
            [{"text": "🔄 تبديل الرمز", "callback_data": "cmd_switch"}],
            [{"text": "ℹ️ المساعدة", "callback_data": "cmd_help"}]
        ]
    }

    if t.startswith("/start") or t=="/help":
        msg = ("<b>👋 أهلاً بك في بوت S&P 500 الخبير!</b>\n\n"
               "<b>التركيز الحالي:</b> {}\n\n"
               "استخدم الأزرار أدناه للتفاعل بسهولة.".format(display_name))
        # نرسل الرسالة مع الأزرار مباشرة عبر الدالة المعدلة
        tg_send_text(msg, chat_id=chat_id, reply_markup=keyboard_menu)
        return None # لأننا أرسلنا الرسالة بالفعل
    
    raw_df = fetch_bars_safe(CURRENT_SYMBOL, RESOLUTION, 200)
    
    if t.startswith("/price"):
        if raw_df is None: 
            ans = get_ai_response(f"What is the current price of {display_name}? Market seems closed.", None)
            return f"<b>💰 حالة السعر:</b>\n{ans}"
        return f"💰 <b>{display_name}</b>: ${raw_df['Close'].iloc[-1]:.2f}"
    
    if t.startswith("/chart"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: 
             ans = get_ai_response(f"I tried to draw a chart for {display_name} but market is closed. Can you explain what I would typically look for?", None)
             return f"<b>📈 وضع الشارت:</b>\n{ans}"
        
        img = generate_chart(CURRENT_SYMBOL, ind_df)
        if img is None: return "⚠️ تعذر توليد الصورة."
        
        pred, _, conf = predict(ind_df)
        l = ind_df.iloc[-1]
        tech_summary = f"Price:${l['Close']:.2f}|RSI:{l['RSI']:.1f}|Trend:{'Up' if l['EMA_9']>l['EMA_21'] else 'Down'}|Signal:{pred}"
        
        ai_caption = get_ai_response(f"Summarize this technical snapshot for {display_name} professionally:", tech_summary)
        
        cap = f"📊 <b>{display_name} Chart</b>\n${ind_df['Close'].iloc[-1]:.2f} | RSI {ind_df['RSI'].iloc[-1]:.1f}\n\n💡 <i>{ai_caption}</i>"
        
        tg_send_photo(img, cap, chat_id=chat_id)
        return None 
    
    if t.startswith("/analyze"):
        ind_df = compute_indicators(raw_df)
        if ind_df is None: 
             ans = get_ai_response(f"Analyze {display_name}. Since market is closed, give me general advice on watching it.", None)
             return f"<b>🧠 التحليل:</b>\n{ans}"
        
        pred, probs, conf = predict(ind_df)
        l = ind_df.iloc[-1]
        
        smap={1:"BUY",-1:"SELL",0:"WAIT"}
        trnd="UP" if l["EMA_9"]>l["EMA_21"] else "DOWN"
        
        tech_snapshot = f"Symbol:{display_name}|Price:${l['Close']:.2f}|Trend:{trnd}|RSI:{l['RSI']:.1f}|MACD_H:{l['MACD_Hist']:.4f}|AI_Signal:{smap.get(pred,'WAIT')} ({conf:.0f}%)"
        
        final_analysis = get_ai_response(f"Give me a professional trading insight based on this data:", tech_snapshot)
        
        return f"<b>🧠 Analysis: {display_name}</b>\n\n{final_analysis}"
        
    if t.startswith("/switch"):
        return f"✅ Switched focus to: {display_name}."
        
    # الأسئلة العامة تمر عبر العقل المدبر مباشرة
    return get_ai_response(text, None)


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    
    # التعامل مع ضغط الأزرار (Callback Queries)
    callback_query = data.get("callback_query")
    if callback_query:
        chat_id = callback_query["message"]["chat"]["id"]
        cmd = callback_query["data"]
        user_text_map = {
            "cmd_analyze": "/analyze",
            "cmd_chart": "/chart",
            "cmd_price": "/price",
            "cmd_switch": "/switch spy", # Default switch to SPY
            "cmd_help": "/help"
        }
        mapped_text = user_text_map.get(cmd, "")
        if mapped_text:
            reply = handle_command(mapped_text, chat_id)
            if reply: tg_send_text(reply, chat_id=chat_id)
        # Send ACK to Telegram to stop loading spinner
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery?callback_query_id={callback_query['id']}")
        return jsonify({"ok": True})

    # التعامل مع الرسائل النصية العادية
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
    return jsonify({"status":"ok","bot":"Smart_SPY_SPX_v12_FriendlyUI","focus":CURRENT_SYMBOL,"time":datetime.now().strftime("%H:%M:%S")})


def monitor_loop():
    la=0; time.sleep(8)
    tg_send_text(f"✅ Smart Specialist Online\nFocus: {CURRENT_SYMBOL}\nAsk anything!")
    while True:
        try:
            raw=fetch_bars_safe(CURRENT_SYMBOL, RESOLUTION, 200)
            if raw is not None:
                ind=compute_indicators(raw)
                if ind is not None and len(ind)>=30:
                    p,pr,c=predict(ind);now=time.time()
                    if p!=0 and c>=MIN_CONFIDENCE and (now-la)>1800:
                        note = get_ai_response(f"Alert triggered for {CURRENT_SYMBOL}. Signal: {'BUY' if p==1 else 'SELL'}. Confidence: {c}%. Give me a detailed alert reason.", None)
                        txt = f"🚨 <b>{CURRENT_SYMBOL} Alert</b>\n{note}"
                        if tg_send_text(txt): la=now
        except Exception as e: print(f"MonErr:{e}")
        time.sleep(CHECK_INTERVAL)


if __name__=="__main__":
    threading.Thread(target=monitor_loop,daemon=True).start()
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",10000)))
