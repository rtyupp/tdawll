import os, io, json, time, threading, requests
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

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ALLOWED = ["SPY", "^GSPC"]
CURRENT = "SPY"
RES = "15"
MIN_CONF = int(os.environ.get("MIN_CONFIDENCE", "70"))
INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))

chat_memory = {}  # يُصحّح: مفتاحه chat_id الفعلي

# 🎯 الـ Prompt الصارم: متداول محترف، لا خطيب
SYSTEM = """أنت متداول مؤسسي محترف على S&P 500 (SPY/^GSPC).
قواعدك الحديدية:
- أجب بالعربية بأسلوب متداول حقيقي، لا بأسلوب أكاديمي أو إنشائي.
- اختصر جداً: 2-4 أسطر كحد أقصى إلا إذا طُلب تفصيل.
- ابنِ حكمك على الأرقام المعطاة لك فقط، لا تختلق أرقاماً.
- استخدم مصطلحات التداول الحقيقية: تشبع، اختراق، ارتداد، سيولة، اتجاه، وقف، هدف.
- لا تقل أبداً 'أنصحك بالشراء'، قل 'الفنّي يرجّح...' أو 'الإعدادية أفضل لـ...'.
- إذا لم تتوفر بيانات حية، اعتمد على خبرتك واذكر أن السوق مغلق."""


def tg_text(text, chat_id=None, kb=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        d = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if kb: d["reply_markup"] = json.dumps(kb)
        return requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                             data=d, timeout=15).status_code == 200
    except: return False

def tg_photo(img, cap, chat_id=None, kb=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id: return False
    try:
        d = {"chat_id": chat_id, "caption": cap, "parse_mode": "HTML"}
        if kb: d["reply_markup"] = json.dumps(kb)
        return requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                             files={"photo": ("c.png", img, "image/png")}, data=d, timeout=30).status_code == 200
    except: return False


def fetch(symbol, res="15", n=200):
    try:
        to = int(time.time()); frm = to - n * int(res) * 60
        u = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={res}&from={frm}&to={to}&token={FINNHUB_KEY}"
        j = requests.get(u, timeout=20).json()
        if j.get("s") != "ok" or not j.get("c"): return None
        return pd.DataFrame({"Open":j["o"],"High":j["h"],"Low":j["l"],"Close":j["c"],"Volume":j["v"]},
                            index=pd.to_datetime(j["t"], unit="s"))
    except: return None

def indicators(df):
    if df is None or len(df) < 20: return None
    d = df.copy()
    d["RSI"] = RSIIndicator(d["Close"],14).rsi()
    d["MACD_H"] = MACD(d["Close"]).macd_diff()
    d["EMA9"] = EMAIndicator(d["Close"],9).ema_indicator()
    d["EMA21"] = EMAIndicator(d["Close"],21).ema_indicator()
    bb = BollingerBands(d["Close"])
    d["BBU"] = bb.bollinger_hband(); d["BBL"] = bb.bollinger_lband()
    d["STOCH"] = StochasticOscillator(d["High"],d["Low"],d["Close"]).stoch()
    d["ATR"] = AverageTrueRange(d["High"],d["Low"],d["Close"]).average_true_range()
    return d.dropna()

def predict(df):
    if df is None or len(df) < 30: return 0, 0.0
    feats = ["RSI","MACD_H","EMA9","EMA21","BBU","BBL","STOCH","ATR"]
    X = df[feats].copy()
    for c in X.columns: X[c] = (X[c]-X[c].mean())/(X[c].std()+1e-9)
    fr = df["Close"].shift(-8)/df["Close"]-1
    y = np.where(fr>0.003,1,np.where(fr<-0.003,-1,0))
    sp = int(len(X)*0.85)
    m = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,class_weight="balanced")
    m.fit(X.iloc[:sp], y[:sp])
    lx = X.iloc[-1:].values
    p = int(m.predict(lx)[0]); pr = m.predict_proba(lx)[0]
    return p, float(max(pr)*100)


# 🧠 البطاقة الفنية: هذا ما يحوّل الغباء لذكاء
def market_card(sym, df, pred, conf):
    l = df.iloc[-1]
    trend = "صاعد" if l["EMA9"] > l["EMA21"] else "هابط"
    sig = {1:"إعدادية شراء", -1:"إعدادية بيع", 0:"حياد/انتظار"}.get(pred,"حياد")
    rsi_z = "تشبع شرائي" if l["RSI"]>70 else "تشبع بيعي" if l["RSI"]<30 else "محايد"
    return (f"SYMBOL={sym} | PRICE=${l['Close']:.2f} | TREND={trend}\n"
            f"RSI14={l['RSI']:.1f}({rsi_z}) | MACD_HIST={l['MACD_H']:.4f}\n"
            f"STOCH={l['STOCH']:.1f} | ATR={l['ATR']:.2f} | BB_WIDTH={(l['BBU']-l['BBL'])/l['Close']*100:.2f}%\n"
            f"MODEL_SIGNAL={sig} CONF={conf:.0f}%")


# 🤖 العقل: يُطعَم البطاقة لا سؤالاً فارغاً
def brain(task, card=None, chat_id=None, tokens=220):
    if not GEMINI_API_KEY: return "⚠️ مفتاح Gemini غير مضبوط."
    hist = chat_memory.get(chat_id, []) if chat_id else []
    ctx = "\n".join([("س: "+m["c"]) if m["r"]=="u" else ("ج: "+m["c"]) for m in hist[-4:]])
    data_block = f"\n[بيانات فنية حية]\n{card}\n" if card else ""
    prompt = f"{SYSTEM}\n\n[سياق سابق]\n{ctx}\n{data_block}\n[المطلوب]\n{task}"
    for mdl in ["gemini-flash-latest","gemini-2.5-flash"]:
        try:
            u = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={GEMINI_API_KEY}"
            pl = {"contents":[{"parts":[{"text":prompt}]}],
                  "generationConfig":{"temperature":0.35,"maxOutputTokens":tokens,"topP":0.9,"topK":50}}
            r = requests.post(u, json=pl, headers={"Content-Type":"application/json"}, timeout=25)
            if r.status_code == 200:
                cd = r.json().get("candidates",[])
                if cd and cd[0].get("content"):
                    txt = "".join([p.get("text","") for p in cd[0]["content"].get("parts",[])]).strip()
                    if txt:
                        if chat_id:
                            hist.append({"r":"u","c":task}); hist.append({"r":"a","c":txt})
                            chat_memory[chat_id] = hist[-10:]
                        return txt
        except: continue
    return "❌ تعذّر توليد الإجابة."


def chart_img(sym, df):
    if df is None or len(df)<20: return None
    p = df.tail(60).copy()
    ap = [mpf.make_addplot(p["EMA9"],color="#00ffff",width=1.2),
          mpf.make_addplot(p["EMA21"],color="#ff8c00",width=1.2),
          mpf.make_addplot(p["BBU"],color="#888",width=.6,linestyle="--"),
          mpf.make_addplot(p["BBL"],color="#888",width=.6,linestyle="--")]
    st = mpf.make_mpf_style(base_mpf_style="charles",rc={"figure.facecolor":"#0a0a0a","axes.facecolor":"#141414"})
    buf = io.BytesIO()
    fig,_ = mpf.plot(p,type="candle",style=st,addplot=ap,volume=True,figsize=(12,8),title=f"\n{sym}",returnfig=True)
    ax = fig.add_subplot(3,1,3)
    ax.plot(p.index,p["RSI"],color="#00ff88",lw=1.5)
    ax.axhline(70,color="#ff3366",ls="--",alpha=.6); ax.axhline(30,color="#00ff88",ls="--",alpha=.6)
    ax.set_ylabel("RSI",color="white"); ax.tick_params(colors="white"); ax.set_facecolor("#141414")
    plt.tight_layout(); fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor="#0a0a0a")
    plt.close(fig); buf.seek(0); return buf.read()


def keyboard():
    return {"inline_keyboard":[
        [{"text":"🧠 تحليل ذكي","callback_data":"analyze"},{"text":"📈 الشارت","callback_data":"chart"}],
        [{"text":"💰 السعر","callback_data":"price"},{"text":"📰 الأخبار","callback_data":"news"}],
        [{"text":"🔄 SPY ⇄ SPX","callback_data":"switch"}]
    ]}


def dispatch(cmd, chat_id):
    global CURRENT
    c = cmd.lower().strip()
    disp = "SPY" if CURRENT=="SPY" else "SPX"
    raw = fetch(CURRENT, RES, 200)
    ind = indicators(raw)

    if c in ("start","help","cmd_help") or c.startswith("/start") or c.startswith("/help"):
        return ("<b>👋 S&P 500 Specialist</b>\n\n"
                f"التركيز: <b>{disp}</b>\n"
                "اضغط زرّاً أو اسألني أي شيء عن المؤشر.\n"
                "<i>أجيب كأحد المتداولين، لا كخطيب.</i>"), keyboard()

    if c.startswith("/switch") or c=="switch":
        CURRENT = "^GSPC" if CURRENT=="SPY" else "SPY"
        return f"✅ التركيز الآن: <b>{'SPY' if CURRENT=='SPY' else 'SPX (^GSPC)'}</b>", keyboard()

    if c.startswith("/price") or c=="price":
        if ind is None:
            return brain("السوق مغلق. أعطني ملاحظة مهنية قصيرة عن ما يجب مراقبته عند الفتح.", None, chat_id, 120), keyboard()
        card = market_card(disp, ind, *predict(ind))
        return f"💰 <b>{disp}</b>: ${ind['Close'].iloc[-1]:.2f}\n\n"+brain("علّق على هذا السعر في سطر واحد بأسلوب متداول.", card, chat_id, 80), keyboard()

    if c.startswith("/chart") or c=="chart":
        if ind is None:
            return brain("السوق مغلق ولا أستطيع رسم شارت. اذكر لي أهم 3 مستويات فنية أراقبها عند الفتح.", None, chat_id, 150), keyboard()
        img = chart_img(CURRENT, ind)
        if img is None: return "⚠️ تعذّر الرسم.", keyboard()
        pred, conf = predict(ind)
        card = market_card(disp, ind, pred, conf)
        note = brain("اقرأ البطاقة الفنية وأعطِني خلاصة الشارت في سطرين كحد أقصى.", card, chat_id, 120)
        tg_photo(img, f"📊 <b>{disp}</b> | ${ind['Close'].iloc[-1]:.2f}\n\n💡 {note}", chat_id, keyboard())
        return None, None

    if c.startswith("/analyze") or c=="analyze":
        if ind is None:
            return brain("السوق مغلق. قدّم لي خطة مراقبة مهنية قصيرة لما سأراه عند الافتتاح.", None, chat_id, 200), keyboard()
        pred, conf = predict(ind)
        card = market_card(disp, ind, pred, conf)
        ans = brain("بناءً على البطاقة الفنية، أعطِني قراءة مهنية: الاتجاه، التشبع، ومستويات الاهتمام. 3 أسطر كحد أقصى.", card, chat_id, 220)
        return f"<b>🧠 {disp}</b>\n\n{ans}", keyboard()

    if c.startswith("/news") or c=="news":
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            past = (datetime.now()-timedelta(days=15)).strftime("%Y-%m-%d")
            u = f"https://finnhub.io/api/v1/company-news?symbol={CURRENT}&from={past}&to={today}&token={FINNHUB_KEY}"
            nw = requests.get(u, timeout=10).json()
            if not isinstance(nw,list) or not nw:
                return brain("لا أخبار حديثة. اذكر لي المحفّزات الاقتصادية التي تحرّك S&P 500 هذا الأسبوع.", None, chat_id, 180), keyboard()
            msg = "<b>📰 آخر أخبار "+disp+"</b>\n\n"
            for n in nw[:3]: msg += f"• <a href='{n['url']}'>{n['headline']}</a>\n"
            return msg, keyboard()
        except: return "⚠️ تعذّر جلب الأخبار.", keyboard()

    # سؤال حر → يُطعَم بالبيانات إن توفرت
    card = market_card(disp, ind, *predict(ind)) if ind is not None else None
    return brain(cmd, card, chat_id, 300), keyboard()


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    if "callback_query" in data:
        cq = data["callback_query"]; cid = cq["message"]["chat"]["id"]
        try:
            rep, kb = dispatch(cq["data"], cid)
            if rep: tg_text(rep, cid, kb)
            requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery?callback_query_id={cq['id']}")
        except Exception as e: print("CB:", e)
        return jsonify({"ok":True})
    msg = data.get("message",{}); cid = msg.get("chat",{}).get("id"); txt = msg.get("text","")
    if not cid or not txt: return jsonify({"ok":True})
    try:
        rep, kb = dispatch(txt, cid)
        if rep: tg_text(rep, cid, kb)
    except Exception as e:
        tg_text(f"❌ {str(e)[:60]}", cid, keyboard())
    return jsonify({"ok":True})


@app.route("/")
@app.route("/health")
def health():
    return jsonify({"status":"ok","bot":"pro-trader-v13","focus":CURRENT,"time":datetime.now().strftime("%H:%M:%S")})


def monitor():
    la = 0; time.sleep(8)
    tg_text(f"✅ Specialist Online | {CURRENT}", TELEGRAM_CHAT_ID, keyboard())
    while True:
        try:
            raw = fetch(CURRENT, RES, 200); ind = indicators(raw)
            if ind is not None and len(ind)>=30:
                p, c = predict(ind); now = time.time()
                if p!=0 and c>=MIN_CONF and (now-la)>1800:
                    card = market_card(CURRENT, ind, p, c)
                    note = brain("تنبيه آلي. لخّص سبب الإشارة في سطر واحد حادّ.", card, None, 90)
                    if tg_text(f"🚨 <b>{CURRENT}</b>\n{note}", TELEGRAM_CHAT_ID, keyboard()): la = now
        except Exception as e: print("M:", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    threading.Thread(target=monitor, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",10000)))
