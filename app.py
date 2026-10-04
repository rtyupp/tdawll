# -*- coding: utf-8 -*-
"""
tdawll v2 — بوت تحليل S&P 500 (SPY / SPX) على تيليجرام
البيانات: Yahoo (بدون مفتاح) + Finnhub احتياطي | الذكاء: Gemini | الاستضافة: Render المجاني
"""
import os, io, re, json, time, html, logging, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tdawll")
app = Flask(__name__)

# ============================== الإعدادات ==============================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
ALLOWED_CHATS = {x.strip() for x in os.environ.get("ALLOWED_CHAT_IDS", TELEGRAM_CHAT_ID).split(",") if x.strip()}
ALERT_SCORE = int(os.environ.get("ALERT_SCORE", "45"))        # أقل درجة تقارب لإرسال تنبيه
INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))       # ثواني بين الفحوصات
BRIEFING = os.environ.get("MORNING_BRIEFING", "1") == "1"     # إحاطة قبل الافتتاح
NOTIFY_STARTUP = os.environ.get("NOTIFY_STARTUP", "0") == "1"
AUTO_WEBHOOK = os.environ.get("AUTO_WEBHOOK", "1") == "1"
FAST_MODELS = os.environ.get("GEMINI_FAST", "gemini-flash-lite-latest,gemini-flash-latest").split(",")
DEEP_MODELS = os.environ.get("GEMINI_DEEP", "gemini-flash-latest,gemini-2.5-flash,gemini-flash-lite-latest").split(",")

SYMBOLS = {"SPY": "SPY", "SPX": "^GSPC"}
state = {"focus": "SPY"}
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (compatible; tdawll/2.0)"}
DIR_TXT = {1: "إعدادية شراء", -1: "إعدادية بيع", 0: "حياد/انتظار"}

chat_memory = {}
pool = ThreadPoolExecutor(max_workers=3)
_seen_updates = deque(maxlen=300)
_cache = {}
_ml_cache = {}
_started = time.time()

SYSTEM = """أنت «تداول»، محلل مؤسسي متخصص في مؤشر S&P 500 (SPY / SPX).
قواعد لا تُكسر:
1) كل رقم تذكره يجب أن يكون موجوداً في «البطاقة الفنية» المرفقة. ممنوع اختلاق أسعار أو مستويات أو أخبار.
2) إذا كان السوق مغلقاً فالبطاقة تحوي آخر بيانات حقيقية: حلّل آخر إغلاق وجهّز خطة للافتتاح. لا تقل «لا توجد بيانات» ما دامت البطاقة موجودة.
3) إذا تعارض الاتجاه على 15 دقيقة مع الاتجاه اليومي فاذكر ذلك صراحة، فهو أهم معلومة في القراءة.
4) إن كان نموذج ML موسوماً «غير موثوق» فلا تبنِ عليه؛ اعتمد على درجة القواعد والسياق وقل ذلك بسطر.
5) في كل قراءة: الاتجاه، الزخم/التشبع، أقرب دعم ومقاومة، وشرط إبطال الفكرة (الوقف).
6) الأخبار: استخدم العناوين المرفقة فقط، وفرّق بين ما يدعم الاتجاه وما يعاكسه. لا أخبار مرفقة = لا تتحدث عن أخبار.
7) لا تقل «اشترِ» أو «بع». قل «الفنّي يرجّح…» وأرفق المخاطرة.
8) أسلوب عربي حاد ومباشر كمتداول محترف، بلا مقدمات ولا وعظ ولا اعتذارات.
9) نص عادي. يمكنك **تمييز** كلمات قليلة فقط. بدون عناوين أو جداول.
10) التزم بالطول المطلوب في الطلب."""


# ============================== أدوات عامة ==============================
def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception as e:
        log.warning("cache-fn %s: %s", key, e)
        val = None
    if val is not None:
        _cache[key] = (time.time(), val)
        return val
    return hit[1] if hit else None          # عند الفشل نرجع آخر نسخة قديمة بدل لا شيء


def now_et():
    return datetime.now(NY)


def session_info(now=None):
    """حالة سوق الأسهم الأمريكي (لا يعرف العطل الرسمية)."""
    now = now or now_et()
    t = now.hour * 60 + now.minute
    if now.weekday() >= 5:
        return "closed", "مغلق (عطلة الأسبوع)"
    if 570 <= t < 960:
        return "open", "مفتوح"
    if 240 <= t < 570:
        return "pre", "ما قبل الافتتاح"
    if 960 <= t < 1200:
        return "after", "ما بعد الإغلاق"
    return "closed", "مغلق"


def ok(x):
    return x is not None and x == x


def fmt_ai(text):
    """يهرّب HTML ثم يحوّل **x** إلى <b>x</b> حتى لا يفشل إرسال تيليجرام."""
    t = html.escape(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    return t.replace("**", "").strip()


# ============================== تيليجرام ==============================
def _tg(method, **kw):
    return requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}", timeout=25, **kw)


def tg_text(text, chat_id=None, kb=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id or not text:
        return False
    for mode in ("HTML", None):
        try:
            body = text[:4000] if mode else re.sub(r"<[^>]+>", "", html.unescape(text))[:4000]
            d = {"chat_id": chat_id, "text": body, "disable_web_page_preview": True}
            if mode: d["parse_mode"] = mode
            if kb: d["reply_markup"] = json.dumps(kb)
            r = _tg("sendMessage", data=d)
            if r.status_code == 200:
                return True
            log.warning("tg_text %s: %s", r.status_code, r.text[:150])
        except Exception as e:
            log.warning("tg_text err: %s", e)
    return False


def tg_photo(img, cap, chat_id=None, kb=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return False
    for mode in ("HTML", None):
        try:
            c = cap[:1000] if mode else re.sub(r"<[^>]+>", "", html.unescape(cap))[:1000]
            d = {"chat_id": chat_id, "caption": c}
            if mode: d["parse_mode"] = mode
            if kb: d["reply_markup"] = json.dumps(kb)
            r = _tg("sendPhoto", files={"photo": ("c.png", img, "image/png")}, data=d)
            if r.status_code == 200:
                return True
            log.warning("tg_photo %s: %s", r.status_code, r.text[:150])
        except Exception as e:
            log.warning("tg_photo err: %s", e)
    return False


def keyboard():
    return {"inline_keyboard": [
        [{"text": "🧠 تحليل ذكي", "callback_data": "analyze"}, {"text": "📈 الشارت", "callback_data": "chart"}],
        [{"text": "🎯 المستويات", "callback_data": "levels"}, {"text": "💰 السعر", "callback_data": "price"}],
        [{"text": "📰 الأخبار", "callback_data": "news"}, {"text": "🔄 SPY ⇄ SPX", "callback_data": "switch"}],
    ]}


# ============================== البيانات ==============================
def yahoo_candles(symbol, interval, rng):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}",
                             params={"interval": interval, "range": rng, "includePrePost": "false"},
                             headers=UA, timeout=15)
            if r.status_code != 200:
                log.warning("yahoo %s %s -> %s", host, symbol, r.status_code)
                continue
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            q = res["indicators"]["quote"][0]
            if not ts:
                continue
            idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY).tz_localize(None)
            df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                               "Close": q["close"], "Volume": q["volume"]}, index=idx)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df["Volume"] = df["Volume"].fillna(0)
            df = df[~df.index.duplicated(keep="last")]
            if len(df):
                return df
        except Exception as e:
            log.warning("yahoo err %s: %s", symbol, e)
    return None


def finnhub_candles(symbol, res="15", days=12):
    """احتياطي: شموع Finnhub قد تكون مدفوعة حسب خطتك، لذلك ليست المصدر الأول."""
    if not FINNHUB_KEY:
        return None
    try:
        to = int(time.time()); frm = to - days * 86400
        j = requests.get("https://finnhub.io/api/v1/stock/candle",
                         params={"symbol": symbol, "resolution": res, "from": frm, "to": to, "token": FINNHUB_KEY},
                         timeout=15).json()
        if j.get("s") != "ok" or not j.get("c"):
            return None
        idx = pd.to_datetime(j["t"], unit="s", utc=True).tz_convert(NY).tz_localize(None)
        return pd.DataFrame({"Open": j["o"], "High": j["h"], "Low": j["l"], "Close": j["c"], "Volume": j["v"]}, index=idx)
    except Exception as e:
        log.warning("finnhub candle err: %s", e)
        return None


def candles(symbol, interval="15m", rng="60d"):
    def go():
        df = yahoo_candles(symbol, interval, rng)
        if df is None and symbol == "SPY" and interval == "15m":
            df = finnhub_candles(symbol)
        return df
    return cached(f"c:{symbol}:{interval}:{rng}", 60 if interval != "1d" else 600, go)


NEWS_KW = re.compile(r"\b(fed|powell|fomc|inflation|cpi|ppi|pce|jobs|payroll|unemployment|rates?|yields?|treasur\w*|tariffs?|"
                     r"s&p|stocks?|wall street|nasdaq|dow|earnings|recession|gdp|oil|economy|economic|market\w*|"
                     r"nvidia|apple|microsoft|amazon|trump|shutdown|dollar)\b", re.I)


def get_news(n=8):
    def go():
        items = []
        if FINNHUB_KEY:
            try:
                j = requests.get("https://finnhub.io/api/v1/news", params={"category": "general", "token": FINNHUB_KEY}, timeout=10).json()
                cut = time.time() - 48 * 3600
                for x in (j if isinstance(j, list) else []):
                    if x.get("datetime", 0) >= cut and x.get("headline") and NEWS_KW.search(x["headline"]):
                        items.append({"h": x["headline"], "u": x.get("url", ""), "s": x.get("source", ""), "t": x["datetime"]})
            except Exception as e:
                log.warning("finnhub news: %s", e)
        if len(items) < 3:
            try:
                j = requests.get("https://query1.finance.yahoo.com/v1/finance/search",
                                 params={"q": "S&P 500 stock market", "newsCount": 12, "quotesCount": 0},
                                 headers=UA, timeout=10).json()
                for x in j.get("news", []):
                    if x.get("title"):
                        items.append({"h": x["title"], "u": x.get("link", ""), "s": x.get("publisher", ""),
                                      "t": x.get("providerPublishTime", 0)})
            except Exception as e:
                log.warning("yahoo news: %s", e)
        seen, out = set(), []
        for x in sorted(items, key=lambda z: -z["t"]):
            k = x["h"].lower()[:60]
            if k not in seen:
                seen.add(k); out.append(x)
        return out[:n]
    return cached("news", 600, go) or []


# ============================== المؤشرات (pandas صافي) ==============================
def _rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ru / (rd + 1e-12))


def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    d["RSI"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()
    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    m, s = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    d["BBU"], d["BBL"] = m + 2 * s, m - 2 * s
    d["PCTB"] = (c - d["BBL"]) / (d["BBU"] - d["BBL"] + 1e-9)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    d["STOCH"] = 100 * (c - ll) / (hh - ll + 1e-9)
    d["ATR"] = _atr(h, l, c)
    d["VOLR"] = v / (v.rolling(20).mean() + 1e-9)
    tp = (h + l + c) / 3
    day = d.index.normalize()
    cv = v.groupby(day).cumsum()
    cpv = (tp * v).groupby(day).cumsum()
    d["VWAP"] = np.where(cv > 0, cpv / cv.replace(0, np.nan), tp)
    return d


def daily_context(sym):
    d = candles(sym, "1d", "1y")
    if d is None or len(d) < 30:
        return None
    c = d["Close"]
    prev = d.iloc[-2]
    P = (prev["High"] + prev["Low"] + prev["Close"]) / 3
    return dict(
        last=float(c.iloc[-1]), prev_close=float(prev["Close"]), prev_high=float(prev["High"]), prev_low=float(prev["Low"]),
        hi20=float(d["High"].tail(20).max()), lo20=float(d["Low"].tail(20).min()),
        sma50=float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else float("nan"),
        sma200=float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan"),
        rsi=float(_rsi(c).iloc[-1]), pivot=float(P),
        r1=float(2 * P - prev["Low"]), s1=float(2 * P - prev["High"]))


def macro_context():
    out = {}
    for key, sym in (("vix", "^VIX"), ("tnx", "^TNX")):
        d = candles(sym, "1d", "1mo")
        if d is not None and len(d) >= 2:
            v, p = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
            if key == "tnx" and v > 20:
                v, p = v / 10, p / 10
            out[key] = (v, (v / p - 1) * 100 if p else 0.0)
    return out


# ============================== النموذج (بدون تسريب مستقبلي + اختبار صادق) ==============================
def ml_features(d):
    c = d["Close"]
    f = pd.DataFrame(index=d.index)
    f["R1"], f["R4"], f["R8"] = c.pct_change(1) * 100, c.pct_change(4) * 100, c.pct_change(8) * 100
    f["RSI"] = d["RSI"]
    f["MACDN"] = d["MACD_H"] / d["ATR"]
    f["DEMA"] = (c - d["EMA21"]) / d["ATR"]
    f["DTREND"] = (d["EMA9"] - d["EMA21"]) / d["ATR"]
    f["PCTB"], f["STOCH"] = d["PCTB"], d["STOCH"]
    f["ATRP"] = d["ATR"] / c * 100
    f["VOLR"] = d["VOLR"].clip(upper=5)
    f["DVWAP"] = (c - d["VWAP"]) / d["ATR"]
    return f.replace([np.inf, -np.inf], np.nan)


def ml_signal(d, sym, horizon=8):
    key = (sym, d.index[-1])
    if key in _ml_cache:
        return _ml_cache[key]
    f = ml_features(d)
    fwd = d["Close"].shift(-horizon) / d["Close"] - 1
    thr = 0.6 * (d["ATR"] / d["Close"])                    # حركة «ذات معنى» = 0.6 ATR
    y = pd.Series(0, index=d.index)
    y[fwd > thr] = 1
    y[fwd < -thr] = -1
    okm = f.notna().all(axis=1) & fwd.notna()
    X, Y = f[okm], y[okm]
    res = None
    if len(X) >= 250 and Y.nunique() >= 2 and f.iloc[[-1]].notna().all(axis=1).iloc[0]:
        cut = int(len(X) * 0.7)
        mk = lambda: RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=15,
                                            class_weight="balanced_subsample", random_state=42, n_jobs=1)
        m = mk().fit(X.iloc[:cut - horizon], Y.iloc[:cut - horizon])       # فجوة تمنع تداخل التسميات
        Xte, Yte = X.iloc[cut:], Y.iloc[cut:].values
        pred = m.predict(Xte)
        mask = pred != 0
        hit = float((pred[mask] == Yte[mask]).mean()) if mask.any() else float("nan")
        m2 = mk().fit(X, Y)
        lx = f.iloc[[-1]]
        p = int(m2.predict(lx)[0])
        conf = float(m2.predict_proba(lx)[0].max() * 100)
        res = dict(pred=p, conf=conf, hit=hit, n_sig=int(mask.sum()), n_test=len(Yte),
                   reliable=bool(mask.sum() >= 20 and ok(hit) and hit >= 0.55))
    if len(_ml_cache) > 20:
        _ml_cache.clear()
    _ml_cache[key] = res
    return res


# ============================== محرك التقارب (قواعد شفافة) ==============================
def confluence(l, D, macro, ml):
    s, why = 0, []
    e9, e21, e50 = l["EMA9"], l["EMA21"], l["EMA50"]
    if ok(e50) and e9 > e21 > e50:
        s += 25; why.append("اتجاه 15د صاعد منظم (EMA9>21>50)")
    elif ok(e50) and e9 < e21 < e50:
        s -= 25; why.append("اتجاه 15د هابط منظم (EMA9<21<50)")
    elif e9 > e21:
        s += 10; why.append("EMA9 فوق EMA21")
    else:
        s -= 10; why.append("EMA9 تحت EMA21")
    if l["Close"] > l["VWAP"]:
        s += 10; why.append("السعر فوق VWAP")
    else:
        s -= 10; why.append("السعر تحت VWAP")
    if l["MACD_H"] > 0:
        s += 10; why.append("هيستوغرام MACD موجب")
    else:
        s -= 10; why.append("هيستوغرام MACD سالب")
    r = l["RSI"]
    if r > 70:
        s -= 10; why.append(f"RSI {r:.0f} تشبع شرائي (خطر ارتداد)")
    elif r < 30:
        s += 10; why.append(f"RSI {r:.0f} تشبع بيعي (فرصة ارتداد)")
    elif 55 <= r <= 70:
        s += 8; why.append(f"RSI {r:.0f} زخم صاعد صحي")
    elif 30 < r <= 45:
        s -= 8; why.append(f"RSI {r:.0f} زخم ضعيف")
    if D:
        c = l["Close"]
        if ok(D["sma50"]) and ok(D["sma200"]):
            if c > D["sma50"] > D["sma200"]:
                s += 20; why.append("يومي: فوق SMA50 و SMA200 (اتجاه صاعد)")
            elif c < D["sma50"] < D["sma200"]:
                s -= 20; why.append("يومي: تحت SMA50 و SMA200 (اتجاه هابط)")
            elif c > D["sma200"]:
                s += 8; why.append("يومي: فوق SMA200 لكن تحت/حول SMA50")
            else:
                s -= 8; why.append("يومي: تحت SMA200")
    v = macro.get("vix")
    if v:
        if v[0] > 25 or v[1] > 8:
            s -= 10; why.append(f"VIX مرتفع/قافز ({v[0]:.1f}, {v[1]:+.1f}%) = ضغط على الأسهم")
        elif v[0] < 16 and v[1] <= 0:
            s += 5; why.append(f"VIX هادئ ({v[0]:.1f})")
    if ml and ml["reliable"] and ml["pred"] != 0:
        s += 15 * ml["pred"]; why.append("نموذج ML (موثوق في الاختبار) يوافق الاتجاه" if ml["pred"] * s > 0 else "نموذج ML (موثوق) يعاكس الاتجاه")
    s = int(max(-100, min(100, s)))
    return s, why


def snapshot():
    label = state["focus"]; sym = SYMBOLS[label]
    raw = candles(sym, "15m", "60d")
    if raw is None or len(raw) < 120:
        return None
    d = add_indicators(raw).dropna(subset=["RSI", "MACD_H", "EMA21", "BBU", "STOCH", "ATR"])
    if len(d) < 100:
        return None
    l = d.iloc[-1]
    D, macro = daily_context(sym), macro_context()
    ml = ml_signal(d, sym)
    score, why = confluence(l, D, macro, ml)
    direction = 1 if score >= 30 else -1 if score <= -30 else 0
    age = (now_et().replace(tzinfo=None) - d.index[-1]).total_seconds() / 60
    sess, sess_txt = session_info()
    price = float(l["Close"])
    ref = D["prev_close"] if D else float(d["Close"].iloc[-2])
    S = dict(label=label, sym=sym, d=d, l=l, D=D, macro=macro, ml=ml, score=score, why=why, dir=direction,
             price=price, chg=(price / ref - 1) * 100, atr=float(l["ATR"]), sess=sess, sess_txt=sess_txt,
             stale=(sess == "open" and age > 45), last_bar=d.index[-1])
    if direction:
        risk = 1.2 * S["atr"]
        S["plan"] = dict(entry=price, stop=price - direction * risk, t1=price + direction * risk,
                         t2=price + direction * 2 * risk, risk=risk)
    return S


def ml_line(S):
    m = S["ml"]
    if not m:
        return "ML: بيانات غير كافية"
    side = {1: "صعود", -1: "هبوط", 0: "حياد"}[m["pred"]]
    tag = "موثوق ✅" if m["reliable"] else "غير موثوق ⚠️"
    hit = f"{m['hit']*100:.0f}%" if ok(m["hit"]) else "—"
    return f"ML: {side} ({m['conf']:.0f}%) | دقة الاتجاه في الاختبار {hit} على {m['n_sig']} إشارة → {tag}"


def level_list(S):
    l, D = S["l"], S["D"]
    lv = [("VWAP", l["VWAP"]), ("EMA21 (15د)", l["EMA21"]), ("EMA50 (15د)", l["EMA50"]),
          ("بولنجر العلوي", l["BBU"]), ("بولنجر السفلي", l["BBL"])]
    if D:
        lv += [("أعلى أمس", D["prev_high"]), ("أدنى أمس", D["prev_low"]), ("إغلاق أمس", D["prev_close"]),
               ("أعلى 20 يوم", D["hi20"]), ("أدنى 20 يوم", D["lo20"]), ("محوري", D["pivot"]),
               ("R1", D["r1"]), ("S1", D["s1"]), ("SMA50 يومي", D["sma50"]), ("SMA200 يومي", D["sma200"])]
    lv = [(n, float(v)) for n, v in lv if ok(v)]
    p = S["price"]
    above = sorted([x for x in lv if x[1] > p], key=lambda x: x[1])[:3]
    below = sorted([x for x in lv if x[1] <= p], key=lambda x: -x[1])[:3]
    return above, below


# ============================== البطاقة المُرسلة للذكاء ==============================
def card_text(S, with_news=True):
    l, D, m = S["l"], S["D"], S["macro"]
    trend = "صاعد" if l["EMA9"] > l["EMA21"] else "هابط"
    n = now_et()
    lines = [
        f"NOW_ET={n:%Y-%m-%d %H:%M} ({n:%A}) | SESSION={S['sess_txt']} | LAST_BAR_ET={S['last_bar']:%Y-%m-%d %H:%M}"
        + (" | تحذير: البيانات متأخرة" if S["stale"] else ""),
        f"SYMBOL={S['label']} | PRICE={S['price']:.2f} | CHG_VS_PREV_CLOSE={S['chg']:+.2f}%",
        f"15M: TREND={trend} EMA9={l['EMA9']:.2f} EMA21={l['EMA21']:.2f} EMA50={l['EMA50']:.2f} | RSI={l['RSI']:.1f} "
        f"MACD_H={l['MACD_H']:.3f} STOCH={l['STOCH']:.0f} ATR={S['atr']:.2f} %B={l['PCTB']:.2f} VOLx={l['VOLR']:.1f} "
        f"VWAP={l['VWAP']:.2f} ({'السعر فوقه' if l['Close'] > l['VWAP'] else 'السعر تحته'})",
    ]
    if D:
        lines.append(f"DAILY: RSI={D['rsi']:.0f} SMA50={D['sma50']:.2f} SMA200={D['sma200']:.2f} | PREV H/L/C="
                     f"{D['prev_high']:.2f}/{D['prev_low']:.2f}/{D['prev_close']:.2f} | 20D H/L={D['hi20']:.2f}/{D['lo20']:.2f} | "
                     f"PIVOT={D['pivot']:.2f} R1={D['r1']:.2f} S1={D['s1']:.2f}")
    if m:
        lines.append("MACRO: " + " | ".join(f"{k.upper()}={v[0]:.2f} ({v[1]:+.1f}%)" for k, v in m.items()))
    ab, bl = level_list(S)
    lines.append("NEAREST_ABOVE: " + ", ".join(f"{n_} {v:.2f}" for n_, v in ab))
    lines.append("NEAREST_BELOW: " + ", ".join(f"{n_} {v:.2f}" for n_, v in bl))
    lines.append(f"RULE_SCORE={S['score']:+d}/100 → {DIR_TXT[S['dir']]} | الأسباب: " + "؛ ".join(S["why"]))
    lines.append(ml_line(S))
    if S.get("plan"):
        p = S["plan"]
        lines.append(f"PLAN: دخول {p['entry']:.2f} وقف {p['stop']:.2f} هدف1 {p['t1']:.2f} هدف2 {p['t2']:.2f} (المخاطرة {p['risk']:.2f}$ = 1.2 ATR)")
    if with_news:
        nw = get_news(5)
        lines.append("NEWS: " + (" || ".join(f"{x['h']} ({x['s']})" for x in nw) if nw else "لا توجد أخبار متاحة الآن"))
    return "\n".join(lines)


# ============================== العقل (Gemini) ==============================
def brain(task, card=None, chat_id=None, deep=False, tokens=500, memo=None):
    if not GEMINI_API_KEY:
        return "⚠️ مفتاح Gemini غير مضبوط."
    hist = chat_memory.get(chat_id, []) if chat_id else []
    contents = [{"role": "user" if m["r"] == "u" else "model", "parts": [{"text": m["c"]}]} for m in hist[-6:]]
    body = (f"[البطاقة الفنية الحية — المصدر الوحيد للأرقام]\n{card}\n\n" if card else
            "[لا توجد بيانات حية الآن. صرّح بذلك بسطر ولا تذكر أي أسعار أو مستويات محددة.]\n\n")
    contents.append({"role": "user", "parts": [{"text": body + "[المطلوب]\n" + task}]})

    budget = 1024 if deep else 0
    cfgs = [{"temperature": 0.3, "topP": 0.9, "maxOutputTokens": tokens + budget, "thinkingConfig": {"thinkingBudget": budget}},
            {"temperature": 0.3, "topP": 0.9, "maxOutputTokens": tokens + 600}]
    for mdl in (DEEP_MODELS if deep else FAST_MODELS):
        for ci, cfg in enumerate(cfgs):
            try:
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{mdl.strip()}:generateContent",
                                  headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                                  json={"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": contents,
                                        "generationConfig": cfg}, timeout=45)
                if r.status_code == 200:
                    cd = r.json().get("candidates", [])
                    parts = ((cd[0].get("content") or {}).get("parts", [])) if cd else []
                    txt = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                    if txt:
                        if chat_id:
                            hist.append({"r": "u", "c": memo or task[:300]}); hist.append({"r": "a", "c": txt})
                            chat_memory[chat_id] = hist[-12:]
                        return txt
                    log.warning("gemini EMPTY %s cfg%d", mdl, ci)
                else:
                    log.warning("gemini HTTP%s %s cfg%d: %s", r.status_code, mdl, ci, r.text[:150])
                    if r.status_code in (429, 500, 503):
                        time.sleep(1.2)
                        break                       # انتقل للنموذج التالي
            except Exception as e:
                log.warning("gemini EXC %s: %s", mdl, str(e)[:100])
    return "❌ تعذّر توليد الإجابة (حد الاستخدام أو خلل مؤقت). جرّب بعد قليل."


# ============================== الشارت (matplotlib صافي) ==============================
def chart_img(S, bars=70):
    d = S["d"].tail(bars)
    n = len(d); x = np.arange(n)
    bg, pan = "#0a0a0a", "#141414"
    fig = plt.figure(figsize=(12, 8), facecolor=bg)
    gs = fig.add_gridspec(3, 1, height_ratios=[6, 1.5, 2.5], hspace=0.06)
    ax = fig.add_subplot(gs[0]); axv = fig.add_subplot(gs[1], sharex=ax); axr = fig.add_subplot(gs[2], sharex=ax)
    for a in (ax, axv, axr):
        a.set_facecolor(pan); a.tick_params(colors="#cccccc", labelsize=8)
        for sp in a.spines.values(): sp.set_color("#333333")
        a.grid(color="#222222", lw=.5)
    up = (d["Close"] >= d["Open"]).values
    col = np.where(up, "#00ff88", "#ff3366")
    ax.vlines(x, d["Low"], d["High"], color=col, lw=1)
    ax.bar(x, (d["Close"] - d["Open"]).abs().values + 1e-9, bottom=np.minimum(d["Open"], d["Close"]).values, color=col, width=.65)
    ax.plot(x, d["EMA9"], color="#00e5ff", lw=1.1, label="EMA9")
    ax.plot(x, d["EMA21"], color="#ff9100", lw=1.1, label="EMA21")
    ax.plot(x, d["VWAP"], color="#ffee58", lw=1, ls=":", label="VWAP")
    ax.plot(x, d["BBU"], color="#777", lw=.6, ls="--"); ax.plot(x, d["BBL"], color="#777", lw=.6, ls="--")
    lo, hi = d["Low"].min(), d["High"].max(); pad = (hi - lo) * .08
    ab, bl = level_list(S)
    for nm, v in ab + bl:
        if lo - pad <= v <= hi + pad and nm not in ("VWAP", "EMA21 (15د)", "EMA50 (15د)", "بولنجر العلوي", "بولنجر السفلي"):
            ax.axhline(v, color="#8888ff", lw=.7, ls="--", alpha=.7)
            ax.text(n - 0.5, v, f" {v:.2f}", color="#aaaaff", fontsize=7, va="center")
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="upper left", fontsize=7, facecolor=pan, edgecolor="#333", labelcolor="#ddd")
    ax.set_title(f"{S['label']}  15m  {S['price']:.2f}", color="white", fontsize=11)
    axv.bar(x, d["Volume"].values, color=col, width=.65); axv.set_ylabel("Vol", color="#ccc", fontsize=8)
    axr.plot(x, d["RSI"], color="#00ff88", lw=1.3)
    axr.axhline(70, color="#ff3366", ls="--", lw=.6); axr.axhline(30, color="#00ff88", ls="--", lw=.6)
    axr.set_ylim(0, 100); axr.set_ylabel("RSI", color="#ccc", fontsize=8)
    step = max(1, n // 8)
    axr.set_xticks(x[::step]); axr.set_xticklabels([t.strftime("%m-%d %H:%M") for t in d.index[::step]], rotation=0)
    plt.setp(ax.get_xticklabels(), visible=False); plt.setp(axv.get_xticklabels(), visible=False)
    ax.set_xlim(-1, n + 4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


# ============================== معالجات الأوامر ==============================
def head(S):
    return (f"<b>{S['label']}</b> ${S['price']:.2f} ({S['chg']:+.2f}%) · السوق: {S['sess_txt']}"
            + ("\n⚠️ <i>البيانات متأخرة عن السوق الحي</i>" if S["stale"] else ""))


def no_data(chat):
    tg_text("⚠️ تعذّر جلب بيانات السوق الآن (Yahoo/Finnhub). أعد المحاولة بعد دقيقة.", chat, keyboard())


def h_start(chat):
    tg_text("<b>👋 S&P 500 Specialist v2</b>\n\n"
            f"التركيز الحالي: <b>{state['focus']}</b>\n"
            "اضغط زراً أو اكتب سؤالك بحرية (مثال: <i>وين أقرب دعم؟ ليش الإشارة بيع؟ وش أثر الـ VIX؟</i>)\n\n"
            "<i>تحليل فني مبني على بيانات حية وقواعد شفافة، وليس توصية استثمارية.</i>", chat, keyboard())


def h_switch(chat):
    state["focus"] = "SPX" if state["focus"] == "SPY" else "SPY"
    tg_text(f"✅ التركيز الآن: <b>{state['focus']}</b>", chat, keyboard())


def h_price(chat):
    S = snapshot()
    if not S: return no_data(chat)
    l = S["l"]
    tg_text(f"💰 {head(S)}\n"
            f"📈 15د: {'صاعد' if l['EMA9'] > l['EMA21'] else 'هابط'} | RSI {l['RSI']:.0f} | "
            f"{'فوق' if l['Close'] > l['VWAP'] else 'تحت'} VWAP {l['VWAP']:.2f}\n"
            f"📐 {DIR_TXT[S['dir']]} ({S['score']:+d})", chat, keyboard())


def h_levels(chat):
    S = snapshot()
    if not S: return no_data(chat)
    ab, bl = level_list(S)
    fmt = lambda lst: "\n".join(f"  • {n} — <b>{v:.2f}</b> ({(v / S['price'] - 1) * 100:+.2f}%)" for n, v in lst) or "  —"
    tg_text(f"🎯 {head(S)}\n\n<b>فوق السعر</b> (الأقرب أولاً)\n{fmt(ab)}\n\n<b>تحت السعر</b>\n{fmt(bl)}\n\n"
            f"ATR(15د) = {S['atr']:.2f} → وقف 1.2×ATR ≈ {1.2 * S['atr']:.2f}$", chat, keyboard())


def h_analyze(chat, task=None, title="🧠"):
    S = snapshot()
    if not S:
        tg_text(fmt_ai(brain("اشرح للمستخدم بسطرين أن البيانات الحية غير متاحة وماذا يراقب عموماً في S&P 500.", None, chat, tokens=150)), chat, keyboard())
        return
    ans = brain(task or "اكتب تحليلاً مهنياً من 5 إلى 7 أسطر بالترتيب: (1) الصورة العامة والتعارض بين الفريمات إن وُجد "
                        "(2) الزخم والتشبع (3) أقرب دعمين ومقاومتين من البطاقة (4) السيناريو الأرجح وشرط إبطاله "
                        "(5) ما يجب مراقبته (VIX/خبر/مستوى).", card_text(S), chat, deep=True, tokens=650, memo="طلب تحليل شامل")
    out = f"{title} {head(S)}\n📐 {DIR_TXT[S['dir']]} ({S['score']:+d}/100)\n🤖 {html.escape(ml_line(S))}\n\n{fmt_ai(ans)}"
    if S.get("plan"):
        p = S["plan"]
        out += (f"\n\n🎯 <b>خطة افتراضية ({'شراء' if S['dir'] > 0 else 'بيع'})</b>: دخول {p['entry']:.2f} · وقف {p['stop']:.2f} · "
                f"هدف1 {p['t1']:.2f} · هدف2 {p['t2']:.2f}")
    tg_text(out, chat, keyboard())


def h_chart(chat):
    S = snapshot()
    if not S: return no_data(chat)
    img = chart_img(S)
    note = brain("اقرأ الشارت من البطاقة واكتب خلاصة في سطرين كحد أقصى مع أهم مستوى.", card_text(S, False), chat, tokens=160, memo="طلب الشارت")
    tg_photo(img, f"📊 {head(S)}\n\n💡 {fmt_ai(note)}", chat, keyboard())


def h_news(chat):
    nw = get_news(6)
    S = snapshot()
    if not nw:
        tg_text("📰 لا أخبار مالية حديثة متاحة الآن من المصادر. جرّب لاحقاً.", chat, keyboard()); return
    heads = "\n".join(f"- {x['h']} ({x['s']})" for x in nw)
    ans = brain("لخّص أثر هذه العناوين على S&P 500 في 3 أسطر: ما الداعم وما الضاغط وأي خبر أهم. لا تضف أخباراً من خارج القائمة.\n" + heads,
                card_text(S, False) if S else None, chat, tokens=260, memo="طلب أثر الأخبار")
    links = "\n".join(f"• <a href=\"{html.escape(x['u'], quote=True)}\">{html.escape(x['h'])}</a> <i>{html.escape(x['s'])}</i>" for x in nw[:5] if x["u"])
    tg_text(f"📰 <b>أخبار السوق</b>\n\n{fmt_ai(ans)}\n\n{links}", chat, keyboard())


def h_free(chat, text):
    S = snapshot()
    ans = brain(f"سؤال المستخدم: {text}\nأجب في 2 إلى 5 أسطر إلا إذا طلب تفصيلاً. إن كان السؤال خارج نطاق السوق فاعتذر بسطر.",
                card_text(S) if S else None, chat, deep=len(text) > 60, tokens=450, memo=text)
    tg_text(fmt_ai(ans), chat, keyboard())


ROUTES = {"start": h_start, "help": h_start, "switch": h_switch, "price": h_price, "levels": h_levels,
          "analyze": h_analyze, "chart": h_chart, "news": h_news}


def handle(text, chat):
    t = (text or "").strip()
    cmd = re.sub(r"@\w+$", "", t.lstrip("/").split()[0].lower()) if t else ""
    if (t.startswith("/") or t.lower() in ROUTES) and cmd in ROUTES:
        return ROUTES[cmd](chat)
    return h_free(chat, t)


# ============================== الويب هوك ==============================
def process_update(data):
    try:
        if "callback_query" in data:
            cq = data["callback_query"]; chat = str(cq["message"]["chat"]["id"]); text = cq.get("data", "")
            try: _tg("answerCallbackQuery", data={"callback_query_id": cq["id"]})
            except Exception: pass
        else:
            msg = data.get("message") or {}
            chat = str((msg.get("chat") or {}).get("id", "")); text = msg.get("text", "")
        if not chat or not text:
            return
        if ALLOWED_CHATS and chat not in ALLOWED_CHATS:
            log.info("blocked chat %s", chat); return
        try: _tg("sendChatAction", data={"chat_id": chat, "action": "typing"})
        except Exception: pass
        handle(text, chat)
    except Exception as e:
        log.exception("process_update")
        try: tg_text(f"❌ خطأ: {html.escape(str(e)[:80])}", locals().get("chat"), keyboard())
        except Exception: pass


@app.route("/webhook", methods=["POST"])
def webhook():
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        return "forbidden", 403
    data = request.get_json(force=True, silent=True) or {}
    uid = data.get("update_id")
    if uid is not None:
        if uid in _seen_updates:
            return jsonify({"ok": True})
        _seen_updates.append(uid)
    pool.submit(process_update, data)          # نرد فوراً حتى لا يعيد تيليجرام الإرسال
    return jsonify({"ok": True})


@app.route("/")
@app.route("/health")
def health():
    s, t = session_info()
    return jsonify({"status": "ok", "bot": "tdawll-v2", "focus": state["focus"], "session": s,
                    "uptime_min": int((time.time() - _started) / 60), "time_et": now_et().strftime("%H:%M:%S")})


def setup_webhook():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not (AUTO_WEBHOOK and TELEGRAM_TOKEN and url):
        return
    try:
        d = {"url": f"{url.rstrip('/')}/webhook", "allowed_updates": json.dumps(["message", "callback_query"])}
        if WEBHOOK_SECRET: d["secret_token"] = WEBHOOK_SECRET
        log.info("setWebhook -> %s", _tg("setWebhook", data=d).text[:120])
    except Exception as e:
        log.warning("setWebhook err: %s", e)


# ============================== المراقب الآلي ==============================
def monitor():
    time.sleep(8)
    setup_webhook()
    if NOTIFY_STARTUP:
        tg_text(f"✅ Specialist Online | {state['focus']}", TELEGRAM_CHAT_ID, keyboard())
    last = {"t": 0.0, "dir": 0}
    briefed = None
    while True:
        try:
            n = now_et(); sess, _ = session_info(n)
            if BRIEFING and n.weekday() < 5 and n.hour == 9 and n.minute < 25 and briefed != n.date():
                briefed = n.date()
                h_analyze(TELEGRAM_CHAT_ID, "إحاطة ما قبل الافتتاح (6 أسطر): ماذا حدث أمس، أين يقف السعر من المستويات اليومية، "
                          "أهم خبر/VIX، خطة السيناريوهين (صعود/هبوط) مع شرط الإبطال.", "🌅")
            if sess == "open":
                S = snapshot()
                if S and not S["stale"] and S["dir"] != 0 and abs(S["score"]) >= ALERT_SCORE:
                    ml = S["ml"]
                    contradicts = bool(ml and ml["reliable"] and ml["pred"] == -S["dir"])
                    fresh = (time.time() - last["t"] > 1800) or (S["dir"] != last["dir"] and time.time() - last["t"] > 600)
                    if fresh and not contradicts:
                        note = brain("تنبيه آلي: في سطرين حادّين — لماذا الإشارة الآن وأين شرط الإبطال.", card_text(S, False), None, tokens=140)
                        p = S["plan"]
                        msg = (f"🚨 <b>{S['label']}</b> ${S['price']:.2f} — {DIR_TXT[S['dir']]} ({S['score']:+d})\n{fmt_ai(note)}\n"
                               f"🎯 وقف {p['stop']:.2f} · هدف1 {p['t1']:.2f} · هدف2 {p['t2']:.2f}")
                        if tg_text(msg, TELEGRAM_CHAT_ID, keyboard()):
                            last = {"t": time.time(), "dir": S["dir"]}
        except Exception:
            log.exception("monitor")
        time.sleep(INTERVAL)


_booted = False
def boot():
    global _booted
    if _booted: return
    _booted = True
    threading.Thread(target=monitor, daemon=True).start()


boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
