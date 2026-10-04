# -*- coding: utf-8 -*-
"""
tdawll v3 — بوت تحليل S&P 500 (SPY / SPX) على تيليجرام
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
10) التزم بالطول المطلوب في الطلب.
11) عند تقييم «فرصة اقتناص»: الدخول والوقف والأهداف محسوبة بالكود فلا تغيّرها ولا تقترح بدائل. دورك تقييم السياق فقط (اتجاه يومي، VIX، أخبار، عوائق قريبة) وإعطاء حكم صريح، ولا تجامل: إن كان السياق ضد الفرصة فقل ذلك."""


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
        [{"text": "🎯 اقتناص الفرص", "callback_data": "scan"}, {"text": "🧠 تحليل ذكي", "callback_data": "analyze"}],
        [{"text": "📈 الشارت", "callback_data": "chart"}, {"text": "🎯 المستويات", "callback_data": "levels"}],
        [{"text": "💰 السعر", "callback_data": "price"}, {"text": "📰 الأخبار", "callback_data": "news"}],
        [{"text": "📊 الإحصائيات", "callback_data": "stats"}, {"text": "🔄 SPY ⇄ SPX", "callback_data": "switch"}],
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
    vm = v.rolling(20).mean()
    d["VOLR"] = np.where(vm > 0, v / (vm + 1e-9), 1.0)
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
    n = now_et()
    today_incomplete = d.index[-1].date() == n.date() and n.hour * 60 + n.minute < 960
    prev = d.iloc[-2] if today_incomplete else d.iloc[-1]       # آخر جلسة مكتملة
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


def dtrend_map(sym):
    """اتجاه اليومي لكل تاريخ (حسب إغلاق اليوم السابق، بدون تسريب مستقبلي)."""
    dd = candles(sym, "1d", "1y")
    if dd is None or len(dd) < 60:
        return {}
    c = dd["Close"]; s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    h200 = s200.notna()
    tr = pd.Series(0, index=dd.index)
    tr[(c > s50) & ((c > s200) | ~h200)] = 1
    tr[(c < s50) & ((c < s200) | ~h200)] = -1
    tr = tr.shift(1).fillna(0)
    return {ix.date(): int(v) for ix, v in tr.items()}


def snapshot(closed_only=False):
    label = state["focus"]; sym = SYMBOLS[label]
    raw = candles(sym, "15m", "60d")
    if raw is None or len(raw) < 120:
        return None
    if closed_only:                                  # نتجاهل الشمعة التي لم تُغلق بعد
        raw = raw[raw.index + pd.Timedelta(minutes=15) <= now_et().replace(tzinfo=None)]
        if len(raw) < 120:
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
    dmap = dtrend_map(sym)
    S = dict(label=label, sym=sym, d=d, l=l, D=D, macro=macro, ml=ml, score=score, why=why, dir=direction,
             dmap=dmap, dtrend=dmap.get(d.index[-1].date(), 0),
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


# ============================== محرك اقتناص الفرص (فريم 15 دقيقة) ==============================
GRADE_MIN = {"A": 75, "B": 62, "C": 48}
GRADE_RANK = {"A": 3, "B": 2, "C": 1}
SETUP_AR = {"ORB": "اختراق نطاق الافتتاح", "VWAP": "استعادة/رفض VWAP", "PULLBACK": "ارتداد من EMA21 مع الاتجاه",
            "SQUEEZE": "اختراق بعد ضغط التذبذب", "SWEEP": "كسر وهمي للسيولة + استعادة",
            "RETEST": "إعادة اختبار أعلى/أدنى أمس بعد الاختراق", "DIVERGENCE": "دايفرجنس RSI", "EXHAUST": "ارتداد من تطرف (بولنجر+RSI)"}
BASE_SCORE = {"ORB": 52, "VWAP": 48, "PULLBACK": 52, "SQUEEZE": 50, "SWEEP": 55, "RETEST": 52, "DIVERGENCE": 50, "EXHAUST": 45}
REVERSAL = {"SWEEP", "DIVERGENCE", "EXHAUST", "VWAP"}
DISABLED = {x.strip().upper() for x in os.environ.get("DISABLED_SETUPS", "").split(",") if x.strip()}
COST_R = 0.05                                            # تكلفة افتراضية (انزلاق/عمولة) بوحدة R
STATE_FILE = os.path.join(os.environ.get("DATA_DIR", "/tmp"), "tdawll_state.json")
settings = {"min_grade": os.environ.get("MIN_GRADE", "B").upper()}
live_sigs, price_alerts, BT = [], [], {}
_lock = threading.Lock()


def save_state():
    try:
        with _lock:
            blob = {"live_sigs": live_sigs[-300:], "price_alerts": price_alerts, "settings": settings, "focus": state["focus"]}
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
    except Exception as e:
        log.warning("save_state: %s", e)


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            b = json.load(f)
        live_sigs[:] = b.get("live_sigs", []); price_alerts[:] = b.get("price_alerts", [])
        settings.update(b.get("settings", {})); state["focus"] = b.get("focus", state["focus"])
        log.info("state loaded: %d signals, %d alerts", len(live_sigs), len(price_alerts))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("load_state: %s", e)


def prep(d):
    """يحوّل DataFrame المؤشرات إلى مصفوفات سريعة + جدول مستويات الأيام (سببي بالكامل)."""
    P = {k: d[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume", "RSI", "EMA9", "EMA21",
                                                  "EMA50", "BBU", "BBL", "ATR", "VOLR", "VWAP")}
    P["t"] = d.index
    P["date"] = np.array(d.index.date)
    P["mins"] = (d.index.hour * 60 + d.index.minute).values
    dates = P["date"]
    g = d.groupby(dates)
    day = pd.DataFrame({"h": g["High"].max(), "l": g["Low"].min(), "c": g["Close"].last()})
    day["prev_high"], day["prev_low"], day["prev_close"] = day["h"].shift(1), day["l"].shift(1), day["c"].shift(1)
    day["hi20"] = day["h"].shift(1).rolling(20, min_periods=5).max()
    day["lo20"] = day["l"].shift(1).rolling(20, min_periods=5).min()
    piv = (day["prev_high"] + day["prev_low"] + day["prev_close"]) / 3
    day["pivot"], day["r1"], day["s1"] = piv, 2 * piv - day["prev_low"], 2 * piv - day["prev_high"]
    orm = d[(P["mins"] >= 570) & (P["mins"] < 600)]
    og = orm.groupby(orm.index.date)
    day["orh"], day["orl"] = og["High"].max(), og["Low"].min()
    P["dayrows"] = day.to_dict("index")
    n = len(d)
    ds, de = np.zeros(n, int), np.zeros(n, int)
    s = 0
    for i in range(n):
        if i == 0 or dates[i] != dates[i - 1]: s = i
        ds[i] = s
    e = n - 1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or dates[i] != dates[i + 1]: e = i
        de[i] = e
    P["ds"], P["de"] = ds, de
    w = (d["BBU"] - d["BBL"]) / d["Close"]
    P["bbw_pct"] = w.rolling(100, min_periods=50).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True).values
    return P


def _pivots(arr, lo, hi, k, kind):
    res = []
    for j in range(max(lo, k), hi + 1):
        w = arr[j - k:j + k + 1]
        if len(w) < 2 * k + 1: continue
        if kind == "low" and arr[j] == w.min() and (w == arr[j]).sum() == 1: res.append(j)
        if kind == "high" and arr[j] == w.max() and (w == arr[j]).sum() == 1: res.append(j)
    return res


def detect(P, i, ctx):
    """يرصد المرشّحين عند إغلاق الشمعة رقم i. لا ينظر أبداً لأي شمعة بعدها."""
    if i < 60: return []
    m = int(P["mins"][i])
    if m < 600 or m >= 930: return []                    # لا إشارات أول 30 دقيقة ولا بعد 15:30
    O, H, L, C = P["Open"], P["High"], P["Low"], P["Close"]
    o, h, l, c = O[i], H[i], L[i], C[i]
    atr, vw, rsi = P["ATR"][i], P["VWAP"][i], P["RSI"]
    if not (ok(atr) and atr > 0 and ok(vw)): return []
    rng = max(h - l, 1e-9); body = abs(c - o); uw = h - max(o, c); lw = min(o, c) - l
    bull, bear, volr = c > o, c < o, P["VOLR"][i]
    row = P["dayrows"].get(P["date"][i], {}); ds = int(P["ds"][i])
    e9, e21, e50, r = P["EMA9"][i], P["EMA21"][i], P["EMA50"][i], rsi[i]
    out = []

    def add(name, dr, stop, why):
        if name not in DISABLED and ok(stop): out.append(dict(name=name, dir=dr, stop=float(stop), why=why))

    # 1) اختراق نطاق الافتتاح (أول 30 دقيقة)
    orh, orl = row.get("orh"), row.get("orl")
    if ok(orh) and ok(orl) and m <= 870 and ds + 2 <= i and P["mins"][ds] == 570:
        w = orh - orl
        if 0.8 * atr <= w <= 6 * atr and volr >= 1.1 and body >= 0.5 * rng:
            earlier = C[ds + 2:i]
            if bull and c > orh and not (earlier > orh).any(): add("ORB", 1, orh - 0.3 * w, f"أول إغلاق فوق {orh:.2f} (أعلى نطاق الافتتاح) بحجم {volr:.1f}x")
            if bear and c < orl and not (earlier < orl).any(): add("ORB", -1, orl + 0.3 * w, f"أول إغلاق تحت {orl:.2f} (أدنى نطاق الافتتاح) بحجم {volr:.1f}x")

    # 2) استعادة / رفض VWAP
    V = P["VWAP"]
    below = int((C[i - 5:i] < V[i - 5:i]).sum()); above = 5 - below
    if bull and below >= 3 and C[i - 1] < V[i - 1] and c > vw + 0.1 * atr and body >= 0.4 * rng:
        add("VWAP", 1, L[i - 3:i + 1].min() - 0.2 * atr, f"استعاد VWAP ({vw:.2f}) بعد {below} شموع تحته")
    if bear and above >= 3 and C[i - 1] > V[i - 1] and c < vw - 0.1 * atr and body >= 0.4 * rng:
        add("VWAP", -1, H[i - 3:i + 1].max() + 0.2 * atr, f"فقد VWAP ({vw:.2f}) بعد {above} شموع فوقه")

    # 3) ارتداد من EMA21 داخل اتجاه منظم
    if ok(e50) and 38 <= r <= 62:
        if e9 > e21 > e50 and e21 > P["EMA21"][i - 5] and L[i - 2:i + 1].min() <= e21 + 0.15 * atr and c > e21 and bull and c > H[i - 1]:
            add("PULLBACK", 1, L[i - 3:i + 1].min() - 0.3 * atr, f"لمس EMA21 ({e21:.2f}) وارتد بشمعة صاعدة تكسر قمة السابقة")
        if e9 < e21 < e50 and e21 < P["EMA21"][i - 5] and H[i - 2:i + 1].max() >= e21 - 0.15 * atr and c < e21 and bear and c < L[i - 1]:
            add("PULLBACK", -1, H[i - 3:i + 1].max() + 0.3 * atr, f"لمس EMA21 ({e21:.2f}) ورُفض بشمعة هابطة تكسر قاع السابقة")

    # 4) اختراق بعد ضغط التذبذب
    bp = P["bbw_pct"]
    if ok(bp[i - 1]) and bp[i - 1] <= 0.2 and (bp[i - 8:i] <= 0.25).sum() >= 5 and volr >= 1.3 and body >= 0.5 * rng:
        mid = (P["BBU"][i - 1] + P["BBL"][i - 1]) / 2
        if bull and c > H[i - 10:i].max(): add("SQUEEZE", 1, mid, f"كسر قمة 10 شموع بعد انضغاط بولنجر (حجم {volr:.1f}x)")
        if bear and c < L[i - 10:i].min(): add("SQUEEZE", -1, mid, f"كسر قاع 10 شموع بعد انضغاط بولنجر (حجم {volr:.1f}x)")

    # 5) كسر وهمي للسيولة ثم استعادة
    lows = [("أدنى أمس", row.get("prev_low")), ("أدنى 20 يوم", row.get("lo20")), ("S1", row.get("s1")), ("أدنى الافتتاح", orl)]
    highs = [("أعلى أمس", row.get("prev_high")), ("أعلى 20 يوم", row.get("hi20")), ("R1", row.get("r1")), ("أعلى الافتتاح", orh)]
    if rng >= 0.7 * atr and volr >= 0.9:
        for nm, lv in lows:
            if ok(lv) and l < lv - 0.05 * atr and c > lv and lw >= 0.5 * rng and c >= l + 0.55 * rng:
                add("SWEEP", 1, l - 0.2 * atr, f"سحب سيولة تحت {nm} ({lv:.2f}) وأغلق فوقه بذيل طويل"); break
        for nm, lv in highs:
            if ok(lv) and h > lv + 0.05 * atr and c < lv and uw >= 0.5 * rng and c <= h - 0.55 * rng:
                add("SWEEP", -1, h + 0.2 * atr, f"سحب سيولة فوق {nm} ({lv:.2f}) وأغلق تحته بذيل علوي طويل"); break

    # 6) إعادة اختبار أعلى/أدنى أمس بعد الاختراق
    ph, pl = row.get("prev_high"), row.get("prev_low")
    if ok(ph) and (C[i - 8:i - 1] > ph).any() and l <= ph + 0.2 * atr and c > ph + 0.05 * atr and bull:
        add("RETEST", 1, min(l, ph) - 0.4 * atr, f"اختراق أعلى أمس ({ph:.2f}) ثم إعادة اختبار ناجحة")
    if ok(pl) and (C[i - 8:i - 1] < pl).any() and h >= pl - 0.2 * atr and c < pl - 0.05 * atr and bear:
        add("RETEST", -1, max(h, pl) + 0.4 * atr, f"كسر أدنى أمس ({pl:.2f}) ثم إعادة اختبار فاشلة")

    # 7) دايفرجنس RSI
    L_, H_ = L, H
    pls = _pivots(L_, i - 40 + 3, i - 3, 3, "low")
    if len(pls) >= 2:
        p1, p2 = pls[-2], pls[-1]
        if p2 - p1 >= 5 and p2 >= i - 10 and L[p2] < L[p1] and rsi[p2] > rsi[p1] + 3 and rsi[p1] < 42 and bull and c > H[i - 1]:
            add("DIVERGENCE", 1, L[p2] - 0.3 * atr, f"قاع أدنى بسعر مع قاع أعلى في RSI ({rsi[p1]:.0f}→{rsi[p2]:.0f})")
    phs = _pivots(H_, i - 40 + 3, i - 3, 3, "high")
    if len(phs) >= 2:
        p1, p2 = phs[-2], phs[-1]
        if p2 - p1 >= 5 and p2 >= i - 10 and H[p2] > H[p1] and rsi[p2] < rsi[p1] - 3 and rsi[p1] > 58 and bear and c < L[i - 1]:
            add("DIVERGENCE", -1, H[p2] + 0.3 * atr, f"قمة أعلى بسعر مع قمة أدنى في RSI ({rsi[p1]:.0f}→{rsi[p2]:.0f})")

    # 8) ارتداد من تطرف بولنجر + RSI
    if l <= P["BBL"][i] and c > P["BBL"][i] and r <= 35 and lw >= 0.5 * rng:
        add("EXHAUST", 1, l - 0.2 * atr, f"لمس بولنجر السفلي ورفضه مع RSI {r:.0f}")
    if h >= P["BBU"][i] and c < P["BBU"][i] and r >= 65 and uw >= 0.5 * rng:
        add("EXHAUST", -1, h + 0.2 * atr, f"لمس بولنجر العلوي ورفضه مع RSI {r:.0f}")
    return out


def finalize(P, i, ctx, cand):
    dr, name = cand["dir"], cand["name"]
    c, atr, m = P["Close"][i], P["ATR"][i], int(P["mins"][i])
    risk = (c - cand["stop"]) * dr
    if risk <= 0: return None
    risk = max(risk, 0.8 * atr)
    if risk > 2.5 * atr: return None                       # وقف منطقي بعيد جداً = هيكل سيء
    stop = c - dr * risk
    row = P["dayrows"].get(P["date"][i], {}); ds = int(P["ds"][i])
    lv = []
    for nm, key in (("أعلى أمس", "prev_high"), ("أدنى أمس", "prev_low"), ("أعلى 20 يوم", "hi20"), ("أدنى 20 يوم", "lo20"),
                    ("محوري", "pivot"), ("R1", "r1"), ("S1", "s1"), ("أعلى الافتتاح", "orh"), ("أدنى الافتتاح", "orl")):
        v = row.get(key)
        if ok(v): lv.append((nm, float(v)))
    if i > ds: lv += [("أعلى اليوم", float(P["High"][ds:i].max())), ("أدنى اليوم", float(P["Low"][ds:i].min()))]
    ahead = sorted([(abs(v - c), nm, v) for nm, v in lv if (v - c) * dr > 0])
    obstacle = next(((nm, v) for dist, nm, v in ahead if dist < 0.8 * risk), None)
    far = [x for x in ahead if x[0] >= 1.2 * risk]
    if far and far[0][0] <= 3 * risk: t1, t1n = far[0][2], far[0][1]
    else: t1, t1n = c + dr * 1.5 * risk, "1.5R"
    d1 = abs(t1 - c)
    nxt = [x for x in far if d1 + 0.5 * risk <= x[0] <= 5 * risk]
    if nxt: t2, t2n = nxt[0][2], nxt[0][1]
    else: t2, t2n = c + dr * max(2.5 * risk, d1 + risk), "2.5R"
    rr1, rr2 = d1 / risk, abs(t2 - c) / risk

    s, notes = BASE_SCORE[name], [cand["why"]]
    rev = name in REVERSAL
    dt = ctx.get("dmap", {}).get(P["date"][i], ctx.get("dtrend", 0))
    if dt == dr: s += 12; notes.append("يتوافق مع الاتجاه اليومي")
    elif dt == -dr:
        s -= 5 if rev else 12; notes.append("⚠️ يعاكس الاتجاه اليومي")
    e9, e21, e50 = P["EMA9"][i], P["EMA21"][i], P["EMA50"][i]
    if ok(e50):
        if (dr == 1 and e9 > e21 > e50) or (dr == -1 and e9 < e21 < e50): s += 8; notes.append("ترتيب EMA على 15د معه")
        elif not rev and ((dr == 1 and e9 < e21 < e50) or (dr == -1 and e9 > e21 > e50)): s -= 6; notes.append("⚠️ ترتيب EMA على 15د ضده")
    if name != "VWAP": s += 5 if (c > P["VWAP"][i]) == (dr == 1) else -3
    v = P["VOLR"][i]
    if v >= 1.5: s += 8; notes.append(f"حجم قوي {v:.1f}x")
    elif v >= 1.2: s += 4
    elif v < 0.7: s -= 6; notes.append("⚠️ حجم ضعيف")
    o_, h_, l_ = P["Open"][i], P["High"][i], P["Low"][i]
    rg = max(h_ - l_, 1e-9)
    eng = (dr == 1 and c > o_ and P["Close"][i - 1] < P["Open"][i - 1] and c >= P["Open"][i - 1] and o_ <= P["Close"][i - 1]) or \
          (dr == -1 and c < o_ and P["Close"][i - 1] > P["Open"][i - 1] and c <= P["Open"][i - 1] and o_ >= P["Close"][i - 1])
    pin = (dr == 1 and (min(o_, c) - l_) >= 0.5 * rg and c > o_) or (dr == -1 and (h_ - max(o_, c)) >= 0.5 * rg and c < o_)
    if eng or pin: s += 5; notes.append("شمعة تأكيد (ابتلاع/ذيل رفض)")
    vix = ctx.get("vix")
    if vix:
        if dr == 1 and vix > 25: s -= 10; notes.append(f"⚠️ VIX مرتفع {vix:.1f}")
        if dr == -1 and vix < 14: s -= 6; notes.append(f"⚠️ VIX هادئ جداً {vix:.1f}")
    if rr1 >= 1.5: s += 4
    if rr2 >= 3: s += 4
    if obstacle: s -= 10; notes.append(f"⚠️ عائق قريب: {obstacle[0]} {obstacle[1]:.2f}")
    if 690 <= m < 810 and name in ("ORB", "SQUEEZE"): s -= 8; notes.append("⚠️ ساعات تذبذب الغداء")
    r = P["RSI"][i]
    if not rev and ((dr == 1 and r > 72) or (dr == -1 and r < 28)): s -= 8; notes.append(f"⚠️ RSI متطرف {r:.0f}")
    ml = ctx.get("ml")
    if ml and ml.get("reliable") and ml["pred"] != 0:
        if ml["pred"] == dr: s += 5; notes.append("نموذج ML الموثوق يوافق")
        else: s -= 10; notes.append("⚠️ نموذج ML الموثوق يعاكس")
    if ctx.get("use_bt"):
        st = BT.get("stats", {}).get(name)
        if st and st["n"] >= 20 and st["avgR"] < -0.15: s -= 12; notes.append("⚠️ هذا السيناريو سلبي في الباكتست التاريخي")
    s = int(max(0, min(100, s)))
    grade = "A" if s >= GRADE_MIN["A"] else "B" if s >= GRADE_MIN["B"] else "C" if s >= GRADE_MIN["C"] else None
    return dict(name=name, dir=dr, entry=float(c), stop=float(stop), t1=float(t1), t2=float(t2), t1n=t1n, t2n=t2n,
                risk=float(risk), rr1=float(rr1), rr2=float(rr2), score=s, grade=grade, notes=notes,
                ts=str(P["t"][i]), status="open", R=0.0)


def signals_at(P, i, ctx):
    res = []
    for cd in detect(P, i, ctx):
        sg = finalize(P, i, ctx, cd)
        if sg and sg["grade"]: res.append(sg)
    return sorted(res, key=lambda x: -x["score"])


def walk_outcome(sg, H, L, C, complete):
    dr = sg["dir"]
    for h, l in zip(H, L):
        if (l <= sg["stop"]) if dr == 1 else (h >= sg["stop"]): return "SL", -1.0 - COST_R     # عند التعارض داخل الشمعة نفترض الوقف أولاً
        if (h >= sg["t1"]) if dr == 1 else (l <= sg["t1"]): return "TP1", sg["rr1"] - COST_R
    if complete and len(C): return "EXP", (C[-1] - sg["entry"]) * dr / sg["risk"] - COST_R
    return None


def summarize(trades):
    def agg(rs):
        a = np.array(rs, float)
        w, ls = a[a > 0].sum(), -a[a < 0].sum()
        return dict(n=len(a), win=float((a > 0).mean()), avgR=float(a.mean()), totR=float(a.sum()),
                    pf=float(w / ls) if ls > 0 else float("inf"))
    by, byg = {}, {}
    for t in trades:
        by.setdefault(t["name"], []).append(t["R"]); byg.setdefault(t["grade"], []).append(t["R"])
    return ({k: agg(v) for k, v in by.items()}, {k: agg(v) for k, v in byg.items()}, agg([t["R"] for t in trades]) if trades else None)


def run_backtest(d, ctx):
    P = prep(d); n = len(d); trades, cool = [], {}
    for i in range(60, n - 1):
        for sg in signals_at(P, i, ctx):
            key = (sg["name"], sg["dir"])
            if i - cool.get(key, -99) < 8: continue
            cool[key] = i
            e = int(P["de"][i]); complete = P["mins"][e] >= 945
            res = walk_outcome(sg, P["High"][i + 1:e + 1], P["Low"][i + 1:e + 1], P["Close"][i + 1:e + 1], complete)
            if res: trades.append(dict(name=sg["name"], grade=sg["grade"], R=res[1]))
    return trades


def refresh_backtest():
    sym = SYMBOLS[state["focus"]]
    raw = candles(sym, "15m", "60d")
    if raw is None or len(raw) < 300: return False
    d = add_indicators(raw).dropna(subset=["RSI", "MACD_H", "EMA21", "BBU", "STOCH", "ATR"])
    ctx = {"dmap": dtrend_map(sym), "dtrend": 0, "vix": None, "ml": None, "use_bt": False}
    t0 = time.time(); trades = run_backtest(d, ctx)
    st, gr, allr = summarize(trades)
    BT.clear(); BT.update(stats=st, grades=gr, all=allr, days=int(len(set(d.index.date))), ts=time.time(), symbol=state["focus"])
    log.info("backtest %s: %d trades in %.1fs", state["focus"], len(trades), time.time() - t0)
    return True


def perf_line(name):
    s = BT.get("stats", {}).get(name)
    if not s or s["n"] < 5: return None
    return f"📊 تاريخياً ({BT['days']} يوم): {s['n']} صفقة · فوز {s['win'] * 100:.0f}% · متوسط {s['avgR']:+.2f}R"


def make_ctx(S):
    v = S["macro"].get("vix")
    return {"dmap": S["dmap"], "dtrend": S["dtrend"], "vix": v[0] if v else None, "ml": S["ml"], "use_bt": True}


def watchlist(S, P):
    i = len(S["d"]) - 1; l = S["l"]; atr = S["atr"]; c = S["price"]; out = []
    bp = P["bbw_pct"][i]
    if ok(bp) and bp <= 0.15:
        out.append(f"🔸 ضغط تذبذب شديد: توقّع اختراقاً. راقب أعلى {P['High'][i - 9:i + 1].max():.2f} / أدنى {P['Low'][i - 9:i + 1].min():.2f} آخر 10 شموع")
    ab, bl = level_list(S)
    for nm, v in (ab[:1] + bl[:1]):
        if abs(v - c) <= 0.5 * atr: out.append(f"🔸 السعر ملاصق لـ {nm} ({v:.2f}): انتظر رد الفعل (كسر أو رفض)")
    e9, e21, e50 = l["EMA9"], l["EMA21"], l["EMA50"]
    if ok(e50) and abs(c - e21) <= 0.5 * atr:
        if e9 > e21 > e50: out.append(f"🔸 تراجع نحو EMA21 ({e21:.2f}) داخل اتجاه صاعد: ابحث عن شمعة انعكاس صاعدة")
        if e9 < e21 < e50: out.append(f"🔸 صعود نحو EMA21 ({e21:.2f}) داخل اتجاه هابط: ابحث عن شمعة رفض")
    if l["RSI"] >= 70: out.append(f"🔸 RSI {l['RSI']:.0f} تشبع شرائي: لا تطارد القمة، انتظر رفضاً")
    if l["RSI"] <= 30: out.append(f"🔸 RSI {l['RSI']:.0f} تشبع بيعي: لا تطارد القاع، انتظر ارتداداً")
    return out


def signal_text(sg, S, ai=None, others=None):
    side = "شراء 🟢" if sg["dir"] > 0 else "بيع 🔴"
    ts = pd.Timestamp(sg["ts"])
    out = [f"🎯 <b>فرصة {sg['grade']}</b> · {S['label']} · فريم 15د · جودة {sg['score']}/100",
           f"<b>{SETUP_AR[sg['name']]}</b> — {side}",
           f"دخول ≈ <b>{sg['entry']:.2f}</b> (إغلاق شمعة {ts:%H:%M} ET)",
           f"وقف <b>{sg['stop']:.2f}</b> (مخاطرة {sg['risk']:.2f}$)",
           f"هدف1 <b>{sg['t1']:.2f}</b> ({sg['rr1']:.1f}R{'' if sg['t1n'].endswith('R') else ' · ' + html.escape(sg['t1n'])}) · هدف2 <b>{sg['t2']:.2f}</b> ({sg['rr2']:.1f}R)",
           "✅ " + html.escape("؛ ".join(sg["notes"]))]
    pl = perf_line(sg["name"])
    if pl: out.append(pl)
    if others: out.append("➕ أخرى بنفس الشمعة: " + html.escape("، ".join(f"{SETUP_AR[o['name']]} {'شراء' if o['dir'] > 0 else 'بيع'} ({o['grade']})" for o in others)))
    if ai: out.append("\n🧠 " + fmt_ai(ai))
    return "\n".join(out)


def sig_task(sg, others):
    conflict = [o for o in (others or []) if o["dir"] != sg["dir"]]
    return ("فرصة اقتناص رُصدت بالكود على فريم 15 دقيقة. الأرقام التالية محسوبة وثابتة ولا تغيّرها:\n"
            f"السيناريو: {SETUP_AR[sg['name']]} | الاتجاه: {'شراء' if sg['dir'] > 0 else 'بيع'} | الجودة {sg['score']}/100 ({sg['grade']})\n"
            f"دخول {sg['entry']:.2f} | وقف {sg['stop']:.2f} | هدف1 {sg['t1']:.2f} ({sg['t1n']}) | هدف2 {sg['t2']:.2f}\n"
            f"أسباب الكود: {'؛ '.join(sg['notes'])}\n"
            + ("تنبيه: توجد إشارة معاكسة بنفس الشمعة.\n" if conflict else "") +
            "اكتب سطرين: (1) هل السياق (الاتجاه اليومي/VIX/الأخبار/المستويات القريبة) يدعم الفرصة أم يعارضها؟ "
            "(2) متى لا يجوز الدخول أو أين تُبطَل. ثم سطراً أخيراً يبدأ بـ «الحكم:» مع ✅ أو ⚠️ أو ⛔ وسبب بكلمات قليلة.")


def live_scan(force_chat=None):
    S = snapshot(closed_only=True)
    if not S or S["stale"]: return 0
    d = S["d"]; P = prep(d); i = len(d) - 1
    minr = GRADE_RANK.get(settings["min_grade"], 2)
    sigs = [s for s in signals_at(P, i, make_ctx(S)) if GRADE_RANK[s["grade"]] >= minr]
    fresh = []
    for s in sigs:
        dup = any(x["name"] == s["name"] and x["dir"] == s["dir"] and
                  abs((pd.Timestamp(s["ts"]) - pd.Timestamp(x["ts"])).total_seconds()) < 7200 for x in live_sigs[-60:])
        if not dup: fresh.append(s)
    if not fresh: return 0
    best, others = fresh[0], fresh[1:]
    ai = brain(sig_task(best, others), card_text(S, True), None, deep=False, tokens=260)
    cap = f"🎯 {S['label']} · {SETUP_AR[best['name']]} · {'شراء' if best['dir'] > 0 else 'بيع'} ({best['grade']})"
    tg_photo(chart_img(S, plan=best), cap, TELEGRAM_CHAT_ID)
    tg_text(signal_text(best, S, ai, others), TELEGRAM_CHAT_ID, keyboard())
    with _lock: live_sigs.extend(fresh)
    save_state()
    return len(fresh)


def track_outcomes(closed):
    changed = False
    for sg in list(live_sigs):
        if sg["status"] != "open": continue
        ts = pd.Timestamp(sg["ts"])
        bars = closed[(closed.index > ts) & (closed.index.date == ts.date())]
        if bars.empty:
            if closed.index[-1].date() > ts.date(): sg["status"] = "EXP"; sg["R"] = 0.0; changed = True
            continue
        lastm = bars.index[-1].hour * 60 + bars.index[-1].minute
        res = walk_outcome(sg, bars["High"].values, bars["Low"].values, bars["Close"].values, lastm >= 945)
        if res:
            sg["status"], sg["R"] = res; changed = True
            ico = "✅" if sg["R"] > 0 else "🛑"
            lab = {"SL": "ضرب الوقف", "TP1": "تحقق الهدف1", "EXP": "إغلاق نهاية الجلسة"}[res[0]]
            tg_text(f"{ico} <b>نتيجة</b>: {SETUP_AR[sg['name']]} {'شراء' if sg['dir'] > 0 else 'بيع'} ({sg['grade']}) — {lab} · <b>{sg['R']:+.2f}R</b>",
                    TELEGRAM_CHAT_ID, keyboard())
    if changed: save_state()


def check_price_alerts(raw):
    if not price_alerts: return
    price = float(raw["Close"].iloc[-1]); hit = []
    for a in list(price_alerts):
        if (a["dir"] == "up" and price >= a["level"]) or (a["dir"] == "down" and price <= a["level"]): hit.append(a)
    for a in hit:
        tg_text(f"🔔 <b>{state['focus']}</b> وصل {a['level']:.2f} (السعر الآن {price:.2f})", a["chat"], keyboard())
        with _lock: price_alerts.remove(a)
    if hit: save_state()



# ============================== الشارت (matplotlib صافي) ==============================
def chart_img(S, bars=70, plan=None):
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
    if plan:
        for nm, v, colr in (("ENTRY", plan["entry"], "#ffffff"), ("STOP", plan["stop"], "#ff3366"),
                            ("T1", plan["t1"], "#00ff88"), ("T2", plan["t2"], "#00cc66")):
            ax.axhline(v, color=colr, lw=1.1, ls="-." if nm != "ENTRY" else "-")
            ax.text(n + 0.3, v, f"{nm} {v:.2f}", color=colr, fontsize=8, va="center", ha="left")
            lo, hi = min(lo, v), max(hi, v)
        pad = (hi - lo) * .06
        ax.scatter([n - 1], [plan["entry"]], marker="^" if plan["dir"] > 0 else "v", s=90, color="#ffee58", zorder=5)
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
    ax.set_xlim(-1, n + 9)
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
    tg_text("<b>👋 S&P 500 Specialist v3 — صائد الفرص</b>\n\n"
            f"التركيز: <b>{state['focus']}</b> · الفريم: 15 دقيقة · الحد الأدنى للتنبيه: <b>{settings['min_grade']}</b>\n\n"
            "أراقب كل شمعة 15د عند إغلاقها وأرسل لك الفرصة فور رصدها مع الدخول والوقف والأهداف وحكم الذكاء.\n\n"
            "<b>الأوامر</b>\n/scan — الفرص النشطة الآن + ما يتشكل\n/stats — أداء الإشارات والباكتست\n"
            "/alert 620 — تنبيه عند سعر\n/alerts — تنبيهاتك · /clear — حذفها\n/grade A|B|C — حدّ جودة التنبيهات\n"
            "/backtest — إعادة قياس السيناريوهات\n/analyze /chart /levels /price /news /switch\n\n"
            "أو اسألني بحرية.\n<i>تحليل فني آلي وليس توصية استثمارية. الأداء السابق لا يضمن المستقبل.</i>", chat, keyboard())


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


def h_scan(chat):
    S = snapshot(closed_only=True)
    if not S: return no_data(chat)
    d = S["d"]; P = prep(d); n = len(d); ctx = make_ctx(S)
    sess, _ = session_info()
    complete = sess != "open" or P["mins"][n - 1] >= 945
    active = []
    for k in range(3):                                     # آخر 3 شموع مغلقة
        i = n - 1 - k
        for sg in signals_at(P, i, ctx):
            if walk_outcome(sg, P["High"][i + 1:], P["Low"][i + 1:], P["Close"][i + 1:], complete) is None:
                active.append((k, sg))
    if sess != "open": active = []
    active.sort(key=lambda x: -x[1]["score"])
    wl = watchlist(S, P)
    lines = [f"🎯 <b>اقتناص {S['label']}</b> · {head(S)}"]
    if active:
        lines.append("\n<b>فرص نشطة</b>")
        for k, sg in active[:4]:
            lines.append(f"• <b>{sg['grade']} {sg['score']}</b> {SETUP_AR[sg['name']]} — {'شراء' if sg['dir'] > 0 else 'بيع'} (قبل {k * 15}د)\n"
                         f"   دخول {sg['entry']:.2f} · وقف {sg['stop']:.2f} · هدف1 {sg['t1']:.2f} ({sg['rr1']:.1f}R) · هدف2 {sg['t2']:.2f}")
    else:
        lines.append("\nلا توجد فرصة نشطة تحقق الشروط الآن. الانتظار صفقة." if sess == "open" else "\nالسوق مغلق: لا فرص حية.")
    if wl: lines.append("\n<b>قيد التشكل</b>\n" + "\n".join(html.escape(x) for x in wl))
    brief = "\n".join(f"{sg['grade']} {SETUP_AR[sg['name']]} {'شراء' if sg['dir'] > 0 else 'بيع'} دخول {sg['entry']:.2f} وقف {sg['stop']:.2f} هدف {sg['t1']:.2f}: {'؛ '.join(sg['notes'][:3])}"
                      for _, sg in active[:3]) or "لا فرص نشطة. قيد التشكل: " + (" | ".join(wl) if wl else "لا شيء")
    ai = brain("بناءً على هذه القائمة (محسوبة بالكود، لا تغيّر أرقامها) أعطني توجيهاً عملياً في 3 أسطر: أي فرصة أفضّل ولماذا، "
               "وإن لم توجد فما الذي أنتظره بالضبط (مستوى/شرط)؟\n" + brief, card_text(S, True), chat, tokens=300, memo="طلب مسح الفرص")
    lines.append("\n🧠 " + fmt_ai(ai))
    pl = [perf_line(sg["name"]) for _, sg in active[:1] if perf_line(sg["name"])]
    if pl: lines.append(pl[0])
    tg_text("\n".join(lines), chat, keyboard())


def h_stats(chat):
    L = ["📊 <b>الإحصائيات</b>"]
    done = [x for x in live_sigs if x["status"] != "open"]
    if done:
        a = np.array([x["R"] for x in done])
        L.append(f"\n<b>الإشارات الحية المسجّلة</b>: {len(done)} · فوز {(a > 0).mean() * 100:.0f}% · متوسط {a.mean():+.2f}R · مجموع {a.sum():+.1f}R")
        for g in ("A", "B", "C"):
            r = [x["R"] for x in done if x["grade"] == g]
            if r: L.append(f"   {g}: {len(r)} · فوز {(np.array(r) > 0).mean() * 100:.0f}% · {np.mean(r):+.2f}R")
    else:
        L.append("\nلا توجد إشارات حية مكتملة بعد.")
    if BT.get("stats"):
        L.append(f"\n<b>باكتست {BT['days']} يوم ({BT['symbol']}) — شموع 15د</b>")
        for k, v in sorted(BT["stats"].items(), key=lambda kv: -kv[1]["avgR"]):
            pf = "∞" if v["pf"] == float("inf") else f"{v['pf']:.2f}"
            L.append(f"• {SETUP_AR[k]}: {v['n']} · فوز {v['win'] * 100:.0f}% · {v['avgR']:+.2f}R · PF {pf}")
        g = BT.get("grades", {})
        if g: L.append("\nحسب الدرجة: " + " | ".join(f"{k}: {v['n']} صفقة {v['avgR']:+.2f}R" for k, v in sorted(g.items())))
        if BT.get("all"): L.append(f"الإجمالي: {BT['all']['n']} صفقة · فوز {BT['all']['win'] * 100:.0f}% · {BT['all']['avgR']:+.2f}R للصفقة")
        L.append("\n<i>الباكتست يفترض الدخول عند إغلاق الشمعة والخروج عند الوقف أو الهدف1 أو نهاية الجلسة، مع خصم تكلفة 0.05R. "
                 "الاتجاه اليومي فيه سببي، لكن VIX وML غير مُضمَّنين. عيّنة 60 يوماً قصيرة: اعتبرها مؤشراً لا دليلاً.</i>")
    else:
        L.append("\nالباكتست لم يكتمل بعد، أرسل /backtest.")
    tg_text("\n".join(L), chat, keyboard())


def h_backtest(chat):
    tg_text("⏳ أعيد قياس السيناريوهات على آخر 60 يوماً…", chat)
    if refresh_backtest(): h_stats(chat)
    else: tg_text("⚠️ لا بيانات كافية للباكتست الآن.", chat, keyboard())


def h_alert(chat, arg=""):
    m = re.search(r"\d+(?:[.,]\d+)?", arg or "")
    if not m:
        return tg_text("اكتب مثلاً: <code>/alert 620.5</code> وسأنبهك عند وصول السعر إليه.", chat, keyboard())
    lvl = float(m.group().replace(",", "."))
    raw = candles(SYMBOLS[state["focus"]], "15m", "60d")
    if raw is None: return no_data(chat)
    price = float(raw["Close"].iloc[-1])
    with _lock: price_alerts.append({"level": lvl, "dir": "up" if lvl > price else "down", "chat": str(chat)})
    save_state()
    tg_text(f"🔔 تم. سأنبهك عندما يصل {state['focus']} إلى <b>{lvl:.2f}</b> (الآن {price:.2f}).", chat, keyboard())


def h_alerts(chat):
    mine = [a for a in price_alerts if a["chat"] == str(chat)]
    tg_text("🔔 تنبيهاتك:\n" + ("\n".join(f"• {a['level']:.2f} ({'فوق' if a['dir'] == 'up' else 'تحت'})" for a in mine) or "لا يوجد."), chat, keyboard())


def h_clear(chat):
    with _lock: price_alerts[:] = [a for a in price_alerts if a["chat"] != str(chat)]
    save_state(); tg_text("🗑️ تم حذف تنبيهات الأسعار.", chat, keyboard())


def h_grade(chat, arg=""):
    g = (arg or "").strip().upper()[:1]
    if g not in GRADE_RANK:
        return tg_text(f"الحد الحالي: <b>{settings['min_grade']}</b>. اكتب /grade A (قليلة وقوية) أو B (متوازنة) أو C (كثيرة).", chat, keyboard())
    settings["min_grade"] = g; save_state()
    tg_text(f"✅ سأرسل الفرص من درجة <b>{g}</b> فأعلى.", chat, keyboard())


ROUTES = {"scan": h_scan, "stats": h_stats, "backtest": h_backtest, "alerts": h_alerts, "clear": h_clear, "start": h_start, "help": h_start, "switch": h_switch, "price": h_price, "levels": h_levels,
          "analyze": h_analyze, "chart": h_chart, "news": h_news}


ARG_ROUTES = {"alert": h_alert, "grade": h_grade}


def handle(text, chat):
    t = (text or "").strip()
    parts = t.lstrip("/").split(None, 1)
    cmd = re.sub(r"@\w+$", "", parts[0].lower()) if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    is_cmd = t.startswith("/") or t.lower() in ROUTES
    if is_cmd and cmd in ARG_ROUTES:
        return ARG_ROUTES[cmd](chat, arg)
    if is_cmd and cmd in ROUTES:
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
    return jsonify({"status": "ok", "bot": "tdawll-v3", "focus": state["focus"], "session": s,
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
SCAN_POLL = int(os.environ.get("SCAN_POLL", "45"))


def bt_loop():
    time.sleep(15)
    while True:
        try: refresh_backtest()
        except Exception: log.exception("backtest")
        time.sleep(6 * 3600)


def monitor():
    time.sleep(8)
    setup_webhook()
    if NOTIFY_STARTUP:
        tg_text(f"✅ Specialist Online | {state['focus']}", TELEGRAM_CHAT_ID, keyboard())
    threading.Thread(target=bt_loop, daemon=True).start()
    last_bar, briefed = None, None
    while True:
        try:
            n = now_et(); sess, _ = session_info(n)
            if BRIEFING and n.weekday() < 5 and n.hour == 9 and n.minute < 25 and briefed != n.date():
                briefed = n.date()
                h_analyze(TELEGRAM_CHAT_ID, "إحاطة ما قبل الافتتاح (6 أسطر): ماذا حدث في الجلسة الماضية، أين يقف السعر من المستويات اليومية، "
                          "أهم خبر/VIX، خطة السيناريوهين (صعود/هبوط) مع شرط الإبطال.", "🌅")
            if sess == "open" or (sess == "after" and n.hour == 16 and n.minute < 20):
                raw = candles(SYMBOLS[state["focus"]], "15m", "60d")
                if raw is not None and len(raw):
                    closed = raw[raw.index + pd.Timedelta(minutes=15) <= n.replace(tzinfo=None)]
                    check_price_alerts(raw)
                    if len(closed):
                        track_outcomes(closed)
                        if sess == "open" and closed.index[-1] != last_bar:
                            last_bar = closed.index[-1]
                            log.info("new closed bar %s -> scan", last_bar)
                            live_scan()
        except Exception:
            log.exception("monitor")
        time.sleep(SCAN_POLL)


_booted = False
def boot():
    global _booted
    if _booted: return
    _booted = True
    load_state()
    threading.Thread(target=monitor, daemon=True).start()


boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
