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
try:
    import websocket
except ImportError:
    websocket = None
from tda import llm, sources, agents, optflow, optstrat, analytics, xmarket
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tdawll")
app = Flask(__name__)

# ============================== الإعدادات ==============================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "").strip()          # @اسم_القناة أو -100xxxxxxxxxx
PUBLIC_BOT = os.environ.get("PUBLIC_BOT", "0") == "1"           # افتراضياً: البوت لك وحدك. 1 = يسمح لأي زائر بأوامر القراءة فقط
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
ALLOWED_CHATS = {x.strip() for x in os.environ.get("ALLOWED_CHAT_IDS", TELEGRAM_CHAT_ID).split(",") if x.strip()}
BRIEFING = os.environ.get("MORNING_BRIEFING", "1") == "1"     # إحاطة قبل الافتتاح
NOTIFY_STARTUP = os.environ.get("NOTIFY_STARTUP", "0") == "1"
AUTO_WEBHOOK = os.environ.get("AUTO_WEBHOOK", "1") == "1"

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
_live_lock = threading.Lock()
_live_bars = {}
_live_ws = {"state": "disabled", "last_trade": None, "error": ""}
_data_meta = {"source": "unknown", "last_ok": None, "last_error": "", "bars": 0}
_scan_status = {"at": None, "result": "never", "reason": "لم تبدأ دورة الفحص بعد", "symbol": "SPY"}
BAR_MINUTES = 5
PRIMARY_LIVE_SYMBOL = "SPY"

SYSTEM = """أنت «تداول»، محلل مؤسسي متخصص في مؤشر S&P 500 (SPY / SPX).
قواعد لا تُكسر:
1) كل رقم تذكره يجب أن يكون موجوداً في «البطاقة الفنية» المرفقة. ممنوع اختلاق أسعار أو مستويات أو أخبار.
2) إذا كان السوق مغلقاً فالبطاقة تحوي آخر بيانات حقيقية: حلّل آخر إغلاق وجهّز خطة للافتتاح. لا تقل «لا توجد بيانات» ما دامت البطاقة موجودة.
3) إذا تعارض الاتجاه على 5 دقائق مع الاتجاه اليومي فاذكر ذلك صراحة، فهو أهم معلومة في القراءة.
4) إن كان نموذج ML موسوماً «غير موثوق» فلا تبنِ عليه؛ اعتمد على درجة القواعد والسياق وقل ذلك بسطر.
5) في كل قراءة: الاتجاه، الزخم/التشبع، أقرب دعم ومقاومة، وشرط إبطال الفكرة (الوقف).
6) الأخبار: استخدم العناوين المرفقة فقط، وفرّق بين ما يدعم الاتجاه وما يعاكسه. لا أخبار مرفقة = لا تتحدث عن أخبار.
7) لا تقل «اشترِ» أو «بع». قل «الفنّي يرجّح…» وأرفق المخاطرة.
8) أسلوب عربي حاد ومباشر كمتداول محترف، بلا مقدمات ولا وعظ ولا اعتذارات.
9) نص عادي. يمكنك **تمييز** كلمات قليلة فقط. بدون عناوين أو جداول.
10) التزم بالطول المطلوب في الطلب.
11) عند تقييم «فرصة اقتناص»: الدخول والوقف والأهداف محسوبة بالكود فلا تغيّرها ولا تقترح بدائل. دورك تقييم السياق فقط (اتجاه يومي، VIX، أخبار، عوائق قريبة) وإعطاء حكم صريح، ولا تجامل: إن كان السياق ضد الفرصة فقل ذلك.
12) OPTIONS_STATE نموذج مبني على مخزون OI المتأخر وبافتراض أن المحترفين طويلو كول وقصيرو بوت. لا تدّعِ معرفة من اشترى أو باع ولا إن كانت الصفقات فتحت أو أغلقت. جدران الغاما وانقلابها سياق وليست دعماً ومقاومة سحرية، وبياناتها متأخرة ~5 دقائق. ومصدر مفقود يعني غير متاح لا صمت."""


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


_tl = threading.local()                       # هل المستخدم الحالي مشرف؟ (يحدد شكل الأزرار)
_last_tg_err = {"v": ""}


def _mid(r):
    try:
        return int(r.json()["result"]["message_id"])
    except Exception:
        return True


def isid(x):
    return isinstance(x, int) and not isinstance(x, bool)


def tg_text(text, chat_id=None, kb=None, reply_to=None):
    """يرجع رقم الرسالة (int) عند النجاح، وإلا False."""
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id or not text:
        return False
    for mode in ("HTML", None):
        try:
            body = text[:4000] if mode else re.sub(r"<[^>]+>", "", html.unescape(text))[:4000]
            d = {"chat_id": chat_id, "text": body, "disable_web_page_preview": True}
            if mode: d["parse_mode"] = mode
            if kb: d["reply_markup"] = json.dumps(kb)
            if reply_to: d["reply_to_message_id"] = reply_to; d["allow_sending_without_reply"] = True
            r = _tg("sendMessage", data=d)
            if r.status_code == 200:
                return _mid(r)
            _last_tg_err["v"] = f"{r.status_code}: {r.text[:200]}"
            log.warning("tg_text %s: %s", r.status_code, r.text[:150])
        except Exception as e:
            _last_tg_err["v"] = str(e)[:200]
            log.warning("tg_text err: %s", e)
    return False


def tg_photo(img, cap, chat_id=None, kb=None, reply_to=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return False
    for mode in ("HTML", None):
        try:
            c = cap[:1000] if mode else re.sub(r"<[^>]+>", "", html.unescape(cap))[:1000]
            d = {"chat_id": chat_id, "caption": c}
            if mode: d["parse_mode"] = mode
            if kb: d["reply_markup"] = json.dumps(kb)
            if reply_to: d["reply_to_message_id"] = reply_to; d["allow_sending_without_reply"] = True
            r = _tg("sendPhoto", files={"photo": ("c.png", img, "image/png")}, data=d)
            if r.status_code == 200:
                return _mid(r)
            _last_tg_err["v"] = f"{r.status_code}: {r.text[:200]}"
            log.warning("tg_photo %s: %s", r.status_code, r.text[:150])
        except Exception as e:
            _last_tg_err["v"] = str(e)[:200]
            log.warning("tg_photo err: %s", e)
    return False


_warn = {"t": 0.0}


def warn_owner(msg):
    """ينبّه المالك في الخاص (مرة كل 30 دقيقة كحد أقصى) عند فشل النشر في القناة."""
    if not TELEGRAM_CHAT_ID or time.time() - _warn["t"] < 1800:
        return
    _warn["t"] = time.time()
    tg_text(msg, TELEGRAM_CHAT_ID, keyboard())


def tg_doc(data, name, cap, chat_id=None, kb=None):
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return False
    try:
        d = {"chat_id": chat_id, "caption": cap[:900], "parse_mode": "HTML"}
        if kb: d["reply_markup"] = json.dumps(kb)
        r = _tg("sendDocument", files={"document": (name, data, "text/html")}, data=d)
        if r.status_code != 200: log.warning("tg_doc %s: %s", r.status_code, r.text[:150])
        return r.status_code == 200
    except Exception as e:
        log.warning("tg_doc err: %s", e)
        return False


def is_owner():
    return getattr(_tl, "owner", True)


def keyboard():
    """مضغوط: زران فقط. كل الأزرار الأخرى مجمّعة تحت زر «الأزرار»."""
    return {"inline_keyboard": [[{"text": "📋 جميع الأوامر", "callback_data": "menu_cmds"},
                                 {"text": "🎛️ الأزرار", "callback_data": "menu_panel"}]]}


def panel_keyboard():
    if is_owner():
        rows = [
            [{"text": "🎯 اقتناص الفرص", "callback_data": "scan"}, {"text": "🧠 تحليل ذكي", "callback_data": "analyze"}],
            [{"text": "📈 الشارت", "callback_data": "chart"}, {"text": "🎯 المستويات", "callback_data": "levels"}],
            [{"text": "💰 السعر", "callback_data": "price"}, {"text": "📰 الأخبار", "callback_data": "news"}],
            [{"text": "⚖️ لجنة التداول", "callback_data": "debate"}, {"text": "🌐 المزاج والماكرو", "callback_data": "sentiment"}],
            [{"text": "📉 حالة الأوبشن", "callback_data": "flow"}, {"text": "🧩 هياكل أوبشن", "callback_data": "options"}],
            [{"text": "📊 الإحصائيات", "callback_data": "stats"}, {"text": "🔬 الأفضلية", "callback_data": "edge"}],
            [{"text": "🔄 SPY ⇄ SPX", "callback_data": "switch"}, {"text": "📣 تحكم القناة", "callback_data": "ch_menu"}],
        ]
    else:
        rows = [[{"text": "💰 السعر", "callback_data": "price"}, {"text": "🎯 المستويات", "callback_data": "levels"}],
                [{"text": "📊 الإحصائيات", "callback_data": "stats"}]]
    rows.append([{"text": "⬅️ إخفاء الأزرار", "callback_data": "menu_hide"}])
    return {"inline_keyboard": rows}


def channel_keyboard():
    on, sl = settings.get("channel_on", True), settings.get("post_sl", True)
    return {"inline_keyboard": [
        [{"text": f"📣 النشر في القناة: {'مفعّل ✅' if on else 'متوقف ⛔'}", "callback_data": "ch_toggle"}],
        [{"text": f"🛑 نشر ضرب الوقف: {'نعم ✅' if sl else 'لا ⛔'}", "callback_data": "ch_sl"}],
        [{"text": "🧪 رسالة اختبار", "callback_data": "ch_test"}],
        [{"text": "⬅️ رجوع للأزرار", "callback_data": "menu_panel"}],
    ]}


def in_channel():
    return bool(CHANNEL_ID and settings.get("channel_on", True))


def chan():
    """وجهة التوصيات: القناة فقط إن كانت مفعّلة، وإلا محادثتك الخاصة."""
    return CHANNEL_ID if in_channel() else TELEGRAM_CHAT_ID


def dest_kb(dest):
    """أزرار الخاص فقط. القناة بلا أي أزرار أو روابط."""
    return None if (CHANNEL_ID and dest == CHANNEL_ID) else keyboard()


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


def _overlay_live_bar(df, symbol):
    """يضيف آخر شمعة مجمّعة من WebSocket فوق تاريخ Finnhub إن توفرت."""
    if df is None or symbol != PRIMARY_LIVE_SYMBOL:
        return df
    with _live_lock:
        b = dict(_live_bars.get(symbol) or {})
    if not b or not b.get("ts"):
        return df
    ix = pd.Timestamp(b["ts"])
    row = {k: float(b[k]) for k in ("Open", "High", "Low", "Close", "Volume")}
    out = df.copy()
    if ix in out.index:
        for k, v in row.items(): out.loc[ix, k] = v
    else:
        out.loc[ix, list(row)] = list(row.values())
    return out.sort_index()


def finnhub_candles(symbol, res="5", days=12):
    """المصدر الرئيسي للشموع: Finnhub؛ جودة اللحظة تعتمد على صلاحية FINNHUB_KEY."""
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
        return _overlay_live_bar(pd.DataFrame({"Open": j["o"], "High": j["h"], "Low": j["l"], "Close": j["c"], "Volume": j["v"]}, index=idx), symbol)
    except Exception as e:
        log.warning("finnhub candle err: %s", e)
        return None


def _live_trade_message(_, message):
    try:
        obj = json.loads(message)
        if obj.get("type") != "trade": return
        with _live_lock:
            for tick in obj.get("data", []):
                if tick.get("s") != PRIMARY_LIVE_SYMBOL: continue
                price, volume, stamp = float(tick["p"]), float(tick.get("v", 0) or 0), float(tick["t"]) / 1000
                _record_live_tick(price, volume, stamp, "finnhub_ws")
    except Exception as e:
        log.debug("finnhub websocket message: %s", e)


def _record_live_tick(price, volume=0.0, stamp=None, source="finnhub"):
    stamp = float(stamp or time.time())
    dt = datetime.fromtimestamp(stamp, NY).replace(tzinfo=None)
    minute = (dt.minute // BAR_MINUTES) * BAR_MINUTES
    ix = dt.replace(minute=minute, second=0, microsecond=0)
    with _live_lock:
        b = _live_bars.get(PRIMARY_LIVE_SYMBOL)
        if not b or b["ts"] != ix:
            b = {"ts": ix, "Open": price, "High": price, "Low": price, "Close": price, "Volume": 0.0}
            _live_bars[PRIMARY_LIVE_SYMBOL] = b
        b["High"] = max(b["High"], price); b["Low"] = min(b["Low"], price)
        b["Close"] = price; b["Volume"] += volume
        _live_ws["last_trade"] = stamp; _live_ws["state"] = f"{source}_connected"; _live_ws["error"] = ""


def _live_quote_loop():
    """مصدر مجاني قريب من اللحظي: Finnhub Quote كل 15 ثانية، ويستمر لو فشل WebSocket."""
    if not FINNHUB_KEY:
        return
    while True:
        try:
            r = requests.get("https://finnhub.io/api/v1/quote",
                             params={"symbol": PRIMARY_LIVE_SYMBOL, "token": FINNHUB_KEY}, timeout=10)
            q = r.json() if r.ok else {}
            price, stamp = q.get("c"), q.get("t") or time.time()
            if price and float(price) > 0:
                _record_live_tick(float(price), 0.0, stamp, "finnhub_quote")
            else:
                with _live_lock: _live_ws["error"] = f"quote HTTP {r.status_code} أو بلا سعر"
        except Exception as e:
            with _live_lock: _live_ws["error"] = str(e)[:180]
            log.warning("finnhub quote: %s", e)
        time.sleep(15)


def _live_ws_error(_, error):
    with _live_lock:
        _live_ws.update(state="error", error=str(error)[:180])
    log.warning("finnhub websocket: %s", error)


def _live_ws_loop():
    if not FINNHUB_KEY or websocket is None:
        with _live_lock: _live_ws["state"] = "unavailable"
        log.warning("live source unavailable: FINNHUB_KEY or websocket-client missing")
        return
    while True:
        try:
            with _live_lock: _live_ws["state"] = "connecting"
            ws = websocket.WebSocketApp(
                "wss://ws.finnhub.io?token=" + FINNHUB_KEY,
                on_open=lambda sock: (sock.send(json.dumps({"type": "subscribe", "symbol": PRIMARY_LIVE_SYMBOL})),
                                      log.info("Finnhub live WebSocket subscribed: %s", PRIMARY_LIVE_SYMBOL)),
                on_message=_live_trade_message, on_error=_live_ws_error)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            _live_ws_error(None, e)
        time.sleep(5)


def candles(symbol, interval="5m", rng="60d"):
    def go():
        # Finnhub + WebSocket هو الأساسي لفريم 5 دقائق؛ Yahoo آخر احتياط.
        if symbol == PRIMARY_LIVE_SYMBOL and interval == "5m":
            df = finnhub_candles(symbol, res="5", days=12)
            if df is not None:
                _data_meta.update(source="finnhub", last_ok=time.time(), last_error="", bars=len(df))
                return df
        df = yahoo_candles(symbol, interval, rng)
        if df is not None:
            _data_meta.update(source="yahoo_fallback", last_ok=time.time(), last_error="", bars=len(df))
        else:
            _data_meta.update(last_error=f"لا بيانات لـ {symbol} {interval}")
        return df
    return cached(f"c:{symbol}:{interval}:{rng}", 20 if interval != "1d" else 600, go)


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
    # خطوط الأساس لنفس الوقت من اليوم (سببية: الأيام السابقة فقط) — الافتتاح ليس كالغداء
    slot = d.index.strftime("%H:%M")
    base = lambda x: x.groupby(slot).transform(lambda s_: s_.shift(1).rolling(20, min_periods=5).median())
    vmed, rmed = base(v), base(h - l)
    d["RVT"] = np.where(vmed > 0, v / vmed.where(vmed > 0), np.nan)
    d["RGX"] = np.where(rmed > 0, (h - l) / rmed.where(rmed > 0), np.nan)
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
        s += 25; why.append("اتجاه 5د صاعد منظم (EMA9>21>50)")
    elif ok(e50) and e9 < e21 < e50:
        s -= 25; why.append("اتجاه 5د هابط منظم (EMA9<21<50)")
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


def snapshot(closed_only=False, label=None):
    label = label or state["focus"]; sym = SYMBOLS[label]
    raw = candles(sym, "5m", "12d")
    if raw is None or len(raw) < 120:
        return None
    if closed_only:                                  # نتجاهل الشمعة التي لم تُغلق بعد
        raw = raw[raw.index + pd.Timedelta(minutes=BAR_MINUTES) <= now_et().replace(tzinfo=None)]
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
             stale=(sess == "open" and age > 12), last_bar=d.index[-1])
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
    lv = [("VWAP", l["VWAP"]), ("EMA21 (5د)", l["EMA21"]), ("EMA50 (5د)", l["EMA50"]),
          ("بولنجر العلوي", l["BBU"]), ("بولنجر السفلي", l["BBL"])]
    if D:
        lv += [("أعلى أمس", D["prev_high"]), ("أدنى أمس", D["prev_low"]), ("إغلاق أمس", D["prev_close"]),
               ("أعلى 20 يوم", D["hi20"]), ("أدنى 20 يوم", D["lo20"]), ("محوري", D["pivot"]),
               ("R1", D["r1"]), ("S1", D["s1"]), ("SMA50 يومي", D["sma50"]), ("SMA200 يومي", D["sma200"])]
    cof = cached_of()
    if cof and cof.get("label") == S["label"]:
        lv += optflow.levels(cof)
    lv = [(n, float(v)) for n, v in lv if ok(v)]
    p = S["price"]
    above = sorted([x for x in lv if x[1] > p], key=lambda x: x[1])[:3]
    below = sorted([x for x in lv if x[1] <= p], key=lambda x: -x[1])[:3]
    return above, below


_of_cache = {"k": None, "v": None, "t": 0.0}
_xm_cache = {"k": None, "v": None, "t": 0.0}


def cached_of():
    return _of_cache["v"] if time.time() - _of_cache["t"] < 900 else None


def get_of(S):
    """حالة الأوبشن (CBOE المتأخرة) مخزّنة دقيقتين؛ تفشل بصمت وتعيد None."""
    k = S["label"]
    if _of_cache["k"] == k and time.time() - _of_cache["t"] < 120:
        return _of_cache["v"]
    try:
        OF = optflow.get(k, S["price"], S["D"]["prev_close"] if S.get("D") else None, now_et().replace(tzinfo=None))
    except Exception:
        log.exception("get_of"); OF = None
    _of_cache.update(k=k, v=OF, t=time.time())
    return OF


def get_xm(S):
    k = S["label"]
    if _xm_cache["k"] == k and time.time() - _xm_cache["t"] < 120:
        return _xm_cache["v"]
    try:
        XM = xmarket.compute(lambda s_, i_, r_: candles(s_, i_, r_), S["d"])
    except Exception:
        log.exception("get_xm"); XM = None
    _xm_cache.update(k=k, v=XM, t=time.time())
    return XM


def grade_of(sc):
    return "A" if sc >= 75 else "B" if sc >= 62 else "C" if sc >= 48 else None


def enrich(sigs, OF, XM):
    """يضيف سياق الأوبشن والأسواق المرتبطة بتعديل صغير ويُسجَّل في الإشارة لنقيسه لاحقاً (لا نصدّقه قبل القياس)."""
    for sg in sigs:
        a1, n1, reg = optflow.adjust(OF, sg) if OF else (0, [], None)
        a2, n2 = xmarket.adjust(XM, sg) if XM else (0, [])
        if a1 or a2: sg["score"] = int(max(0, min(100, sg["score"] + a1 + a2)))
        sg["notes"] = list(sg["notes"]) + n1 + n2
        sg["opt_regime"], sg["opt_adj"], sg["xm_adj"] = reg, int(a1), int(a2)
        sg["grade"] = grade_of(sg["score"])
    return sorted([x for x in sigs if x["grade"]], key=lambda x: -x["score"])


# ============================== البطاقة المُرسلة للذكاء ==============================
def card_text(S, with_news=True, ext=False):
    l, D, m = S["l"], S["D"], S["macro"]
    trend = "صاعد" if l["EMA9"] > l["EMA21"] else "هابط"
    n = now_et()
    lines = [
        f"NOW_ET={n:%Y-%m-%d %H:%M} ({n:%A}) | SESSION={S['sess_txt']} | LAST_BAR_ET={S['last_bar']:%Y-%m-%d %H:%M}"
        + (" | تحذير: البيانات متأخرة" if S["stale"] else ""),
        f"SYMBOL={S['label']} | PRICE={S['price']:.2f} | CHG_VS_PREV_CLOSE={S['chg']:+.2f}%",
        f"5M: TREND={trend} EMA9={l['EMA9']:.2f} EMA21={l['EMA21']:.2f} EMA50={l['EMA50']:.2f} | RSI={l['RSI']:.1f} "
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
    if ext:
        lines += sources.card_lines(sources.gather("SPY"))
        lines += optflow.card_lines(get_of(S))
        lines += xmarket.card_lines(get_xm(S))
    pv = pos_state["v"]
    if pv:
        lines.append("MY_POSITION: " + ("flat" if pv["side"] == "flat" else
                                         f"{pv['side']} من {pv['entry']:.2f}" + (f" وقف {pv['stop']:.2f}" if pv.get("stop") else "")))
    return "\n".join(lines)


# ============================== العقل (Gemini) ==============================
def brain(task, card=None, chat_id=None, deep=False, tokens=500, memo=None):
    if not llm.GEMINI_API_KEY:
        return "⚠️ مفتاح Gemini غير مضبوط."
    hist = chat_memory.get(chat_id, []) if chat_id else []
    contents = [{"role": "user" if m["r"] == "u" else "model", "parts": [{"text": m["c"]}]} for m in hist[-6:]]
    body = (f"[البطاقة الفنية الحية — المصدر الوحيد للأرقام]\n{card}\n\n" if card else
            "[لا توجد بيانات حية الآن. صرّح بذلك بسطر ولا تذكر أي أسعار أو مستويات محددة.]\n\n")
    contents.append({"role": "user", "parts": [{"text": body + "[المطلوب]\n" + task}]})
    txt = llm.generate(SYSTEM, contents, deep=deep, tokens=tokens)
    if txt:
        if chat_id:
            hist.append({"r": "u", "c": memo or task[:300]}); hist.append({"r": "a", "c": txt})
            chat_memory[chat_id] = hist[-12:]
        return txt
    return "❌ تعذّر توليد الإجابة (حد الاستخدام أو خلل مؤقت). جرّب بعد قليل."


# ============================== محرك اقتناص الفرص (فريم 5 دقائق) ==============================
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
settings = {"min_grade": os.environ.get("MIN_GRADE", "B").upper(), "gate": os.environ.get("AI_GATE", "soft").lower(), "account": None,
            "channel_on": True, "post_sl": True}
live_sigs, price_alerts, BT, lessons = [], [], {}, []
pos_state = {"v": None}
_lock = threading.Lock()


def save_state():
    try:
        with _lock:
            blob = {"live_sigs": live_sigs[-300:], "price_alerts": price_alerts, "settings": settings, "focus": state["focus"],
                    "lessons": lessons[-80:], "position": pos_state["v"]}
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
    except Exception as e:
        log.warning("save_state: %s", e)


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            b = json.load(f)
        live_sigs[:] = b.get("live_sigs", []); price_alerts[:] = b.get("price_alerts", [])
        lessons[:] = b.get("lessons", []); pos_state["v"] = b.get("position")
        settings.update(b.get("settings", {})); state["focus"] = b.get("focus", state["focus"])
        log.info("state loaded: %d signals, %d alerts", len(live_sigs), len(price_alerts))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("load_state: %s", e)


def prep(d):
    """يحوّل DataFrame المؤشرات إلى مصفوفات سريعة + جدول مستويات الأيام (سببي بالكامل)."""
    P = {k: d[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume", "RSI", "EMA9", "EMA21",
                                                  "EMA50", "BBU", "BBL", "ATR", "VOLR", "VWAP", "RVT", "RGX")}
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
    lv += [(nm, float(v)) for nm, v in ctx.get("opt_levels", []) if ok(v)]
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
        if (dr == 1 and e9 > e21 > e50) or (dr == -1 and e9 < e21 < e50): s += 8; notes.append("ترتيب EMA على 5د معه")
        elif not rev and ((dr == 1 and e9 < e21 < e50) or (dr == -1 and e9 > e21 > e50)): s -= 6; notes.append("⚠️ ترتيب EMA على 5د ضده")
    if name != "VWAP": s += 5 if (c > P["VWAP"][i]) == (dr == 1) else -3
    v = P["VOLR"][i]
    if v >= 1.5: s += 8; notes.append(f"حجم قوي {v:.1f}x")
    elif v >= 1.2: s += 4
    elif v < 0.7: s -= 6; notes.append("⚠️ حجم ضعيف")
    rvt = P["RVT"][i]
    if ok(rvt):
        if rvt >= 1.8: s += 3; notes.append(f"نشاط شاذ لنفس الوقت ({rvt:.1f}x وسيط الأيام السابقة)")
        elif rvt < 0.6: s -= 3; notes.append("⚠️ نشاط ضعيف مقارنةً بنفس الوقت سابقاً")
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


def walk_detail(sg, H, L, C, complete):
    dr = sg["dir"]
    for k, (h, l) in enumerate(zip(H, L)):
        if (l <= sg["stop"]) if dr == 1 else (h >= sg["stop"]): return "SL", -1.0 - COST_R, k     # عند التعارض داخل الشمعة نفترض الوقف أولاً
        if (h >= sg["t1"]) if dr == 1 else (l <= sg["t1"]): return "TP1", sg["rr1"] - COST_R, k
    if complete and len(C): return "EXP", (C[-1] - sg["entry"]) * dr / sg["risk"] - COST_R, len(C) - 1
    return None


def walk_outcome(sg, H, L, C, complete):
    r = walk_detail(sg, H, L, C, complete)
    return (r[0], r[1]) if r else None


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
            H_, L_, C_ = P["High"][i + 1:e + 1], P["Low"][i + 1:e + 1], P["Close"][i + 1:e + 1]
            res = walk_outcome(sg, H_, L_, C_, complete)
            if res:
                tr = analytics.walk_trail(sg, H_, L_, C_, complete, float(P["ATR"][i]), COST_R)
                trades.append(dict(name=sg["name"], grade=sg["grade"], R=res[1], R_trail=tr[1] if tr else res[1], i=i,
                                   dir=sg["dir"], risk=sg["risk"], rr1=sg["rr1"], date=str(P["date"][i])))
    return trades, P


def refresh_backtest():
    sym = SYMBOLS[state["focus"]]
    raw = candles(sym, "5m", "12d")
    if raw is None or len(raw) < 300: return False
    d = add_indicators(raw).dropna(subset=["RSI", "MACD_H", "EMA21", "BBU", "STOCH", "ATR"])
    ctx = {"dmap": dtrend_map(sym), "dtrend": 0, "vix": None, "ml": None, "use_bt": False}
    t0 = time.time(); trades, P = run_backtest(d, ctx)
    st, gr, allr = summarize(trades)
    by = {}
    for t in trades: by.setdefault(t["name"], []).append(t)
    plc, trail = {}, {}
    for nm, ts in by.items():
        trail[nm] = float(np.mean([t["R_trail"] for t in ts]))
        if len(ts) >= 15:
            try: plc[nm] = analytics.placebo(P, ts, walk_outcome, n_sims=120)
            except Exception: log.exception("placebo %s", nm)
    Rs = [t["R"] for t in sorted(trades, key=lambda t: t["i"])]
    BT.clear(); BT.update(stats=st, grades=gr, all=allr, days=int(len(set(d.index.date))), ts=time.time(), symbol=state["focus"],
                          placebo=plc, trail=trail, metrics=analytics.metrics(Rs) if Rs else None,
                          mc=analytics.monte_carlo(Rs) if len(Rs) >= 8 else None, R=Rs[-400:],
                          trail_all=float(np.mean([t["R_trail"] for t in trades])) if trades else 0.0)
    log.info("backtest %s: %d trades in %.1fs", state["focus"], len(trades), time.time() - t0)
    return True


def perf_line(name):
    s = BT.get("stats", {}).get(name)
    if not s or s["n"] < 5: return None
    return f"📊 تاريخياً ({BT['days']} يوم): {s['n']} صفقة · فوز {s['win'] * 100:.0f}% · متوسط {s['avgR']:+.2f}R"


def make_ctx(S):
    v = S["macro"].get("vix")
    OF = cached_of()
    return {"dmap": S["dmap"], "dtrend": S["dtrend"], "vix": v[0] if v else None, "ml": S["ml"], "use_bt": True,
            "opt_levels": optflow.levels(OF) if OF and OF.get("label") == S["label"] else []}


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
    out = [f"🎯 <b>فرصة {sg['grade']}</b> · {S['label']} · فريم 5د · جودة {sg['score']}/100",
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
    return ("فرصة اقتناص رُصدت بالكود على فريم 5 دقائق. الأرقام التالية محسوبة وثابتة ولا تغيّرها:\n"
            f"السيناريو: {SETUP_AR[sg['name']]} | الاتجاه: {'شراء' if sg['dir'] > 0 else 'بيع'} | الجودة {sg['score']}/100 ({sg['grade']})\n"
            f"دخول {sg['entry']:.2f} | وقف {sg['stop']:.2f} | هدف1 {sg['t1']:.2f} ({sg['t1n']}) | هدف2 {sg['t2']:.2f}\n"
            f"أسباب الكود: {'؛ '.join(sg['notes'])}\n"
            + ("تنبيه: توجد إشارة معاكسة بنفس الشمعة.\n" if conflict else "") +
            "اكتب سطرين: (1) هل السياق (الاتجاه اليومي/VIX/الأخبار/المستويات القريبة) يدعم الفرصة أم يعارضها؟ "
            "(2) متى لا يجوز الدخول أو أين تُبطَل. ثم سطراً أخيراً يبدأ بـ «الحكم:» مع ✅ أو ⚠️ أو ⛔ وسبب بكلمات قليلة.")


def active_signals(S, lookback=3):
    """الإشارات الصالحة (لم تُحسم) على آخر `lookback` شموع مغلقة. تُستخدم في /scan و /debate."""
    d = S["d"]; P = prep(d); n = len(d)
    OF, XM = get_of(S), get_xm(S)
    ctx = make_ctx(S)
    sess, _ = session_info()
    complete = sess != "open" or P["mins"][n - 1] >= 945
    out = []
    if sess != "open": return out, P
    for k in range(lookback):
        i = n - 1 - k
        for sg in enrich(signals_at(P, i, ctx), OF, XM):
            if walk_outcome(sg, P["High"][i + 1:], P["Low"][i + 1:], P["Close"][i + 1:], complete) is None:
                out.append((k, sg))
    out.sort(key=lambda x: -x[1]["score"])
    return out, P


def _scan_set(result, reason, symbol=None):
    _scan_status.update(at=time.time(), result=result, reason=reason, symbol=symbol or state["focus"])


def live_scan(force_chat=None):
    S = snapshot(closed_only=True)
    if not S:
        _scan_set("blocked", "لا توجد بطاقة بيانات كافية")
        return 0
    if S["stale"]:
        _scan_set("blocked", "الشمعة الأخيرة قديمة؛ تم منع التوصية")
        return 0
    d = S["d"]; P = prep(d); i = len(d) - 1
    if len(d) < 120:
        _scan_set("blocked", f"بيانات غير كافية: {len(d)} شمعة فقط")
        return 0
    # سقف مجاني ومحافظ: لا نسمح بأكثر من ثلاث إشارات جديدة في جلسة واحدة.
    today = now_et().date()
    today_count = sum(1 for x in live_sigs if pd.Timestamp(x.get("ts", 0)).date() == today and x.get("ann"))
    if today_count >= int(os.environ.get("MAX_DAILY_SIGNALS", "3")):
        _scan_set("blocked", f"تم بلوغ سقف إشارات الجلسة ({today_count})")
        return 0
    minr = GRADE_RANK.get(settings["min_grade"], 2)
    OF, XM = get_of(S), get_xm(S)
    sigs = [s for s in enrich(signals_at(P, i, make_ctx(S)), OF, XM) if GRADE_RANK[s["grade"]] >= minr]
    fresh = []
    for s in sigs:
        dup = any(x["name"] == s["name"] and x["dir"] == s["dir"] and
                  abs((pd.Timestamp(s["ts"]) - pd.Timestamp(x["ts"])).total_seconds()) < 7200 for x in live_sigs[-60:])
        if not dup: fresh.append(s)
    if not fresh:
        _scan_set("empty", "لا توجد إشارة جديدة اجتازت حد الجودة والتكرار")
        return 0
    best, others = fresh[0], fresh[1:]

    # لجنة التداول: ثور/دب/حكم في طلب Gemini واحد، مع دروس سابقة وموقفك الحالي
    gate = settings.get("gate", "soft")
    cm = None
    if gate != "off":
        cm = agents.committee_lite(card_text(S, True, ext=True), S["label"], best,
                                   agents.lessons_context(lessons, best["name"]),
                                   agents.position_context(pos_state["v"], S["price"], S["atr"]))
    if cm:
        best["ai_rating"], best["ai_agree"], best["ai_conf"] = cm["rating"], cm["agree"], cm["confidence"]
        best["ai_decision"], best["ai_regime"] = cm.get("decision", "wait"), cm.get("regime", "unclear")
        best["ai_data_quality"] = cm.get("data_quality", "degraded")
    veto = bool(cm and cm["agree"] == -1)
    if cm and cm.get("data_quality") == "stale":
        veto = True
    if cm and cm.get("decision") in {"reject", "wait"} and cm.get("confidence") == "high":
        veto = True
    if veto and gate == "hard":                            # نسجّلها لنقيس: هل كان الفيتو في محله؟
        for x in fresh: x["sent"] = False; x["ann"] = False; x["done"] = False; x["label"] = S["label"]; x["ev"] = []
        best["vetoed"] = True
        with _lock: live_sigs.extend(fresh)
        save_state()
        log.info("alert vetoed by committee: %s %s", best["name"], best["dir"])
        _scan_set("vetoed", f"اللجنة حجبت الإشارة: {cm.get('decision', 'conflict') if cm else 'conflict'}")
        return 0
    txt = signal_text(best, S, None, others)
    if cm:
        txt += "\n\n" + agents.lite_block(cm)
        if veto: txt += "\n⛔ <b>اللجنة تعارض هذه الإشارة</b>: خفّض الحجم أو تجاهلها."
    else:
        txt += "\n\n🧠 " + fmt_ai(brain(sig_task(best, others), card_text(S, True), None, deep=False, tokens=260))
    dest = chan(); pub = dest == CHANNEL_ID and bool(CHANNEL_ID)
    if pub: txt += "\n\n⚠️ <i>تحليل فني آلي وليس توصية استثمارية. التزم بالوقف وإدارة المخاطر.</i>"
    side = "شراء" if best["dir"] > 0 else "بيع"
    cap = (f"🎯 <b>{S['label']}</b> · {SETUP_AR[best['name']]} · {side} ({best['grade']})\n"
           f"دخول {best['entry']:.2f} · وقف {best['stop']:.2f}\nهدف1 {best['t1']:.2f} · هدف2 {best['t2']:.2f}")
    pid = tg_photo(chart_img(S, plan=best), cap, dest)
    mid = tg_text(txt, dest, dest_kb(dest), reply_to=pid if isid(pid) else None)
    posted = bool(pid or mid)
    for x in fresh:
        x["label"] = S["label"]; x["dest"] = dest; x["ev"] = []; x["done"] = False; x["ann"] = False
    best["ann"] = posted
    best["msg_id"] = mid if isid(mid) else (pid if isid(pid) else None)
    if not posted:
        log.warning("signal NOT delivered to %s: %s", dest, _last_tg_err["v"])
        if dest != TELEGRAM_CHAT_ID:
            warn_owner("⚠️ <b>تعذّر نشر التوصية في القناة</b>\n" + html.escape(_last_tg_err["v"]) +
                       f"\n\nالتوصية (لم تُنشر): {SETUP_AR[best['name']]} {side} {S['label']} — دخول {best['entry']:.2f} · وقف {best['stop']:.2f} · "
                       f"هدف1 {best['t1']:.2f} · هدف2 {best['t2']:.2f}\nتأكد أن البوت <b>مشرف</b> في القناة ولديه صلاحية نشر الرسائل (/channel test).")
    with _lock: live_sigs.extend(fresh)
    save_state()
    _scan_set("posted" if posted else "send_failed", "تم نشر الإشارة" if posted else f"فشل الإرسال: {_last_tg_err['v']}")
    return len(fresh)


def _reflect_store(sg):
    """يحوّل نتيجة صفقة محسومة إلى درس قصير يُحقن في قرارات اللجنة القادمة (ذاكرة TradingAgents)."""
    try:
        rec = dict(setup=sg["name"], dir=sg["dir"], grade=sg["grade"], score=sg["score"], notes=sg.get("notes", []),
                   ai_rating=sg.get("ai_rating"), ai_agree=sg.get("ai_agree"), status=sg["status"], R=sg["R"],
                   mfe=sg.get("mfe", 0.0), mae=sg.get("mae", 0.0), bars=sg.get("bars", 0))
        text = agents.reflect(rec)
        if text:
            with _lock:
                lessons.append({"date": str(pd.Timestamp(sg["ts"]).date()), "setup": sg["name"], "dir": sg["dir"],
                                "grade": sg["grade"], "R": round(sg["R"], 2), "text": text})
            save_state()
    except Exception:
        log.exception("reflect")


def walk_events(sg, H, L):
    """أحداث الصفقة بالترتيب: T1 ثم T2 أو SL أو BE (عودة للدخول بعد T1). عند التعارض داخل الشمعة نفترض الأسوأ.
    نتجاهل شموعاً مدى حركتها شاذ (>6R) لأنها غالباً تسعيرة خاطئة ولا نريد إعلاناً كاذباً في قناة عامة."""
    dr, entry, stop, t1, t2 = sg["dir"], sg["entry"], sg["stop"], sg["t1"], sg["t2"]
    lim = 6 * sg["risk"]
    ev, stage = [], 0
    for k, (h, l) in enumerate(zip(H, L)):
        if h - l > lim: continue
        hit_t2 = (h >= t2) if dr == 1 else (l <= t2)
        if stage == 0:
            if (l <= stop) if dr == 1 else (h >= stop):
                ev.append(("SL", k)); return ev
            if (h >= t1) if dr == 1 else (l <= t1):
                ev.append(("T1", k)); stage = 1
                if hit_t2: ev.append(("T2", k)); return ev
            continue
        if (l <= entry) if dr == 1 else (h >= entry):
            ev.append(("BE", k)); return ev
        if hit_t2:
            ev.append(("T2", k)); return ev
    return ev


TITLE_EN = {"T1": "TARGET 1 REACHED", "T2": "TARGET 2 REACHED", "SL": "STOP HIT", "BE": "CLOSED AT BREAKEVEN", "EXP": "SESSION END"}


def result_text(sg, name, hit_ts, close=None, R=None):
    dr = sg["dir"]; side = "شراء 🟢" if dr > 0 else "بيع 🔴"
    head = f"<b>{sg.get('label', '')}</b> · {SETUP_AR[sg['name']]} · {side}"
    when = f"\n🕒 {pd.Timestamp(hit_ts):%H:%M} ET" if hit_ts is not None else ""
    if name == "T1":
        return (f"✅ <b>تم الوصول للهدف الأول</b> 🎯\n{head}\nدخول {sg['entry']:.2f} ← هدف1 <b>{sg['t1']:.2f}</b> (+{sg['rr1']:.1f}R){when}\n"
                f"💡 حرّك الوقف إلى الدخول {sg['entry']:.2f} (تعادل) وأبقِ الباقي نحو الهدف الثاني {sg['t2']:.2f}")
    if name == "T2":
        return (f"🏆 <b>تم الوصول للهدف الثاني</b> 🎯🎯\n{head}\nدخول {sg['entry']:.2f} ← هدف2 <b>{sg['t2']:.2f}</b> (+{sg['rr2']:.1f}R){when}\n"
                f"تحقق الهدفان ✅")
    if name == "SL":
        return (f"🛑 <b>ضرب الوقف</b>\n{head}\nدخول {sg['entry']:.2f} ← وقف <b>{sg['stop']:.2f}</b> (-1R){when}\n"
                f"لم تنجح هذه الإشارة؛ الالتزام بالوقف جزء من الخطة.")
    if name == "BE":
        return (f"⏹ <b>أُغلقت الصفقة عند التعادل</b>\n{head}\nعاد السعر إلى الدخول {sg['entry']:.2f} بعد تحقق الهدف الأول ✅ "
                f"(الربح الجزئي محقق){when}")
    return (f"⏹ <b>انتهت الجلسة</b>\n{head}\nإغلاق عند {close:.2f} ({R:+.2f}R)" +
            ("\nالهدف الأول كان قد تحقق ✅" if "T1" in sg.get("ev", []) else ""))


def announce(sg, name, hit_ts, close=None, R=None):
    """رسالة + صورة شارت (رداً على المنشور الأصلي). ضرب الوقف يُنشر افتراضياً حفاظاً على الشفافية مع المتابعين."""
    if name == "SL" and not settings.get("post_sl", True):
        return
    txt = result_text(sg, name, hit_ts, close, R)
    dest = sg.get("dest") or chan()
    reply = sg.get("msg_id") if isid(sg.get("msg_id")) else None
    img = None
    try:
        S = snapshot(closed_only=True, label=sg.get("label"))
        if S:
            plan = dict(entry=sg["entry"], stop=sg["stop"], t1=sg["t1"], t2=sg["t2"], dir=sg["dir"], ts=sg["ts"],
                        hits=list(sg.get("ev", [])), hit_ts=hit_ts, hit_name=name, title=TITLE_EN.get(name, ""))
            img = chart_img(S, plan=plan)
    except Exception:
        log.exception("announce chart")
    kb = dest_kb(dest)
    ok_ = tg_photo(img, txt, dest, kb, reply_to=reply) if img else tg_text(txt, dest, kb, reply_to=reply)
    if not ok_:
        log.warning("result NOT delivered (%s %s): %s", name, dest, _last_tg_err["v"])
        if dest != TELEGRAM_CHAT_ID:
            warn_owner(f"⚠️ تعذّر نشر نتيجة ({name}) في القناة: " + html.escape(_last_tg_err["v"]))


def track_outcomes(closed, live=None):
    """closed: شموع مغلقة (للإحصاء الدقيق). live: تشمل الشمعة الجارية (لإعلان الأهداف فور لمسها)."""
    changed = False
    live = live if live is not None else closed
    for sg in list(live_sigs):
        if sg.get("done", True) and sg["status"] != "open": continue
        if sg.get("label") and sg["label"] != state["focus"]: continue
        ts = pd.Timestamp(sg["ts"])
        cb = closed[(closed.index > ts) & (closed.index.date == ts.date())]
        if cb.empty and closed.index[-1].date() > ts.date():           # يوم جديد ولم نحسم: ننهيها بصمت
            if sg["status"] == "open": sg["status"] = "EXP"; sg["R"] = 0.0
            sg["done"] = True; changed = True
            continue
        # (1) الإحصاء على الشموع المغلقة فقط
        if sg["status"] == "open" and not cb.empty:
            lastm = cb.index[-1].hour * 60 + cb.index[-1].minute
            res = walk_detail(sg, cb["High"].values, cb["Low"].values, cb["Close"].values, lastm >= 945)
            if res:
                st, R, k = res
                Hh, Ll = cb["High"].values[:k + 1], cb["Low"].values[:k + 1]
                fav = (Hh.max() - sg["entry"]) if sg["dir"] == 1 else (sg["entry"] - Ll.min())
                adv = (sg["entry"] - Ll.min()) if sg["dir"] == 1 else (Hh.max() - sg["entry"])
                sg.update(status=st, R=R, mfe=float(max(0, fav) / sg["risk"]), mae=float(-max(0, adv) / sg["risk"]), bars=int(k + 1))
                changed = True
                if not sg.get("ann"): sg["done"] = True
                pool.submit(_reflect_store, dict(sg))
        # (2) إعلانات القناة
        if sg.get("ann") and not sg.get("done"):
            lb = live[(live.index > ts) & (live.index.date == ts.date())]
            evs = walk_events(sg, lb["High"].values, lb["Low"].values) if not lb.empty else []
            for name, k in evs:
                if name in sg.setdefault("ev", []): continue
                sg["ev"].append(name); changed = True
                announce(sg, name, lb.index[k])
            terminal = any(e in sg.get("ev", []) for e in ("SL", "T2", "BE"))
            if not terminal and not cb.empty and (cb.index[-1].hour * 60 + cb.index[-1].minute) >= 945:
                close = float(cb["Close"].iloc[-1]); R = (close - sg["entry"]) * sg["dir"] / sg["risk"]
                sg["ev"].append("EXP"); terminal = True; changed = True
                announce(sg, "EXP", cb.index[-1], close, R)
            if terminal: sg["done"] = True
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
        if lo - pad <= v <= hi + pad and nm not in ("VWAP", "EMA21 (5د)", "EMA50 (5د)", "بولنجر العلوي", "بولنجر السفلي"):
            ax.axhline(v, color="#8888ff", lw=.7, ls="--", alpha=.7)
            ax.text(n - 0.5, v, f" {v:.2f}", color="#aaaaff", fontsize=7, va="center")
    if plan:
        hits = set(plan.get("hits", []))
        for nm, v, colr, key in (("ENTRY", plan["entry"], "#ffffff", None), ("STOP", plan["stop"], "#ff3366", "SL"),
                                 ("T1", plan["t1"], "#00ff88", "T1"), ("T2", plan["t2"], "#00cc66", "T2")):
            hit = key in hits
            ax.axhline(v, color=colr, lw=2.4 if hit else 1.1, ls="-." if nm != "ENTRY" else "-")
            ax.text(n + 0.3, v, f"{nm} {v:.2f}" + (" REACHED" if hit and key != "SL" else " HIT" if hit else ""),
                    color=colr, fontsize=9 if hit else 8, fontweight="bold" if hit else "normal", va="center", ha="left")
            lo, hi = min(lo, v), max(hi, v)
        pad = (hi - lo) * .06
        pos = n - 1
        if plan.get("ts"):
            tsx = pd.Timestamp(plan["ts"]); ix = int(d.index.searchsorted(tsx))
            if ix < n and d.index[ix] == tsx: pos = ix
        ax.scatter([pos], [plan["entry"]], marker="^" if plan["dir"] > 0 else "v", s=90, color="#ffee58", zorder=5)
        if plan.get("hit_ts") is not None:
            hx = pd.Timestamp(plan["hit_ts"]); jx = int(d.index.searchsorted(hx))
            if jx < n and d.index[jx] == hx:
                lvl = {"T1": plan["t1"], "T2": plan["t2"], "SL": plan["stop"]}.get(plan.get("hit_name"), plan["entry"])
                ax.scatter([jx], [lvl], marker="*", s=260, color="#ffd54f", edgecolor="white", zorder=6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="upper left", fontsize=7, facecolor=pan, edgecolor="#333", labelcolor="#ddd")
    ax.set_title(f"{S['label']}  5m  {S['price']:.2f}" + (f"   |   {plan['title']}" if plan and plan.get("title") else ""), color="white", fontsize=11)
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


def commands_text(owner=True):
    if not owner:
        return ("📋 <b>الأوامر المتاحة لك</b>\n\n/price — السعر والحالة الآن\n/levels — أهم الدعوم والمقاومات\n"
                "/stats — أداء التوصيات المنشورة\n/commands — هذه القائمة\n\n"
                "<i>التوصيات تصل في القناة. هذا تحليل فني آلي وليس توصية استثمارية.</i>")
    return ("📋 <b>جميع الأوامر</b>\n\n"
            "<b>🎯 الفرص والتحليل</b>\n/scan — الفرص النشطة الآن + ما يتشكل\n/analyze — تحليل ذكي شامل\n/debate — لجنة كاملة + تقرير HTML\n"
            "/chart — الشارت · /levels — المستويات\n/price — السعر · /news — الأخبار\n/data_status — حالة المصدر اللحظي\n/why_no_signal — لماذا لم تصدر توصية؟\n\n"
            "<b>🌐 السياق</b>\n/sentiment — مزاج StockTwits/Reddit/Polymarket\n/macro — FRED والأسواق التنبؤية\n"
            "/flow — حالة الأوبشن (جدران الغاما والانقلاب)\n/options — هياكل أوبشن للفرصة\n\n"
            "<b>📊 القياس</b>\n/stats — الأداء الحي والباكتست\n/edge — هل لدينا أفضلية؟\n/backtest — إعادة القياس\n/memory — دروس اللجنة\n\n"
            "<b>🛡️ الإدارة</b>\n/risk 25000 — حجم المركز\n/pos long 618.5 stop 617 — صفقتك (أو flat / clear)\n"
            "/alert 620 — تنبيه سعر · /alerts · /clear\n/grade A|B|C — حد الجودة\n/gate soft|hard|off — صرامة اللجنة\n/switch — SPY ⇄ SPX\n\n"
            "<b>📣 القناة</b>\n/channel — الحالة وأزرار التحكم\n/channel on|off — تشغيل/إيقاف النشر\n/channel sl on|off — نشر ضرب الوقف\n"
            "/channel test — رسالة اختبار للقناة\n/panel — الأزرار هنا\n\n"
            "أو اسألني بحرية.")


def h_commands(chat, arg=""):
    tg_text(commands_text(is_owner()), chat, keyboard())


def h_data_status(chat, arg=""):
    with _live_lock: live = dict(_live_ws)
    age = (time.time() - live["last_trade"] if live.get("last_trade") else None)
    msg = (f"📡 <b>حالة البيانات</b>\nالمصدر الأساسي: Finnhub WebSocket + Quote كل 15 ثانية\n"
           f"الحالة: <b>{live.get('state')}</b>\n"
           f"آخر Tick: {f'قبل {age:.1f} ثانية' if age is not None else 'لا يوجد بعد'}\n"
           f"مصدر الشموع: <b>{_data_meta.get('source')}</b> · عددها {_data_meta.get('bars', 0)}\n"
           f"الاحتياطي: Yahoo Chart\nالفريم: <b>5 دقائق</b>")
    tg_text(msg, chat, keyboard())


def h_why_no_signal(chat, arg=""):
    st = _scan_status
    when = datetime.fromtimestamp(st["at"], NY).strftime("%H:%M:%S") if st.get("at") else "—"
    tg_text(f"🔎 <b>سبب آخر فحص</b> · {when}\nالحالة: <b>{st.get('result')}</b>\n{html.escape(st.get('reason', '—'))}", chat, keyboard())


def h_panel(chat, arg=""):
    tg_text("🎛️ <b>لوحة الأزرار</b>", chat, panel_keyboard())


def h_start(chat, arg=""):
    a = (arg or "").strip().lower()
    if a == "cmds": return h_commands(chat)
    if a == "panel": return h_panel(chat)
    if not is_owner():
        return tg_text("👋 <b>أهلاً بك</b>\nهذا البوت يرسل توصيات مؤشر S&P 500 على فريم 5 دقائق في القناة، "
                       "ويتابع تحقق الأهداف.\nيمكنك هنا معرفة السعر والمستويات وأداء التوصيات.\n\n"
                       "<i>تحليل فني آلي وليس توصية استثمارية.</i>", chat, keyboard())
    ch = (f"القناة: <b>{html.escape(CHANNEL_ID)}</b> ({'مفعّلة ✅' if in_channel() else 'متوقفة ⛔'})" if CHANNEL_ID
          else "القناة: غير مضبوطة (أضف CHANNEL_ID) — التوصيات تصلك هنا")
    tg_text("<b>👋 S&P 500 Specialist v3.4 — صائد الفرص</b>\n\n"
            f"التركيز: <b>{state['focus']}</b> · فريم 5 دقائق · حد التنبيه: <b>{settings['min_grade']}</b> · بوابة اللجنة: <b>{settings.get('gate', 'soft')}</b>\n"
            f"{ch}\n\nأراقب كل شمعة عند إغلاقها وأنشر التوصية بصورة الشارت في القناة، ثم أعلن تحقق الهدف الأول والثاني.\n\n"
            "اضغط <b>📋 جميع الأوامر</b> لقائمة الأوامر، أو <b>🎛️ الأزرار</b> لباقي الأزرار.\n"
            "<i>تحليل فني آلي وليس توصية استثمارية. الأداء السابق لا يضمن المستقبل.</i>", chat, keyboard())


def ch_test(chat):
    if not CHANNEL_ID:
        return tg_text("لم تضبط <code>CHANNEL_ID</code> بعد (راجع /channel).", chat, keyboard())
    S = snapshot(closed_only=True)
    cap = "🧪 <b>رسالة اختبار</b>\nإذا ظهرت هنا مع الصورة فالقناة جاهزة لاستقبال التوصيات ✅"
    r = tg_photo(chart_img(S), cap, CHANNEL_ID) if S else tg_text(cap, CHANNEL_ID)
    if r:
        tg_text("✅ وصلت رسالة الاختبار إلى القناة (بلا أزرار ولا روابط).", chat, channel_keyboard())
    else:
        tg_text("❌ <b>فشل الإرسال إلى القناة</b>\n" + html.escape(_last_tg_err["v"]) +
                "\n\nتأكد أن: (1) البوت <b>مشرف</b> في القناة (2) لديه صلاحية <b>نشر الرسائل</b> (3) المعرّف صحيح: "
                "<code>@اسم_القناة</code> للعامة، أو <code>-100…</code> للخاصة.", chat, channel_keyboard())


def h_channel(chat, arg=""):
    a = (arg or "").strip().lower()
    if not CHANNEL_ID:
        return tg_text("📣 <b>القناة غير مضبوطة</b>\n\n1) أنشئ قناة وأضف البوت <b>مشرفاً</b> بصلاحية نشر الرسائل (وتثبيتها)\n"
                       "2) في Render أضف متغيراً: <code>CHANNEL_ID</code> = <code>@اسم_القناة</code> (أو الرقم <code>-100…</code> للخاصة)\n"
                       "3) أعد النشر ثم أرسل <code>/channel test</code>\n\nالقناة تستقبل التوصيات والنتائج فقط، بلا أزرار ولا روابط. كل الأزرار هنا في خاصك.", chat, keyboard())
    if a in ("on", "off"):
        settings["channel_on"] = a == "on"; save_state()
    elif a.startswith("sl"):
        settings["post_sl"] = "off" not in a; save_state()
    elif a == "test":
        return ch_test(chat)
    last = next((x for x in reversed(live_sigs) if x.get("ann")), None)
    tg_text("📣 <b>القناة</b>\n"
            f"المعرّف: <code>{html.escape(CHANNEL_ID)}</code>\n"
            f"النشر: {'مفعّل ✅ (التوصيات في القناة فقط)' if in_channel() else 'متوقف ⛔ (التوصيات تصل لخاصك)'}\n"
            f"نشر ضرب الوقف: {'نعم ✅' if settings.get('post_sl', True) else 'لا ⛔'}\n"
            f"حد الجودة: <b>{settings['min_grade']}</b> · المنشورات بلا أزرار ولا روابط\n"
            + (f"آخر توصية: {SETUP_AR.get(last['name'], last['name'])} {'شراء' if last['dir'] > 0 else 'بيع'} ({last['grade']}) "
               f"{pd.Timestamp(last['ts']):%m-%d %H:%M} · الأحداث: {', '.join(last.get('ev', [])) or 'لا شيء بعد'}" if last else "لم تُنشر توصيات بعد.")
            + ("\n<i>ملاحظة: نشر ضرب الوقف مُفعّل افتراضياً لأن إخفاء الخسائر يضلّل المتابعين.</i>" if settings.get("post_sl", True) else
               "\n⚠️ <i>أنت لا تنشر ضرب الوقف؛ المتابعون سيرون النجاحات فقط. فكّر في الشفافية معهم.</i>"),
            chat, channel_keyboard())


def h_switch(chat):
    state["focus"] = "SPX" if state["focus"] == "SPY" else "SPY"
    tg_text(f"✅ التركيز الآن: <b>{state['focus']}</b>", chat, keyboard())


def h_price(chat):
    S = snapshot()
    if not S: return no_data(chat)
    l = S["l"]
    tg_text(f"💰 {head(S)}\n"
            f"📈 5د: {'صاعد' if l['EMA9'] > l['EMA21'] else 'هابط'} | RSI {l['RSI']:.0f} | "
            f"{'فوق' if l['Close'] > l['VWAP'] else 'تحت'} VWAP {l['VWAP']:.2f}\n"
            f"📐 {DIR_TXT[S['dir']]} ({S['score']:+d})", chat, keyboard())


def h_levels(chat):
    S = snapshot()
    if not S: return no_data(chat)
    ab, bl = level_list(S)
    fmt = lambda lst: "\n".join(f"  • {n} — <b>{v:.2f}</b> ({(v / S['price'] - 1) * 100:+.2f}%)" for n, v in lst) or "  —"
    tg_text(f"🎯 {head(S)}\n\n<b>فوق السعر</b> (الأقرب أولاً)\n{fmt(ab)}\n\n<b>تحت السعر</b>\n{fmt(bl)}\n\n"
            f"ATR(5د) = {S['atr']:.2f} → وقف 1.2×ATR ≈ {1.2 * S['atr']:.2f}$", chat, keyboard())


def h_analyze(chat, task=None, title="🧠"):
    S = snapshot()
    if not S:
        tg_text(fmt_ai(brain("اشرح للمستخدم بسطرين أن البيانات الحية غير متاحة وماذا يراقب عموماً في S&P 500.", None, chat, tokens=150)), chat, keyboard())
        return
    ans = brain(task or "اكتب تحليلاً مهنياً من 5 إلى 7 أسطر بالترتيب: (1) الصورة العامة والتعارض بين الفريمات إن وُجد "
                        "(2) الزخم والتشبع (3) أقرب دعمين ومقاومتين من البطاقة (4) السيناريو الأرجح وشرط إبطاله "
                        "(5) ما يجب مراقبته (VIX/خبر/مستوى). استخدم مزاج المتداولين والأسواق التنبؤية إن كانت متاحة.", card_text(S, True, ext=True), chat, deep=True, tokens=650, memo="طلب تحليل شامل")
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
                card_text(S, True, ext=True) if S else None, chat, deep=len(text) > 60, tokens=450, memo=text)
    tg_text(fmt_ai(ans), chat, keyboard())


def h_scan(chat):
    S = snapshot(closed_only=True)
    if not S: return no_data(chat)
    sess, _ = session_info()
    active, P = active_signals(S)
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
        grp = {1: "توافق اللجنة", 0: "محايدة", -1: "تعارض اللجنة"}
        ai_rows = []
        for k, lab in grp.items():
            r = [x["R"] for x in done if x.get("ai_agree") == k]
            if r: ai_rows.append(f"   {lab}: {len(r)} · فوز {(np.array(r) > 0).mean() * 100:.0f}% · {np.mean(r):+.2f}R")
        if ai_rows:
            L.append("\n<b>هل تضيف اللجنة قيمة؟</b> (الأداء حسب رأيها)\n" + "\n".join(ai_rows))
            L.append("<i>إذا كان «تعارض اللجنة» أسوأ من «توافق» بوضوح بعد عشرات الصفقات، فاللجنة تعمل وتستحق /gate hard.</i>")
        rg = {"POS": "غاما موجبة", "NEG": "غاما سالبة"}
        rows = []
        for k, lab in rg.items():
            r = [x["R"] for x in done if x.get("opt_regime") == k]
            if r: rows.append(f"   {lab}: {len(r)} · فوز {(np.array(r) > 0).mean() * 100:.0f}% · {np.mean(r):+.2f}R")
        for lab, f in (("تعديل الأوبشن موجب", lambda x: x.get("opt_adj", 0) > 0), ("تعديل الأوبشن سالب", lambda x: x.get("opt_adj", 0) < 0),
                       ("الأسواق تؤكد", lambda x: x.get("xm_adj", 0) > 0), ("الأسواق تتباعد", lambda x: x.get("xm_adj", 0) < 0)):
            r = [x["R"] for x in done if f(x)]
            if r: rows.append(f"   {lab}: {len(r)} · فوز {(np.array(r) > 0).mean() * 100:.0f}% · {np.mean(r):+.2f}R")
        if rows:
            L.append("\n<b>هل يضيف سياق الأوبشن/الأسواق قيمة؟</b>\n" + "\n".join(rows))
            L.append("<i>هذه فرضيات من بحث الغاما: حتى تثبت هنا بعشرات الصفقات اعتبرها سياقاً لا قاعدة.</i>")
    else:
        L.append("\nلا توجد إشارات حية مكتملة بعد.")
    L.append(f"\n🧠 دروس محفوظة: {len(lessons)} · استدعاءات Gemini منذ التشغيل: {llm.stats['calls']} (فشل {llm.stats['fail']})")
    if BT.get("stats"):
        L.append(f"\n<b>باكتست {BT['days']} يوم ({BT['symbol']}) — شموع 5د</b>")
        for k, v in sorted(BT["stats"].items(), key=lambda kv: -kv[1]["avgR"]):
            pf = "∞" if v["pf"] == float("inf") else f"{v['pf']:.2f}"
            ev = analytics.evidence(v, BT.get("placebo", {}).get(k), len([x for x in live_sigs if x["name"] == k and x["status"] != "open"]),
                                    float(np.mean([x["R"] for x in live_sigs if x["name"] == k and x["status"] != "open"] or [0])), len(BT["stats"]))
            L.append(f"• {SETUP_AR[k]}: {v['n']} · فوز {v['win'] * 100:.0f}% · {v['avgR']:+.2f}R · PF {pf} · {ev[0]}")
        g = BT.get("grades", {})
        if g: L.append("\nحسب الدرجة: " + " | ".join(f"{k}: {v['n']} صفقة {v['avgR']:+.2f}R" for k, v in sorted(g.items())))
        if BT.get("all"): L.append(f"الإجمالي: {BT['all']['n']} صفقة · فوز {BT['all']['win'] * 100:.0f}% · {BT['all']['avgR']:+.2f}R للصفقة")
        L.append("\n<i>الباكتست يفترض الدخول عند إغلاق الشمعة والخروج عند الوقف أو الهدف1 أو نهاية الجلسة، مع خصم تكلفة 0.05R. "
                 "الاتجاه اليومي فيه سببي، لكن VIX وML والأوبشن والأسواق المرتبطة غير مُضمَّنة. عيّنة 60 يوماً قصيرة. التفاصيل: /edge</i>")
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
    raw = candles(SYMBOLS[state["focus"]], "5m", "12d")
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


def h_debate(chat, arg=""):
    S = snapshot(closed_only=True)
    if not S: return no_data(chat)
    rounds = 2 if re.search(r"\b2\b", arg or "") else 1
    tg_text(f"⚖️ تنعقد لجنة التداول على {S['label']}… (جولة {rounds}، يستغرق حوالي {20 * rounds + 15} ثانية)", chat)
    act, _ = active_signals(S)
    sig = act[0][1] if act else None
    ext = sources.gather("SPY")
    card = card_text(S, True, ext=True)
    lines = sources.card_lines(ext, detail=True)
    les = agents.lessons_context(lessons, sig["name"]) if sig else agents.lessons_context(lessons, "")
    res = agents.full_debate(card, S["label"], S["price"], S["atr"], sig, les,
                             agents.position_context(pos_state["v"], S["price"], S["atr"]), rounds=rounds)
    sent = agents.sentiment_report(lines, get_news(8), S["label"]) if res.get("pm") else None
    txt = agents.debate_text(res, S["label"], S["price"])
    if sig: txt = f"🎯 على فرصة: <b>{SETUP_AR[sig['name']]}</b> ({sig['grade']}) {'شراء' if sig['dir'] > 0 else 'بيع'}\n\n" + txt
    tg_text(txt, chat, keyboard())
    if res.get("history") or res.get("pm"):
        tg_doc(agents.report_html(S["label"], S["price"], res, card + "\n\n" + "\n".join(lines), sent).encode("utf-8"),
               f"committee_{S['label']}_{now_et():%Y%m%d_%H%M}.html", "📎 تقرير اللجنة الكامل (افتحه بالمتصفح)", chat)


def h_sentiment(chat):
    ext = sources.gather("SPY")
    lines = sources.card_lines(ext, detail=True)
    rep_ = agents.sentiment_report(lines, get_news(8), state["focus"])
    out = ["🌐 <b>المزاج والأسواق التنبؤية</b>"]
    if rep_:
        out.append(f"\n<b>{agents.BAND_AR.get(rep_['overall_band'], rep_['overall_band'])}</b> · {float(rep_['overall_score']):.1f}/10 · "
                   f"{agents.CONF_AR.get(rep_.get('confidence'), '')}\n{fmt_ai(rep_['narrative'])}")
    else:
        out.append("\n⚠️ تعذّر توليد تقرير المزاج (Gemini). هذه البيانات الخام:")
    out.append("\n" + html.escape("\n".join(x for x in lines if not x.startswith("  "))))
    tg_text("\n".join(out), chat, keyboard())


def h_macro(chat):
    ext = sources.gather("SPY")
    fr, pm = ext.get("fr") or {}, ext.get("pm") or {}
    out = ["🏛️ <b>الماكرو</b>"]
    out.append("\n<b>FRED</b>\n" + ("\n".join("• " + html.escape(x) for x in fr["rows"]) if fr.get("ok") else html.escape(fr.get("note", "غير متاح"))))
    out.append("\n<b>Polymarket (ما تسعّره السوق للأحداث القادمة)</b>\n" +
               ("\n".join("• " + html.escape(x) for x in pm["lines"][:6]) if pm.get("ok") else html.escape(pm.get("note", "غير متاح"))))
    m = macro_context()
    if m: out.append("\n<b>السوق</b>: " + " · ".join(f"{k.upper()} {v[0]:.2f} ({v[1]:+.1f}%)" for k, v in m.items()))
    tg_text("\n".join(out), chat, keyboard())


def h_memory(chat):
    if not lessons:
        return tg_text("🧠 لا دروس بعد. تُكتب تلقائياً عند حسم كل إشارة (هدف أو وقف أو نهاية جلسة).", chat, keyboard())
    rows = [f"• <b>{x['date']}</b> {SETUP_AR.get(x['setup'], x['setup'])} {'شراء' if x['dir'] > 0 else 'بيع'} ({x['grade']}) "
            f"<b>{x['R']:+.2f}R</b>\n  {html.escape(x['text'])}" for x in reversed(lessons[-8:])]
    tg_text(f"🧠 <b>ذاكرة القرارات</b> ({len(lessons)} درس) — تُحقن في اللجنة القادمة\n\n" + "\n\n".join(rows), chat, keyboard())


def h_pos(chat, arg=""):
    a = (arg or "").strip().lower()
    cur = pos_state["v"]
    if not a:
        msg = ("لم تخبرني بصفقتك." if cur is None else "أنت خارج السوق (flat)." if cur["side"] == "flat" else
               f"صفقتك: {cur['side']} من {cur['entry']:.2f}" + (f" · وقف {cur['stop']:.2f}" if cur.get("stop") else ""))
        return tg_text(msg + "\nالصيغة: <code>/pos long 618.5 stop 617</code> أو <code>/pos short 620</code> أو <code>/pos flat</code> أو <code>/pos clear</code>", chat, keyboard())
    if a in ("clear", "مسح"):
        pos_state["v"] = None; save_state(); return tg_text("🗑️ نسيتُ صفقتك (لن أفترض شيئاً عن حسابك).", chat, keyboard())
    if a in ("flat", "off", "خارج"):
        pos_state["v"] = {"side": "flat"}; save_state(); return tg_text("✅ مسجّل: أنت خارج السوق.", chat, keyboard())
    m = re.match(r"(long|short|buy|sell|شراء|بيع|لونق|شورت)\s+(\d+(?:[.,]\d+)?)(?:\s+(?:stop|sl|وقف)\s*(\d+(?:[.,]\d+)?))?", a)
    if not m:
        return tg_text("لم أفهم. مثال: <code>/pos long 618.5 stop 617</code>", chat, keyboard())
    side = "long" if m.group(1) in ("long", "buy", "شراء", "لونق") else "short"
    pos_state["v"] = {"side": side, "entry": float(m.group(2).replace(",", ".")),
                      "stop": float(m.group(3).replace(",", ".")) if m.group(3) else None}
    save_state()
    tg_text(f"✅ سجّلتُ صفقتك: {side} من {pos_state['v']['entry']:.2f}. ستراعيها اللجنة (احتفظ/خفّف/اخرج/حرّك الوقف).", chat, keyboard())


def h_gate(chat, arg=""):
    g = (arg or "").strip().lower()
    if g not in ("soft", "hard", "off"):
        return tg_text(f"البوابة الحالية: <b>{settings.get('gate', 'soft')}</b>\n• soft: تُرسل كل الفرص وتُوسم إن عارضتها اللجنة\n"
                       "• hard: الفرصة التي تعارضها اللجنة لا تُرسل (لكن تُسجَّل لنقيس صحة الفيتو)\n• off: بدون لجنة على التنبيهات (يوفر حصة Gemini)",
                       chat, keyboard())
    settings["gate"] = g; save_state()
    tg_text(f"✅ بوابة اللجنة: <b>{g}</b>", chat, keyboard())


def h_flow(chat):
    S = snapshot(closed_only=True)
    if not S: return no_data(chat)
    OF = get_of(S)
    if not OF:
        return tg_text(f"📉 حالة الأوبشن غير متاحة الآن ({html.escape(optflow.status.get('note', ''))}). مصدرها سلسلة CBOE المتأخرة؛ قد تُحجب أحياناً من خوادم السحابة.", chat, keyboard())
    s0 = OF["spot"]
    reg = "🟢 غاما موجبة: تخميد، الأسعار تميل للارتداد حول الجدران" if OF["regime"] == "POS" else "🔴 غاما سالبة: تسارع، الاختراقات تتغذى على نفسها"
    L = [f"📉 <b>حالة الأوبشن · {OF['label']}</b> ${s0:.2f} <i>(متأخرة ~15د)</i>", reg + f" ({OF['strength']})", ""]
    for nm, v in optflow.levels(OF):
        L.append(f"• {nm}: <b>{v:.2f}</b> ({optflow.pct(v, s0)})")
    L.append(f"\nGEX القريب {optflow.money(OF['gex'])}/1%" + (f" · 0DTE {optflow.money(OF['gex0'])}" if OF["has0"] else ""))
    if OF["em"]: L.append(f"الحركة المتوقعة ±{OF['em']:.2f} ({OF['em'] / s0 * 100:.2f}%)" + (f" · الحركة منذ أمس {OF['em_used'] * 100:.0f}% من المتوقع اليومي" if OF["em_used"] is not None else ""))
    if OF["skew"] is not None: L.append(f"انحراف 25Δ: {OF['skew']:+.1f} نقطة IV")
    if OF["pc_vol"] is not None: L.append(f"نسبة بوت/كول: حجم {OF['pc_vol']:.2f}" + (f" · OI {OF['pc_oi']:.2f}" if OF["pc_oi"] is not None else ""))
    p = OF["prem"]
    if p["call"] + p["put"] > 0: L.append(f"أقساط اليوم: كول ${p['call'] / 1e6:.1f}M / بوت ${p['put'] / 1e6:.1f}M <i>(لا نعرف من البادئ)</i>")
    ai = brain("اشرح في 4 أسطر ماذا تعني حالة الأوبشن هذه لتداول 5 دقائق اليوم: أين الجدران والانقلاب بالنسبة للسعر، "
               "هل نتوقع تخميداً أم تسارعاً، وما الذي يبطل القراءة. لا تدّعِ معرفة من اشترى ومن باع.", card_text(S, False, ext=False) + "\n" + "\n".join(optflow.card_lines(OF)),
               chat, tokens=330, memo="طلب حالة الأوبشن")
    L.append("\n🧠 " + fmt_ai(ai))
    L.append("<i>OI مخزون الأمس؛ وإشارة GEX نموذج يفترض أن المحترفين طويلو كول/قصيرو بوت. سياق وليس قاعدة.</i>")
    tg_text("\n".join(L), chat, keyboard())


def h_options(chat):
    S = snapshot(closed_only=True)
    if not S: return no_data(chat)
    ch = optflow.fetch_chain(S["label"], S["price"], now_et().replace(tzinfo=None))
    if not ch:
        return tg_text(f"📉 سلسلة الأوبشن غير متاحة ({html.escape(optflow.status.get('note', ''))}).", chat, keyboard())
    OF = get_of(S)
    act, _ = active_signals(S)
    sg = act[0][1] if act else None
    ml = optflow._mins_left(now_et().replace(tzinfo=None))
    ideas, dte = optstrat.build(ch, ml, sg, OF)
    if not ideas:
        return tg_text("لم أجد سلسلة سائلة كافية لبناء هياكل الآن.", chat, keyboard())
    head_ = (f"🧩 <b>هياكل أوبشن لفرصة {SETUP_AR[sg['name']]}</b> ({'شراء' if sg['dir'] > 0 else 'بيع'} {sg['grade']}) · انتهاء {'اليوم (0DTE)' if dte == 0 else str(dte) + ' يوم'}"
             if sg else f"🧩 <b>هياكل محايدة</b> (لا فرصة نشطة) · انتهاء {'اليوم (0DTE)' if dte == 0 else str(dte) + ' يوم'}")
    L = [head_, f"السعر {ch['spot']:.2f} · انزلاق {optstrat.SLIP_PCT:.0f}% من نصف السبريد · عمولة ${optstrat.COMMISSION}/عقد/ساق"]
    scn = {}
    if sg:
        scn = {f"الهدف1 {sg['t1']:.2f}": sg["t1"], f"الوقف {sg['stop']:.2f}": sg["stop"]}
    for it in ideas:
        st = optstrat.evaluate(it["legs"], ch["spot"], ml, scenarios=scn)
        L.append(f"\n<b>{html.escape(it['title'])}</b>\n<code>{html.escape(optstrat._name(it['legs']))}</code>\n" + html.escape(optstrat.describe(st, sg)))
    L.append("\n<i>الأسعار من سلسلة متأخرة ~15د. السيناريوهات تقدير BSM بعد ساعة. تحقق من الأسعار الحية في وسيطك. لا توصية.</i>")
    tg_text("\n".join(L), chat, keyboard())


def h_edge(chat):
    if not BT.get("stats"):
        return tg_text("الباكتست لم يكتمل بعد. أرسل /backtest.", chat, keyboard())
    m, mc = BT.get("metrics"), BT.get("mc")
    L = [f"🔬 <b>هل لدينا أفضلية حقيقية؟</b> ({BT['days']} يوم · {BT['symbol']})"]
    if m:
        L.append(f"\n<b>كل الإشارات</b>: {m['n']} صفقة · متوسط {m['mean']:+.2f}R (حدود 95%: {m['ci95'][0]:+.2f} إلى {m['ci95'][1]:+.2f}) · t={m['tstat']:.1f}")
        L.append(f"Sharpe/صفقة {m['sharpe']:.2f} · Sortino {m['sortino']:.2f} · PF {m['pf']:.2f} · أقصى تراجع {m['mdd']:.1f}R · R² {m['r2']:.2f}")
        L.append("<i>الحد الأدنى للمتوسط يشمل الصفر؟ إذن لا نستطيع الجزم بوجود أفضلية.</i>" if m["ci95"][0] <= 0 else "<i>حد الثقة الأدنى موجب: إشارة جيدة لكنها على عيّنة قصيرة.</i>")
    L.append("\n<b>اختبار التوقيت العشوائي</b> (نفس الاتجاه والمخاطرة، لكن شمعة عشوائية في نفس اليوم):")
    n_tests = len(BT["stats"])
    for k, v in sorted(BT["stats"].items(), key=lambda kv: -kv[1]["avgR"]):
        pl = BT.get("placebo", {}).get(k)
        ev = analytics.evidence(v, pl, len([x for x in live_sigs if x["name"] == k and x["status"] != "open"]),
                                float(np.mean([x["R"] for x in live_sigs if x["name"] == k and x["status"] != "open"] or [0])), n_tests)
        L.append(f"• {SETUP_AR[k]} ({v['n']}): {v['avgR']:+.2f}R" + (f" مقابل عشوائي {pl['null_mean']:+.2f}R · p={pl['p']:.2f}" if pl else "") + f"\n   {ev[0]} — {html.escape(ev[1])}")
    tr = BT.get("trail", {})
    if tr:
        L.append(f"\n<b>الخروج</b>: ثابت (هدف1) {BT['all']['avgR']:+.2f}R مقابل تتبّع بعد الهدف1 {BT.get('trail_all', 0):+.2f}R")
    if mc:
        L.append(f"\n<b>مونت كارلو</b> (إعادة عيّنة، {mc['horizon']} صفقة قادمة): وسيط {mc['median']:+.1f}R · نطاق 5–95%: {mc['p5']:+.1f} إلى {mc['p95']:+.1f} · "
                 f"احتمال الخسارة {mc['p_loss'] * 100:.0f}% · تراجع نموذجي {mc['mdd_med']:.1f}R (سيئ 95%: {mc['mdd_p95']:.1f}R)")
    L.append(f"\n<i>p الصغير لا يكفي: عند اختبار {n_tests} سيناريوهات يجب أن يصمد بعد التصحيح (×{n_tests}). والأصدق دائماً الأداء الحي الذي يتراكم في /stats.</i>")
    tg_text("\n".join(L), chat, keyboard())


def h_risk(chat, arg=""):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", arg or "")
    if m:
        settings["account"] = float(m.group().replace(",", "")); save_state()
    acc = settings.get("account")
    mt = BT.get("metrics")
    pct_ = analytics.sizing(mt["kelly"], mt["mean"], mt["n"]) if mt else 0.0
    L = ["🛡️ <b>إدارة المخاطر</b>"]
    if not acc:
        return tg_text("حدّد حجم حسابك لأحسب لك الحجم: <code>/risk 25000</code>", chat, keyboard())
    L.append(f"الحساب: <b>${acc:,.0f}</b>")
    if mt:
        L.append(f"كيلي الكامل من الباكتست: {mt['kelly'] * 100:.0f}% <i>(غير موثوق بعيّنة قصيرة؛ لا تستخدمه كاملاً)</i>")
    use = pct_ if pct_ > 0 else 0.0025
    L.append(f"مخاطرة مقترحة لكل صفقة: <b>{use * 100:.2f}%</b> = ${acc * use:,.0f}" +
             ("  ← ربع كيلي بسقف 1%" if pct_ > 0 else "  ← حدّ أدنى حذر لأن الباكتست لم يثبت أفضلية بعد"))
    S = snapshot(closed_only=True)
    act = active_signals(S)[0] if S else []
    if act:
        sg = act[0][1]; rps = sg["risk"]; sh = int(acc * use / rps) if rps > 0 else 0
        L.append(f"\nعلى فرصة {SETUP_AR[sg['name']]} ({sg['grade']}): مخاطرة {rps:.2f}$ للسهم → <b>{sh} سهم</b> (قيمة المركز ${sh * sg['entry']:,.0f})")
        if sh * sg["entry"] > acc: L.append("⚠️ قيمة المركز أكبر من الحساب: ستحتاج رافعة. قلّل الحجم.")
    mc = BT.get("mc")
    if mc:
        L.append(f"\nتراجع نموذجي متوقع {mc['mdd_med']:.1f}R (سيئ 95%: {mc['mdd_p95']:.1f}R) = {mc['mdd_med'] * use * 100:.1f}% إلى {mc['mdd_p95'] * use * 100:.1f}% من الحساب بهذه المخاطرة.")
    L.append("<i>أدوات حسابية فقط. القرار والمسؤولية عليك.</i>")
    tg_text("\n".join(L), chat, keyboard())


ROUTES = {"commands": h_commands, "data_status": h_data_status, "why_no_signal": h_why_no_signal, "flow": h_flow, "options": h_options, "edge": h_edge, "scan": h_scan, "sentiment": h_sentiment, "macro": h_macro, "memory": h_memory, "stats": h_stats, "backtest": h_backtest, "alerts": h_alerts, "clear": h_clear, "start": h_start, "help": h_start, "switch": h_switch, "price": h_price, "levels": h_levels,
          "analyze": h_analyze, "chart": h_chart, "news": h_news}


ARG_ROUTES = {"start": h_start, "help": h_start, "commands": h_commands, "panel": h_panel, "channel": h_channel, "risk": h_risk, "alert": h_alert, "grade": h_grade, "debate": h_debate, "pos": h_pos, "gate": h_gate}


BUTTON_INTENTS = (
    (("الأزرار", "الازرار", "لوحة الأزرار", "لوحه الازرار", "زر التحكم", "قائمة الأزرار"), h_panel),
    (("زر السعر", "زر الاسعار", "زر الأسعار", "السعر الآن", "السعر الحين"), h_price),
    (("زر المستويات", "زر الدعم", "زر المقاومة", "الدعوم والمقاومات", "المستويات"), h_levels),
    (("زر الشارت", "زر الرسم", "الرسم البياني", "الشارت"), h_chart),
    (("زر الأخبار", "زر الاخبار", "الأخبار", "الاخبار"), h_news),
    (("زر التحليل", "التحليل الذكي", "حلل لي"), h_analyze),
    (("زر الفرص", "الفرص", "اقتناس الفرص", "اقتناص الفرص"), h_scan),
    (("زر الإحصائيات", "زر الاحصائيات", "الإحصائيات", "الاحصائيات"), h_stats),
)


def natural_button(text, chat):
    """يفهم طلبات الأزرار العربية بدون استهلاك حصة Gemini."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not any(k in t for k in ("زر", "الأزرار", "الازرار", "لوحة", "أرسل", "ارسل", "ابغى", "ابي")):
        return False
    for phrases, fn in BUTTON_INTENTS:
        if any(p in t for p in phrases):
            fn(chat)
            return True
    h_panel(chat)
    return True


def handle(text, chat):
    t = (text or "").strip()
    parts = t.lstrip("/").split(None, 1)
    cmd = re.sub(r"@\w+$", "", parts[0].lower()) if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    is_cmd = t.startswith("/") or t.lower() in ROUTES or t.lower() in ARG_ROUTES
    if is_cmd and cmd in ARG_ROUTES:
        return ARG_ROUTES[cmd](chat, arg)
    if is_cmd and cmd in ROUTES:
        return ROUTES[cmd](chat)
    if natural_button(t, chat):
        return None
    return h_free(chat, t)


# ============================== الويب هوك ==============================
PUBLIC_CMDS = {"start", "help", "commands", "price", "levels", "stats"}
PUBLIC_CB = {"menu_panel", "menu_hide", "menu_cmds", "price", "levels", "stats"}
_pub_hits = {}


def _cmd_of(text):
    parts = (text or "").strip().lstrip("/").split(None, 1)
    return re.sub(r"@\w+$", "", parts[0].lower()) if parts else ""


def _pub_limited(chat):
    """حد 8 طلبات/دقيقة لكل مستخدم عام كي لا يستنزف الخادم."""
    if len(_pub_hits) > 2000: _pub_hits.clear()
    now = time.time(); q = _pub_hits.setdefault(chat, deque(maxlen=12))
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= 8: return True
    q.append(now); return False


def _answer(cq_id, text=None, alert=False):
    try:
        d = {"callback_query_id": cq_id}
        if text: d["text"] = text; d["show_alert"] = "true" if alert else "false"
        _tg("answerCallbackQuery", data=d)
    except Exception:
        pass


def handle_callback(data, chat, mid, owner):
    """أزرار القوائم. يرجع True إن تم استهلاك الضغطة."""
    def edit(kb):
        try: _tg("editMessageReplyMarkup", data={"chat_id": chat, "message_id": mid, "reply_markup": json.dumps(kb)})
        except Exception: pass
    if data == "menu_panel": edit(panel_keyboard()); return True
    if data == "menu_hide": edit(keyboard()); return True
    if data == "menu_cmds": tg_text(commands_text(owner), chat, keyboard()); return True
    if not data.startswith("ch_") or not owner:
        return False
    if not CHANNEL_ID and data != "ch_menu":
        h_channel(chat); return True
    if data == "ch_menu":
        if not CHANNEL_ID: h_channel(chat)
        else: edit(channel_keyboard())
    elif data == "ch_toggle": settings["channel_on"] = not settings.get("channel_on", True); save_state(); edit(channel_keyboard())
    elif data == "ch_sl": settings["post_sl"] = not settings.get("post_sl", True); save_state(); edit(channel_keyboard())
    elif data == "ch_test": ch_test(chat)
    return True


def process_update(data):
    chat = None
    try:
        cq = data.get("callback_query")
        if cq:
            m = cq.get("message") or {}
            chat = str((m.get("chat") or {}).get("id", "")); text = cq.get("data", ""); mid = m.get("message_id"); ctype = (m.get("chat") or {}).get("type", "private")
        else:
            m = data.get("message") or {}
            chat = str((m.get("chat") or {}).get("id", "")); text = m.get("text", ""); mid = None; ctype = (m.get("chat") or {}).get("type", "private")
        if not chat or not text:
            if cq: _answer(cq["id"])
            return
        owner = (not ALLOWED_CHATS) or (chat in ALLOWED_CHATS)
        _tl.owner = owner
        if not owner:
            allowed = PUBLIC_BOT and ctype == "private" and ((text in PUBLIC_CB) if cq else (_cmd_of(text) in PUBLIC_CMDS and text.strip().startswith("/")))
            if not allowed:
                if cq: _answer(cq["id"], "🔒 هذه الميزة للمشرف فقط", True)
                elif ctype == "private" and PUBLIC_BOT and not _pub_limited(chat):
                    tg_text("🔒 هذه الميزة للمشرف.\nالمتاح لك: /price /levels /stats /commands", chat, keyboard())
                else: log.info("blocked chat %s", chat)
                return
            if _pub_limited(chat):
                if cq: _answer(cq["id"], "⏳ تمهّل قليلاً", False)
                return
        if cq:
            _answer(cq["id"])
            if handle_callback(text, chat, mid, owner): return
        try: _tg("sendChatAction", data={"chat_id": chat, "action": "typing"})
        except Exception: pass
        handle(text, chat)
    except Exception as e:
        log.exception("process_update")
        try: tg_text(f"❌ خطأ: {html.escape(str(e)[:80])}", chat, keyboard())
        except Exception: pass
    finally:
        _tl.owner = True


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
@app.route("/data_status")
def health():
    s, t = session_info()
    with _live_lock:
        live = dict(_live_ws)
    live["last_trade_age_sec"] = (round(time.time() - live["last_trade"], 1)
                                   if live.get("last_trade") else None)
    return jsonify({"status": "ok", "bot": "tdawll-v3.4", "focus": state["focus"], "timeframe": "5m",
                    "data_primary": "finnhub_ws_or_quote", "data_fallback": "yahoo_chart",
                    "live": live, "data": dict(_data_meta), "last_scan": dict(_scan_status), "session": s,
                    "uptime_min": int((time.time() - _started) / 60), "time_et": now_et().strftime("%H:%M:%S")})


@app.route("/why_no_signal")
def why_no_signal():
    return jsonify(dict(_scan_status))


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
                raw = candles(SYMBOLS[state["focus"]], "5m", "12d")
                if raw is not None and len(raw):
                    closed = raw[raw.index + pd.Timedelta(minutes=BAR_MINUTES) <= n.replace(tzinfo=None)]
                    check_price_alerts(raw)
                    if len(closed):
                        track_outcomes(closed, raw)
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
    threading.Thread(target=_live_ws_loop, daemon=True).start()
    threading.Thread(target=_live_quote_loop, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()


boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
