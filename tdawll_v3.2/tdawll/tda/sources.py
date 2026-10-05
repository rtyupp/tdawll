# -*- coding: utf-8 -*-
"""
مصادر إضافية مقتبسة فكرةً من TradingAgents (Apache-2.0):
  StockTwits (مزاج المتداولين الأفراد) · Reddit RSS · Polymarket (احتمالات الأحداث القادمة) · FRED (ماكرو رسمي)

المبدأ المأخوذ منهم: المصدر الذي يفشل يُعلَّم «غير متاح» ولا يُعامَل كأنه «صمت». الذكاء يعرف الفرق.
كل المصادر مجانية: FRED فقط يحتاج مفتاحاً مجانياً اختيارياً (FRED_API_KEY).
"""
import os, re, json, html, time, logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone

import requests

log = logging.getLogger("tdawll.sources")
UA = {"User-Agent": "Mozilla/5.0 (compatible; tdawll/3.1)", "Accept": "application/json"}
FRED_KEY = os.environ.get("FRED_API_KEY")
_c, _neg = {}, {}
_pool = ThreadPoolExecutor(max_workers=4)        # مهام المصادر الرئيسية
_pool2 = ThreadPoolExecutor(max_workers=6)       # مهام FRED الداخلية (منفصل كي لا يتعلّق المجمّع)


def _cached(key, ttl, fn):
    hit = _c.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    neg = _neg.get(key)
    if neg and time.time() - neg[0] < 300:             # فشل حديث: لا نعيد المحاولة الآن
        return neg[1]
    try:
        val = fn()
    except Exception as e:
        log.warning("source %s: %s", key, e)
        val = None
    if val is not None and val.get("ok"):
        _c[key] = (time.time(), val)
        _neg.pop(key, None)
        return val
    if hit:                                            # نسخة قديمة أفضل من لا شيء (نوسمها)
        res = dict(hit[1]); res["stale"] = True
    else:
        res = val or {"ok": False, "note": "فشل الجلب"}
    _neg[key] = (time.time(), res)
    return res


ON_TOPIC = re.compile(r"\b(spy|spx|s&p|sp500|market|fed|powell|rates?|cpi|inflation|yields?|vix|puts?|calls?|0dte|"
                      r"nasdaq|qqq|stocks?|earnings|tariffs?|recession)\b", re.I)


# ---------------------------------------------------------------- StockTwits
def stocktwits(symbol="SPY", limit=30):
    def go():
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json", headers=UA, timeout=8)
        if r.status_code != 200:
            return {"ok": False, "note": f"StockTwits غير متاح (HTTP {r.status_code})"}
        msgs = r.json().get("messages", [])[:limit]
        if not msgs:
            return {"ok": False, "note": "StockTwits رجع بلا رسائل"}
        bull = bear = unl = 0
        lines, newest = [], None
        for m in msgs:
            tag = (((m.get("entities") or {}).get("sentiment")) or {}).get("basic")
            if tag == "Bullish": bull += 1
            elif tag == "Bearish": bear += 1
            else: unl += 1
            body = html.unescape(m.get("body") or "").replace("\n", " ").strip()
            body = body[:200] + ("…" if len(body) > 200 else "")
            lines.append(f"[{m.get('created_at', '')[:16]} · {tag or 'no-label'}] {body}")
            try:
                ts = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00"))
                newest = max(newest, ts) if newest else ts
            except Exception:
                pass
        age = (datetime.now(timezone.utc) - newest).total_seconds() / 60 if newest else None
        return {"ok": True, "bull": bull, "bear": bear, "unl": unl, "total": len(msgs), "lines": lines, "age_min": age}
    return _cached(f"st:{symbol}", 600, go)


# -------------------------------------------------------------------- Reddit
def reddit(limit=10):
    def go():
        url = ("https://www.reddit.com/r/wallstreetbets+stocks+investing+options/search.rss"
               "?q=SPY+OR+%22S%26P+500%22+OR+%22S%26P500%22&restrict_sr=on&sort=new&t=week")
        r = requests.get(url, headers={"User-Agent": "tdawll/3.1 (rss reader)"}, timeout=8)
        if r.status_code == 429:
            time.sleep(min(float(r.headers.get("Retry-After", 2) or 2), 3))
            r = requests.get(url, headers={"User-Agent": "tdawll/3.1 (rss reader)"}, timeout=8)
        if r.status_code != 200:
            return {"ok": False, "note": f"Reddit غير متاح (HTTP {r.status_code})"}
        ns = {"a": "http://www.w3.org/2005/Atom"}
        posts = []
        for e in ET.fromstring(r.content).findall("a:entry", ns):
            title = (e.findtext("a:title", "", ns) or "").strip()
            body = re.sub(r"<[^>]+>", " ", html.unescape(e.findtext("a:content", "", ns) or ""))
            body = re.sub(r"\s+", " ", body).strip()
            if ON_TOPIC.search(title + " " + body):                # فلتر محلي خفيف بدل Jev
                posts.append({"title": title[:160], "text": body[:220], "t": (e.findtext("a:updated", "", ns) or "")[:16]})
        if not posts:
            return {"ok": False, "note": "Reddit: لا منشورات ذات صلة خلال الأسبوع"}
        return {"ok": True, "posts": posts[:limit]}
    return _cached("reddit", 900, go)


# ---------------------------------------------------------------- Polymarket
def _jl(v):
    if isinstance(v, list): return v
    try: return json.loads(v)
    except Exception: return []


def polymarket(topics=("Fed rate cut", "recession"), per=3):
    def go():
        now, out = datetime.now(timezone.utc), []
        for topic in topics:
            d = requests.get("https://gamma-api.polymarket.com/public-search", params={"q": topic, "limit_per_type": 20},
                             headers=UA, timeout=10).json()
            cands = []
            for ev in d.get("events", []):
                for m in ev.get("markets", []):
                    if m.get("closed"): continue
                    end = m.get("endDate")
                    try:
                        if end and datetime.fromisoformat(end.replace("Z", "+00:00")) < now: continue
                    except ValueError:
                        pass
                    if _jl(m.get("outcomePrices")) and _jl(m.get("outcomes")): cands.append(m)
            cands.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)
            for m in cands[:per]:
                try: p = float(_jl(m["outcomePrices"])[0])
                except Exception: continue
                wk = m.get("oneWeekPriceChange")
                out.append(f"{m.get('question')} — {(_jl(m.get('outcomes')) or ['Yes'])[0]} {p:.0%} "
                           f"(حجم ${(m.get('volumeNum') or 0):,.0f}، ينتهي {(m.get('endDate') or '')[:10]}"
                           + (f"، أسبوعياً {wk * 100:+.1f}نقطة" if isinstance(wk, (int, float)) and wk else "") + ")")
        return {"ok": True, "lines": out} if out else {"ok": False, "note": "Polymarket: لا أسواق مفتوحة مطابقة"}
    return _cached("poly", 1800, go)


# ---------------------------------------------------------------------- FRED
FRED_SERIES = [("سعر الفائدة الفيدرالية", "FEDFUNDS", "pct"), ("عائد 10 سنوات", "DGS10", "pct"),
               ("منحنى 10-2", "T10Y2Y", "pct"), ("التضخم CPI سنوياً", "CPIAUCSL", "yoy"),
               ("البطالة", "UNRATE", "pct"), ("طلبات إعانة البطالة", "ICSA", "int")]


def fred():
    if not FRED_KEY:
        return {"ok": False, "note": "FRED غير مفعّل (أضف FRED_API_KEY مجاني لبيانات رسمية)", "disabled": True}

    def one(item):
        label, sid, kind = item
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": sid, "api_key": FRED_KEY, "file_type": "json", "sort_order": "desc", "limit": 15},
                         timeout=10).json()
        obs = [(o["date"], float(o["value"])) for o in r.get("observations", []) if o.get("value") not in (".", None)]
        if len(obs) < 2: return None
        (d0, v0), (_, v1) = obs[0], obs[1]
        if kind == "yoy":
            if len(obs) < 13: return None
            yoy = (v0 / obs[12][1] - 1) * 100
            return f"{label}: {yoy:.1f}% ({d0})"
        if kind == "int":
            return f"{label}: {v0:,.0f} ({d0}، السابق {v1:,.0f})"
        return f"{label}: {v0:.2f} ({d0}، السابق {v1:.2f})"

    def go():
        futs = [_pool2.submit(one, it) for it in FRED_SERIES]
        wait(futs, timeout=14)
        rows = []
        for f in futs:
            try:
                v = f.result(timeout=0)
                if v: rows.append(v)
            except Exception:
                pass
        return {"ok": True, "rows": rows} if rows else {"ok": False, "note": "FRED لم يرجع بيانات"}
    return _cached("fred", 6 * 3600, go)


# ------------------------------------------------------------------ التجميع
def gather(symbol="SPY"):
    """يجلب كل المصادر بالتوازي (مخزّنة مؤقتاً). لا يرفع استثناء أبداً."""
    jobs = {"st": _pool.submit(stocktwits, symbol), "rd": _pool.submit(reddit),
            "pm": _pool.submit(polymarket), "fr": _pool.submit(fred)}
    wait(jobs.values(), timeout=16)
    out = {}
    for k, f in jobs.items():
        try: out[k] = f.result(timeout=0)
        except Exception: out[k] = {"ok": False, "note": "انتهت المهلة"}
    return out


def card_lines(ext, detail=False):
    """أسطر البطاقة الفنية. المصدر الفاشل يُذكر صراحةً."""
    L = []
    st = ext.get("st") or {}
    if st.get("ok"):
        n = max(st["bull"] + st["bear"], 1)
        extra = " ⚠️ تطرّف شديد: خطر عكسي" if (st["bull"] / n >= .9 or st["bear"] / n >= .9) and st["bull"] + st["bear"] >= 8 else ""
        am = st.get("age_min")
        age = (f"، أحدث رسالة قبل {am / 60:.1f} ساعة" + (" ⚠️ قديمة: المزاج غير حي" if am > 240 else "")) if am is not None else ""
        L.append(f"SOCIAL_STOCKTWITS: صاعد {st['bull']} / هابط {st['bear']} / بلا وسم {st['unl']} من {st['total']} رسالة{age}{extra}"
                 + (" [نسخة قديمة]" if st.get("stale") else ""))
        if detail: L += ["  " + x for x in st["lines"][:8]]
    else:
        L.append(f"SOCIAL_STOCKTWITS: {st.get('note', 'غير متاح')} — غير متاح وليس صمتاً")
    rd = ext.get("rd") or {}
    if rd.get("ok"):
        L.append(f"SOCIAL_REDDIT: {len(rd['posts'])} منشور ذو صلة. أحدثها: " + " | ".join(p["title"] for p in rd["posts"][:3]))
        if detail: L += [f"  [{p['t']}] {p['title']} — {p['text']}" for p in rd["posts"][:6]]
    else:
        L.append(f"SOCIAL_REDDIT: {rd.get('note', 'غير متاح')} — غير متاح وليس صمتاً")
    pm = ext.get("pm") or {}
    L.append("PREDICTION_MARKETS: " + (" || ".join(pm["lines"][:5 if detail else 4]) if pm.get("ok") else f"{pm.get('note', 'غير متاح')}"))
    fr = ext.get("fr") or {}
    if fr.get("ok"): L.append("FRED_MACRO: " + " | ".join(fr["rows"]))
    elif not fr.get("disabled"): L.append(f"FRED_MACRO: {fr.get('note', 'غير متاح')}")
    return L
