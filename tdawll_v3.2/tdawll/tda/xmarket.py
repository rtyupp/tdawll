# -*- coding: utf-8 -*-
"""
سياق الأسواق المتقاطعة (Layer 1 في بحث options-flow: «لا نعامل كل أصل مرتبط كمتنبئ مستقل؛ المعلومة في التباعد»).

نقيس خلال آخر ساعتين (8 شموع 15د): هل QQQ وIWM وXLK وSMH تؤكد حركة SPY؟ هل الائتمان (HYG) يوافق؟ وما شكل منحنى VIX
(VIX9D / VIX / VIX3M)؟ لا بيانات تاريخية متوازية هنا، لذلك التعديلات صغيرة ومُسجَّلة لتُقاس حيّاً في /stats.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

log = logging.getLogger("tdawll.xmarket")
EQUITY = ("QQQ", "IWM", "XLK", "SMH")
CREDIT = "HYG"
VIX_SYMS = {"vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M", "vvix": "^VVIX"}
BARS = 8
_pool = ThreadPoolExecutor(max_workers=6)


def _ret(df, idx):
    """عائد آخر BARS شموع بالنسبة المئوية على نفس طوابع SPY."""
    if df is None or len(df) < 20:
        return None
    s = df["Close"].reindex(idx).ffill()
    if s.isna().iloc[-(BARS + 1):].any():
        return None
    a, b = float(s.iloc[-(BARS + 1)]), float(s.iloc[-1])
    return (b / a - 1) * 100 if a else None


def compute(fetch, spy_df):
    """fetch(sym, interval, rng) → DataFrame أو None. spy_df: شموع SPY المغلقة (مع Close)."""
    if spy_df is None or len(spy_df) < BARS + 2:
        return None
    idx = spy_df.index[-60:]
    spy_ret = _ret(spy_df, idx)
    jobs = {s: _pool.submit(fetch, s, "15m", "5d") for s in (*EQUITY, CREDIT)}
    vjobs = {k: _pool.submit(fetch, s, "1d", "1mo") for k, s in VIX_SYMS.items()}
    wait(list(jobs.values()) + list(vjobs.values()), timeout=20)
    peers = {}
    for s, f in jobs.items():
        try:
            peers[s] = _ret(f.result(timeout=0), idx)
        except Exception:
            peers[s] = None
    vix = {}
    for k, f in vjobs.items():
        try:
            d = f.result(timeout=0)
            if d is not None and len(d):
                vix[k] = float(d["Close"].iloc[-1])
        except Exception:
            pass
    got = {k: v for k, v in peers.items() if v is not None}
    if spy_ret is None or not got:
        return None
    near = (vix["vix9d"] / vix["vix"]) if vix.get("vix9d") and vix.get("vix") else None
    far = (vix["vix"] / vix["vix3m"]) if vix.get("vix") and vix.get("vix3m") else None
    return dict(spy=spy_ret, peers=peers, vix=vix, term_near=near, term_far=far, n_ok=len(got))


def card_lines(XM):
    if not XM:
        return ["XMARKET: غير متاح (تعذّر جلب الأسواق المرتبطة) — غير متاح وليس صمتاً"]
    pr = " ".join(f"{s}{v:+.2f}%" if v is not None else f"{s}?" for s, v in XM["peers"].items())
    L = [f"XMARKET[آخر ساعتين]: SPY {XM['spy']:+.2f}% | {pr}"]
    t = []
    if XM["term_near"] is not None: t.append(f"VIX9D/VIX={XM['term_near']:.2f} ({'توتر قصير الأجل' if XM['term_near'] > 1.05 else 'هادئ' if XM['term_near'] < 0.9 else 'عادي'})")
    if XM["term_far"] is not None: t.append(f"VIX/VIX3M={XM['term_far']:.2f} ({'منحنى مقلوب = ضغط' if XM['term_far'] > 1.0 else 'طبيعي'})")
    if XM["vix"].get("vvix"): t.append(f"VVIX={XM['vix']['vvix']:.0f}")
    if t: L.append("VIX_TERM: " + " | ".join(t))
    return L


def adjust(XM, sg):
    """تعديل صغير (±8 كحد أقصى). التباعد مع السوق الأوسع أهم من التوافق."""
    if not XM:
        return 0, []
    dr, adj, notes = sg["dir"], 0, []
    eq = {s: v for s, v in XM["peers"].items() if s in EQUITY and v is not None}
    if len(eq) >= 3:
        agree = [s for s, v in eq.items() if v * dr >= 0.02]
        against = [s for s, v in eq.items() if v * dr <= -0.02]
        frac = len(agree) / len(eq)
        if frac >= 0.75:
            adj += 4; notes.append(f"السوق الأوسع يؤكد ({', '.join(agree)})")
        elif frac <= 0.25:
            adj -= 5; notes.append(f"⚠️ تباعد: {', '.join(against) or 'الأسواق المرتبطة'} لا تؤكد")
    h = XM["peers"].get("HYG")
    if h is not None and h * dr <= -0.03:
        adj -= 2; notes.append("⚠️ الائتمان (HYG) لا يؤكد")
    n, f = XM.get("term_near"), XM.get("term_far")
    if dr == 1 and ((f is not None and f > 1.0) or (n is not None and n > 1.05)):
        adj -= 4; notes.append("⚠️ منحنى VIX مقلوب/متوتر يضغط على الشراء")
    if dr == -1 and n is not None and f is not None and n < 0.9 and f < 0.85:
        adj -= 3; notes.append("هدوء VIX يضعف البيع")
    return int(max(-8, min(8, adj))), notes
