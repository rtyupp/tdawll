# -*- coding: utf-8 -*-
"""
حالة سوق الأوبشن (SPY / SPX) من سلسلة CBOE المتأخرة المجانية (بلا مفتاح).

الأفكار من بحث options-flow (Layer 0): تركّز الأسعار (strike concentration)، التموضع (OI)،
تعرّض الغاما (GEX) ومستوى انقلابها، الجدران، max pain، الحركة المتوقعة، انحراف الـIV،
جودة التسعير. التنفيذ هنا مستقل وخفيف ولا يستخدم أي كود من ذلك المشروع (ليس فيه كود أصلاً).

تنبيهات صادقة (مأخوذة من فلسفة ذلك البحث):
  • OI هو مخزون نهاية اليوم السابق. لا نعرف من اشترى ومن باع، ولا هل الصفقة فتحت أو أغلقت.
  • إشارة GEX تفترض أن المتداولين المحترفين «طويلون كولات وقصيرون بوتات». هذا نموذج، لا ميزانية حقيقية.
  • البيانات متأخرة ~15 دقيقة. الاستنتاجات سياق وليست أوامر.
"""
import os, re, time, math, gc, logging
from datetime import datetime, date

import numpy as np
import requests

from . import bsm

log = logging.getLogger("tdawll.optflow")
UA = {"User-Agent": "Mozilla/5.0 (compatible; tdawll/3.2)", "Accept": "application/json"}
ENABLED = os.environ.get("OPTFLOW", "1") == "1"
SYM_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
BAND = float(os.environ.get("OPT_BAND", "0.07"))             # نطاق الأسعار حول السعر (±7%)
MAX_DTE = int(os.environ.get("OPT_MAX_DTE", "45"))
_cache = {}
status = {"ok": None, "note": "لم يُجلب بعد", "ts": 0.0}


def _num(x, default=0.0):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def cboe_symbol(label):
    return "_SPX" if label == "SPX" else label


def fetch_chain(label="SPY", spot_hint=None, now_et=None):
    """يرجع {'spot','contracts':[...],'asof'} مختزلاً (نطاق سعر + حد DTE) أو None. مخزّن 5 دقائق."""
    if not ENABLED:
        status.update(ok=False, note="OPTFLOW=0 (معطّل)")
        return None
    key = f"chain:{label}"
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    sym = cboe_symbol(label)
    try:
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json", headers=UA, timeout=20)
        if r.status_code != 200:
            status.update(ok=False, note=f"CBOE HTTP {r.status_code}", ts=time.time())
            return hit[1] if hit else None
        j = r.json()
        del r
    except Exception as e:
        status.update(ok=False, note=f"CBOE: {type(e).__name__}", ts=time.time())
        return hit[1] if hit else None
    data = (j or {}).get("data") or {}
    spot = _num(data.get("current_price")) or _num(spot_hint)
    opts = data.get("options") or []
    if not spot or not opts:
        status.update(ok=False, note="CBOE رجع بلا بيانات", ts=time.time())
        return hit[1] if hit else None
    today = (now_et or datetime.now()).date()
    out = []
    for o in opts:
        m = SYM_RE.match(str(o.get("option", "")))
        if not m:
            continue
        try:
            exp = datetime.strptime(m.group(2), "%y%m%d").date()
        except ValueError:
            continue
        dte = (exp - today).days
        if dte < 0 or dte > MAX_DTE:
            continue
        strike = int(m.group(4)) / 1000.0
        if abs(strike / spot - 1) > BAND:
            continue
        bid, ask = _num(o.get("bid")), _num(o.get("ask"))
        iv = _num(o.get("iv"))
        iv = iv / 100.0 if iv > 3 else iv                          # قد يأتي كنسبة مئوية
        out.append({"sym": o["option"], "exp": exp, "dte": dte, "k": strike, "t": "call" if m.group(3) == "C" else "put",
                    "bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else _num(o.get("last_trade_price")),
                    "iv": iv, "delta": _num(o.get("delta"), None), "gamma": _num(o.get("gamma")),
                    "oi": _num(o.get("open_interest")), "vol": _num(o.get("volume")), "last": _num(o.get("last_trade_price"))})
    del j, data, opts
    gc.collect()
    if not out:
        status.update(ok=False, note="لا عقود ضمن النطاق", ts=time.time())
        return None
    res = {"spot": spot, "contracts": out, "asof": time.time(), "label": label}
    _cache[key] = (time.time(), res)
    status.update(ok=True, note=f"{len(out)} عقد", ts=time.time())
    return res


# ------------------------------------------------------------------ التحليل
def _mins_left(now_et):
    n = now_et or datetime.now()
    return max((16 * 60) - (n.hour * 60 + n.minute), 0)


def _T(dte, mins_left):
    """زمن الانتهاء بالسنوات؛ 0DTE يُفرض له حد أدنى 15 دقيقة كي لا تنفجر الغاما."""
    return max(dte * 1440 + mins_left, 15) / 525600.0


def analyze(ch, prev_close=None, now_et=None):
    """يحسب مؤشرات الحالة. يرجع dict جاهز للبطاقة والتسجيل."""
    spot, cs = ch["spot"], ch["contracts"]
    ml = _mins_left(now_et)
    near = [c for c in cs if c["dte"] <= 7]
    if not near:
        return None
    exps = sorted({c["dte"] for c in cs})
    dte0 = 0 if 0 in exps else None
    nearest = exps[0]

    k = np.array([c["k"] for c in near]); oi = np.array([c["oi"] for c in near])
    sign = np.array([1.0 if c["t"] == "call" else -1.0 for c in near])
    iv = np.array([c["iv"] for c in near]); Tt = np.array([_T(c["dte"], ml) for c in near])
    gam = np.array([c["gamma"] for c in near])
    miss = gam <= 0
    if miss.any():                                                    # غاما ناقصة: نحسبها من BSM
        gam[miss] = bsm.gamma_grid(spot, k[miss], Tt[miss], iv[miss])

    mult = 100.0
    gex = sign * gam * oi * mult * spot ** 2 * 0.01                  # $ لكل 1% حركة
    total = float(gex.sum()); gross = float(np.abs(gex).sum()) or 1.0

    # --- مستوى انقلاب الغاما: نعيد حساب المجموع عند أسعار افتراضية
    grid = spot * (1 + np.linspace(-0.05, 0.05, 41))
    vals = np.array([float((sign * bsm.gamma_grid(s, k, Tt, iv) * oi * mult * s ** 2 * 0.01).sum()) for s in grid])
    flip = None
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            cand = grid[i] - vals[i] * (grid[i + 1] - grid[i]) / (vals[i + 1] - vals[i]) if vals[i + 1] != vals[i] else grid[i]
            if flip is None or abs(cand - spot) < abs(flip - spot):
                flip = float(cand)

    # --- جدران (على GEX لكل سعر) وتركّز
    by_k = {}
    for kk, g in zip(k, gex):
        by_k[kk] = by_k.get(kk, 0.0) + g
    call_by = {}; put_by = {}
    for c, g in zip(near, gex):
        (call_by if c["t"] == "call" else put_by)[c["k"]] = (call_by if c["t"] == "call" else put_by).get(c["k"], 0.0) + g
    above = {s: g for s, g in call_by.items() if s >= spot and g > 0}
    below = {s: -g for s, g in put_by.items() if s <= spot and g < 0}
    call_wall = max(above, key=above.get) if above else None
    put_wall = max(below, key=below.get) if below else None
    absg = np.sort(np.abs(np.array(list(by_k.values()))))[::-1]
    top5 = float(absg[:5].sum() / (absg.sum() or 1.0)) if len(absg) else 0.0
    hhi = float(((absg / (absg.sum() or 1.0)) ** 2).sum()) if len(absg) else 0.0

    g0 = float(gex[[c["dte"] == 0 for c in near]].sum()) if dte0 == 0 else 0.0

    # --- max pain للانتهاء الأقرب
    ex = [c for c in cs if c["dte"] == nearest and c["oi"] > 0]
    pain = None
    if ex:
        strikes = sorted({c["k"] for c in ex})
        best = None
        for s in strikes:
            tot = sum(c["oi"] * (max(s - c["k"], 0) if c["t"] == "call" else max(c["k"] - s, 0)) for c in ex)
            if best is None or tot < best[0]:
                best = (tot, s)
        pain = best[1]

    # --- الحركة المتوقعة من ستراددل ATM
    def straddle(dte):
        rows = [c for c in cs if c["dte"] == dte and c["bid"] > 0 and c["ask"] > 0]
        if not rows:
            return None
        ks = sorted({c["k"] for c in rows}, key=lambda s: abs(s - spot))
        for s in ks[:4]:
            cl = [c for c in rows if c["k"] == s and c["t"] == "call"]; pt = [c for c in rows if c["k"] == s and c["t"] == "put"]
            if cl and pt:
                return cl[0]["mid"] + pt[0]["mid"], s, (cl[0]["iv"] + pt[0]["iv"]) / 2
        return None
    st0 = straddle(nearest)
    em = st0[0] if st0 else None                       # الحركة المتوقعة المتبقية حتى الانتهاء الأقرب (تنكمش خلال اليوم)
    atm_iv = st0[2] if st0 else None
    # الحركة اليومية الكاملة من IV أقرب انتهاء ≥1 يوم (ثابتة عبر اليوم فتصلح للمقارنة): S·σ·√(1/252)
    st1 = next((straddle(d) for d in exps if d >= 1 and straddle(d)), None)
    iv_day = st1[2] if st1 else atm_iv
    em_day = spot * iv_day * math.sqrt(1 / 252) if iv_day else None
    prev = prev_close or None
    used = (abs(spot - prev) / em_day) if (em_day and prev) else None
    if used is not None and used > 3:                  # قيمة غير معقولة = عدم تطابق بيانات، لا نستعملها
        used = None

    # --- انحراف 25 دلتا (انتهاء 1..7 أيام إن وُجد)
    skew = None
    sk_dte = next((d for d in exps if 1 <= d <= 7), None)
    if sk_dte is not None:
        rows = [c for c in cs if c["dte"] == sk_dte and c["iv"] > 0 and c["delta"] is not None]
        puts = [c for c in rows if c["t"] == "put" and c["delta"] < 0]; calls = [c for c in rows if c["t"] == "call" and c["delta"] > 0]
        if puts and calls:
            p = min(puts, key=lambda c: abs(c["delta"] + 0.25)); q = min(calls, key=lambda c: abs(c["delta"] - 0.25))
            if abs(p["delta"] + 0.25) < 0.12 and abs(q["delta"] - 0.25) < 0.12:
                skew = (p["iv"] - q["iv"]) * 100

    # --- نسب وتدفق الأقساط
    def pcr(sel, key):
        c = sum(x[key] for x in sel if x["t"] == "call"); p_ = sum(x[key] for x in sel if x["t"] == "put")
        return (p_ / c) if c > 0 else None
    sel0 = [c for c in cs if c["dte"] == nearest]
    pc_vol, pc_oi, pc_vol0 = pcr(cs, "vol"), pcr(cs, "oi"), pcr(sel0, "vol")
    vol_tot = sum(c["vol"] for c in cs) or 1.0
    share0 = sum(c["vol"] for c in cs if c["dte"] == 0) / vol_tot if dte0 == 0 else 0.0

    prem = {"call": 0.0, "put": 0.0, "call_near": 0.0, "put_near": 0.0, "call_otm": 0.0, "put_otm": 0.0}
    for c in sel0:
        p_ = c["vol"] * c["mid"] * 100
        prem[c["t"]] += p_
        if abs(c["k"] / spot - 1) <= 0.005: prem[c["t"] + "_near"] += p_
        otm = (c["t"] == "call" and c["k"] > spot) or (c["t"] == "put" and c["k"] < spot)
        if otm: prem[c["t"] + "_otm"] += p_
    hot = max(sel0, key=lambda c: c["vol"] * c["mid"], default=None)

    atm = [c for c in near if abs(c["k"] / spot - 1) <= 0.01]
    qual = (sum(1 for c in atm if c["bid"] > 0 and c["ask"] > 0 and (c["ask"] - c["bid"]) / max(c["mid"], 1e-9) < 0.10) / len(atm)) if atm else None

    ratio = total / gross
    if flip is None:
        regime = "POS" if total > 0 else "NEG"
    else:
        regime = "POS" if (spot > flip and total > 0) else "NEG" if (spot < flip) else ("POS" if total > 0 else "NEG")
    strength = "قوي" if abs(ratio) > 0.35 else "متوسط" if abs(ratio) > 0.15 else "ضعيف"

    return dict(label=ch["label"], spot=spot, n=len(cs), nearest=nearest, has0=dte0 == 0, gex=total, gex0=g0, ratio=ratio,
                regime=regime, strength=strength, flip=flip, call_wall=call_wall, put_wall=put_wall, top5=top5, hhi=hhi,
                pain=pain, em=em, atm_iv=atm_iv, em_used=used, skew=skew, pc_vol=pc_vol, pc_oi=pc_oi, pc_vol0=pc_vol0, em_day=em_day,
                share0=share0, prem=prem, hot=(hot["sym"], hot["vol"], hot["mid"]) if hot else None, quality=qual,
                mins_left=ml, ts=time.time())


def get(label="SPY", spot_hint=None, prev_close=None, now_et=None):
    ch = fetch_chain(label, spot_hint, now_et)
    if not ch:
        return None
    try:
        return analyze(ch, prev_close, now_et)
    except Exception:
        log.exception("optflow.analyze")
        status.update(ok=False, note="خطأ في التحليل")
        return None


def pct(x, spot):
    return f"{(x / spot - 1) * 100:+.2f}%"


def money(v):
    a = abs(v)
    s = f"{a / 1e9:.2f}B" if a >= 1e9 else f"{a / 1e6:.1f}M" if a >= 1e6 else f"{a / 1e3:.0f}K"
    return ("-" if v < 0 else "+") + "$" + s


def levels(OF):
    """مستويات الأوبشن كنقاط سعر (اسم، قيمة) لتدخل في الأهداف والعوائق."""
    if not OF:
        return []
    out = []
    for nm, key in (("جدار كول", "call_wall"), ("جدار بوت", "put_wall"), ("انقلاب الغاما", "flip"), ("ماكس بين", "pain")):
        v = OF.get(key)
        if v: out.append((nm, float(v)))
    return out


def card_lines(OF):
    if not OF:
        return [f"OPTIONS_STATE: غير متاح ({status.get('note', '')}) — غير متاح وليس صمتاً"]
    s = OF["spot"]
    reg = {"POS": "غاما موجبة (تخميد: ميل لارتداد الحركة)", "NEG": "غاما سالبة (تسارع: الحركة تتغذى على نفسها)"}[OF["regime"]]
    L = [f"OPTIONS_STATE[{OF['label']}، متأخرة ~15د، نموذج: المحترفون طويلو كول/قصيرو بوت]: GEX قريب={money(OF['gex'])}/1% "
         f"({OF['strength']}) → {reg}" + (f" | 0DTE GEX={money(OF['gex0'])}" if OF["has0"] else ""),
         "OPTIONS_LEVELS: " + " | ".join(f"{n} {v:.2f} ({pct(v, s)})" for n, v in levels(OF)) if levels(OF) else "OPTIONS_LEVELS: لا مستويات"]
    ex = []
    if OF["em"]: ex.append(f"EM(ستراددل {'0DTE' if OF['nearest'] == 0 else str(OF['nearest']) + 'DTE'})=±{OF['em']:.2f} ({OF['em'] / s * 100:.2f}%)")
    if OF["em_used"] is not None: ex.append(f"الحركة منذ إغلاق أمس = {OF['em_used'] * 100:.0f}% من الحركة اليومية المتوقعة (±{OF['em_day']:.2f})")
    if OF["skew"] is not None: ex.append(f"SKEW25Δ={OF['skew']:+.1f} نقطة IV ({'خوف/طلب حماية' if OF['skew'] > 5 else 'عادي' if OF['skew'] > 1 else 'ميل صاعد'})")
    if OF["atm_iv"]: ex.append(f"IV_ATM={OF['atm_iv'] * 100:.1f}%")
    if OF["pc_vol"] is not None: ex.append(f"PCR_vol={OF['pc_vol']:.2f}")
    if OF["pc_oi"] is not None: ex.append(f"PCR_OI={OF['pc_oi']:.2f}")
    if OF["has0"]: ex.append(f"حصة 0DTE من الحجم={OF['share0'] * 100:.0f}%")
    ex.append(f"تركّز أعلى5={OF['top5'] * 100:.0f}%")
    if OF["quality"] is not None: ex.append(f"جودة التسعير ATM={OF['quality'] * 100:.0f}%")
    L.append("OPTIONS_STATS: " + " | ".join(ex))
    p = OF["prem"]
    if p["call"] + p["put"] > 0:
        L.append(f"OPTIONS_PREMIUM[{'0DTE' if OF['nearest'] == 0 else 'أقرب انتهاء'}، حجم×سعر]: كول ${p['call'] / 1e6:.1f}M / بوت ${p['put'] / 1e6:.1f}M "
                 f"(قرب السعر ≤0.5%: كول ${p['call_near'] / 1e6:.1f}M / بوت ${p['put_near'] / 1e6:.1f}M) — لا نعرف من البادئ بالصفقة")
    return L


# ------------------------------------------------------------------ تعديل جودة الإشارة
MOMENTUM = {"ORB", "SQUEEZE", "PULLBACK", "RETEST"}
REVERSAL = {"SWEEP", "EXHAUST", "DIVERGENCE", "VWAP"}


def adjust(OF, sg):
    """تعديل صغير (|±10| كحد أقصى) على درجة الإشارة وفق سياق الأوبشن.

    هذه فرضيات من بحث الغاما والجدران، وليست مُثبتة على بياناتنا. لذلك نخزّن النتيجة في الإشارة
    ونقيس لاحقاً في /stats هل ساعدت فعلاً (حسب النظام وحسب إشارة التعديل).
    """
    if not OF:
        return 0, [], None
    adj, notes, dr, name = 0, [], sg["dir"], sg["name"]
    atr_like = sg["risk"] / 1.2 if sg.get("risk") else 0.0
    # 1) نظام الغاما
    if OF["regime"] == "NEG":
        if name in MOMENTUM: adj += 4; notes.append("غاما سالبة تدعم الزخم")
        elif name in REVERSAL: adj -= 3; notes.append("⚠️ غاما سالبة: الارتدادات أخطر")
    else:
        if name in REVERSAL: adj += 3; notes.append("غاما موجبة تدعم الارتداد")
        elif name in MOMENTUM: adj -= 3; notes.append("⚠️ غاما موجبة قد تخمّد الاختراق")
    # 2) الجدران: عائق أمام الهدف1 أو دعم خلف الدخول
    ent, t1 = sg["entry"], sg["t1"]
    if dr == 1:
        wall = OF.get("call_wall")
        if wall and ent < wall <= t1 and (wall - ent) < 0.8 * abs(t1 - ent): adj -= 6; notes.append(f"⚠️ جدار كول {wall:.2f} قبل الهدف1")
        sup = OF.get("put_wall")
        if sup and sup < ent and (ent - sup) <= 1.2 * max(atr_like, 1e-9) * 1.5: adj += 3; notes.append(f"جدار بوت {sup:.2f} خلف الدخول")
    else:
        wall = OF.get("put_wall")
        if wall and t1 <= wall < ent and (ent - wall) < 0.8 * abs(ent - t1): adj -= 6; notes.append(f"⚠️ جدار بوت {wall:.2f} قبل الهدف1")
        sup = OF.get("call_wall")
        if sup and sup > ent and (sup - ent) <= 1.2 * max(atr_like, 1e-9) * 1.5: adj += 3; notes.append(f"جدار كول {sup:.2f} فوق الدخول")
    # 3) الحركة المتوقعة المستهلكة
    u = OF.get("em_used")
    if u is not None and u > 1.0:
        if name in MOMENTUM: adj -= 5; notes.append(f"⚠️ تحرك السعر {u * 100:.0f}% من الحركة اليومية المتوقعة")
        else: adj += 2; notes.append("تجاوز الحركة المتوقعة يدعم الارتداد")
    adj = int(max(-10, min(10, adj)))
    return adj, notes, OF["regime"]
