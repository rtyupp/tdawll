# -*- coding: utf-8 -*-
"""
هياكل أوبشن محدّدة المخاطرة تُبنى على السلسلة الحية (CBOE المتأخرة) — مقتبسة فكرةً ومنطقاً من OSBT
(https://github.com/DrEMG/osbt, MIT License, Copyright (c) 2026 DrEMG):
  • اختيار الأسعار بالدلتا أو المسافة أو أقرب سعر هدف (Strike selection).
  • نموذج الانزلاق: نسبة من نصف السبريد + عمولة لكل عقد (slippage_pct, commission_per_contract).
  • الهامش للمراكز المحدّدة المخاطرة = أقصى خسارة.
  • الجريكس الصافية بـ BSM.
الإضافة هنا: قيمة الهيكل عند وصول السعر لهدف الإشارة أو وقفها خلال ساعة (إعادة تسعير BSM).
"""
import math
from datetime import datetime

import numpy as np

from . import bsm

SLIP_PCT = 25.0            # نفس افتراض OSBT: 25% من نصف السبريد
COMMISSION = 0.65          # $ لكل عقد لكل ساق
R_FREE = 0.045


def _by(cs, dte, typ):
    return sorted([c for c in cs if c["dte"] == dte and c["t"] == typ and c["bid"] > 0 and c["ask"] > 0], key=lambda c: c["k"])


def pick_expiry(cs, mins_left, min_minutes=90):
    """0DTE إن بقي وقت كافٍ لفكرة ساعتين، وإلا أقرب انتهاء بعده."""
    dtes = sorted({c["dte"] for c in cs})
    if 0 in dtes and mins_left >= min_minutes:
        return 0
    for d in dtes:
        if d >= 1:
            return d
    return dtes[0] if dtes else None


def _delta(c, spot, T):
    if c.get("delta") is not None and c["delta"] != 0:
        return c["delta"]
    return bsm.BSMCalculator.calculate(spot, c["k"], T, R_FREE, c["iv"], c["t"])["delta"]


def by_delta(rows, target_abs, spot, T):
    live = [c for c in rows if c["iv"] > 0]
    return min(live, key=lambda c: abs(abs(_delta(c, spot, T)) - target_abs), default=None)


def nearest_strike(rows, price):
    return min(rows, key=lambda c: abs(c["k"] - price), default=None)


def step_of(rows):
    ks = sorted({c["k"] for c in rows})
    gaps = [b - a for a, b in zip(ks, ks[1:]) if b - a > 0]
    return min(gaps) if gaps else 1.0


def leg(c, side, qty=1):
    return {"c": c, "side": side, "qty": qty}


def _entry_price(l, slip):
    """شراء: الأسك + انزلاق، بيع: البيد − انزلاق (انزلاق = نسبة من نصف السبريد)."""
    c = l["c"]; half = (c["ask"] - c["bid"]) / 2
    return c["ask"] + slip * half if l["side"] > 0 else max(c["bid"] - slip * half, 0.0)


def evaluate(legs, spot, mins_left, slip_pct=SLIP_PCT, comm=COMMISSION, scenarios=None, scn_minutes=60):
    """يقيّم الهيكل: التكلفة، أقصى ربح/خسارة، التعادل، احتمال الربح (لوغاريتمي)، الجريكس، وP&L للسيناريوهات."""
    if not legs:
        return None
    slip = slip_pct / 100.0
    prices = [_entry_price(l, slip) for l in legs]
    net = sum(l["side"] * l["qty"] * p for l, p in zip(legs, prices))          # >0 دفع (debit)، <0 استلام (credit)
    comms = comm * sum(l["qty"] for l in legs)
    dte = legs[0]["c"]["dte"]
    T = max(dte * 1440 + mins_left, 15) / 525600.0
    S = np.linspace(spot * 0.85, spot * 1.15, 3001)

    def payoff(s):
        tot = np.zeros_like(s)
        for l in legs:
            k = l["c"]["k"]
            v = np.maximum(s - k, 0) if l["c"]["t"] == "call" else np.maximum(k - s, 0)
            tot += l["side"] * l["qty"] * v
        return tot - net
    pnl = payoff(S) * 100 - comms
    max_p, max_l = float(pnl.max()), float(pnl.min())
    # شكل الأطراف: هل الخسارة/الربح غير محدودين؟
    ends = payoff(np.array([spot * 0.01, spot * 5.0])) * 100 - comms
    unlimited_loss = bool(ends.min() < max_l - 1e-6 or ends.min() < -1e9)
    unlimited_prof = bool(ends.max() > max_p + 1e-6)
    sign = np.sign(pnl)
    be = [float(S[i] + (S[i + 1] - S[i]) * (-pnl[i]) / (pnl[i + 1] - pnl[i])) for i in range(len(S) - 1)
          if sign[i] != 0 and sign[i] != sign[i + 1] and pnl[i + 1] != pnl[i]]
    ivs = [l["c"]["iv"] for l in legs if l["c"]["iv"] > 0]
    sigma = float(np.mean(ivs)) if ivs else 0.2
    sq = sigma * math.sqrt(T)
    edges = (np.log(S / spot) + 0.5 * sigma ** 2 * T) / sq
    cdf = bsm.norm_cdf_arr(edges)
    p_i = np.diff(cdf, prepend=0.0)                                           # كتلة الاحتمال لكل نقطة شبكة
    p_i = p_i / max(p_i.sum(), 1e-12)
    pop = float(p_i[pnl > 0].sum())
    ev = float((p_i * pnl).sum())

    # الجريكس الصافية عبر BSM (هنا theta بالدولار/يوم، vega بالدولار لكل 1% IV)
    gk = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    for l in legs:
        c = l["c"]; Tl = max(c["dte"] * 1440 + mins_left, 15) / 525600.0
        g = bsm.BSMCalculator.calculate(spot, c["k"], Tl, R_FREE, c["iv"] or sigma, c["t"])
        m = 100 * l["qty"] * l["side"]
        gk["delta"] += g["delta"] * m; gk["gamma"] += g["gamma"] * m
        gk["theta"] += g["theta_per_day"] * m; gk["vega"] += g["vega_per_pct"] * m

    scn = {}
    for nm, price in (scenarios or {}).items():
        val = 0.0
        for l, p in zip(legs, prices):
            c = l["c"]; Tl = max(c["dte"] * 1440 + mins_left - scn_minutes, 5) / 525600.0
            v = bsm.BSMCalculator.price(price, c["k"], Tl, R_FREE, c["iv"] or sigma, c["t"])
            val += l["side"] * l["qty"] * (v - p)
        scn[nm] = val * 100 - comms * 2                                       # عمولة دخول وخروج
    return dict(net=net, cost=net * 100 + comms, max_profit=max_p, max_loss=max_l, unlimited_loss=unlimited_loss,
                unlimited_profit=unlimited_prof, breakevens=be, pop=pop, ev=ev, margin=abs(max_l) if not unlimited_loss else None,
                greeks=gk, scn=scn, comms=comms, dte=dte, sigma=sigma, prices=prices)


def _name(legs):
    return " / ".join(f"{'+' if l['side'] > 0 else '-'}{l['c']['k']:g}{'C' if l['c']['t'] == 'call' else 'P'}" for l in legs)


def build(chain, mins_left, sg=None, OF=None, width_hint=None):
    """يبني حتى 3 هياكل للإشارة (أو هيكلاً محايداً بلا إشارة). كل هيكل {kind,title,legs,ev}."""
    cs, spot = chain["contracts"], chain["spot"]
    dte = pick_expiry(cs, mins_left)
    if dte is None:
        return [], None
    calls, puts = _by(cs, dte, "call"), _by(cs, dte, "put")
    if len(calls) < 4 or len(puts) < 4:
        return [], dte
    T = max(dte * 1440 + mins_left, 15) / 525600.0
    step = step_of(calls)
    out = []

    def pick(rows, price):
        return nearest_strike(rows, price)

    def other(rows, k, offset):
        return min((c for c in rows if abs(c["k"] - (k + offset)) < step * 0.51), key=lambda c: abs(c["k"] - (k + offset)), default=None)

    if sg:
        dr = sg["dir"]
        w = width_hint or max(step, min(step * 5, round(abs(sg["t2"] - sg["entry"]) / step) * step))
        rows = calls if dr == 1 else puts
        typ = "call" if dr == 1 else "put"
        # (1) فرق دفع اتجاهي: شراء قرب ATM وبيع عند الهدف/العرض
        a = by_delta(rows, 0.50, spot, T)
        b = other(rows, a["k"], w * dr) if a else None
        if a and b:
            out.append(dict(kind="debit", title=f"{'Bull Call' if dr == 1 else 'Bear Put'} Debit Spread", legs=[leg(a, 1), leg(b, -1)]))
        # (2) فرق ائتماني: بيع بدلتا ~0.30 على الجهة المعاكسة للاتجاه (يربح إن بقي السعر معنا)
        crow = puts if dr == 1 else calls
        s_ = by_delta(crow, 0.30, spot, T)
        l_ = other(crow, s_["k"], -w * dr) if s_ else None
        if s_ and l_:
            out.append(dict(kind="credit", title=f"{'Bull Put' if dr == 1 else 'Bear Call'} Credit Spread", legs=[leg(s_, -1), leg(l_, 1)]))
        # (3) شراء مباشر
        n = by_delta(rows, 0.45, spot, T)
        if n:
            out.append(dict(kind="single", title=f"Long {typ.title()} (ثيتا عالية)", legs=[leg(n, 1)]))
    else:
        em = OF["em"] if OF and OF.get("em") else spot * 0.005
        w = width_hint or max(step, step * 3)
        sp = pick(puts, spot - em); sc = pick(calls, spot + em)
        lp = other(puts, sp["k"], -w) if sp else None; lc = other(calls, sc["k"], w) if sc else None
        if sp and sc and lp and lc:
            out.append(dict(kind="condor", title="Iron Condor (خارج الحركة المتوقعة)", legs=[leg(lp, 1), leg(sp, -1), leg(sc, -1), leg(lc, 1)]))
        # فراشة مكسورة الجناح (بوت) مثل OSBT: ميل صاعد خفيف
        m_ = pick(puts, spot - em * 0.5)
        if m_:
            up = other(puts, m_["k"], w); dn = other(puts, m_["k"], -2 * w)
            if up and dn:
                out.append(dict(kind="bwb", title="Broken-Wing Put Butterfly (جناح سفلي أوسع)", legs=[leg(up, 1), leg(m_, -2), leg(dn, 1)]))
    return out, dte


def describe(st, sg=None):
    """نص قصير لتيليجرام (بدون HTML خاص)."""
    e = st["ev"]
    cost = st["cost"]
    side = "دفع" if cost > 0 else "استلام"
    parts = [f"{side} ${abs(cost):.0f}"]
    parts.append("أقصى ربح غير محدود" if st["unlimited_profit"] else f"أقصى ربح ${st['max_profit']:.0f}")
    parts.append("أقصى خسارة غير محدودة ⚠️" if st["unlimited_loss"] else f"أقصى خسارة ${abs(st['max_loss']):.0f}")
    be = " / ".join(f"{b:.2f}" for b in st["breakevens"][:2]) or "—"
    g = st["greeks"]
    lines = [" · ".join(parts), f"تعادل {be} · احتمال الربح ≈ {st['pop'] * 100:.0f}% (نموذج لوغاريتمي)",
             f"دلتا {g['delta']:+.1f} · " + ("ثيتا: تتسارع بشدة نحو الإغلاق (0DTE)" if st["dte"] == 0 else f"ثيتا {g['theta']:+.1f}$/يوم")
             + f" · فيغا {g['vega']:+.1f}$/1%IV"]
    if st["scn"]:
        lines.append("إذا " + " · ".join(f"{k}: {v:+.0f}$" for k, v in st["scn"].items()))
    return "\n".join(lines)
