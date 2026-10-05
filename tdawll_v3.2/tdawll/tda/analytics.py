# -*- coding: utf-8 -*-
"""
تحليلات الأداء على سلسلة R (مضاعفات المخاطرة).

مقتبسة من OSBT (MIT, © 2026 DrEMG): Sharpe/Sortino/Profit factor/Max drawdown/R²/Kelly والمونت كارلو بإعادة العينة.
تعديلان مقصودان: (1) لا نضرب بـ 52 لتحويلها «سنوياً» كما يفعل OSBT، فالتحويل الأسبوعي وهمي لصفقات يومية عشوائية التوزيع؛
نعرض النسب لكل صفقة. (2) نعرض حدود الثقة لأن العيّنات صغيرة.

الاختبار العشوائي (placebo) والتحذيرات من المقارنات المتعددة مأخوذة من منهجية بحث options-flow:
«مدخلات عشوائية مطابقة» و«نفس اليوم بتوقيت مخلوط»: هل تتفوق إشاراتك على توقيت عشوائي بنفس الاتجاه والمخاطرة؟
"""
import math

import numpy as np


def metrics(R):
    a = np.asarray(R, float)
    n = len(a)
    if n == 0:
        return None
    win = float((a > 0).mean())
    mean, std = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    neg = a[a < 0]
    down = math.sqrt(float((neg ** 2).sum()) / n) if len(neg) else 0.0
    gp, gl = float(a[a > 0].sum()), float(-a[a < 0].sum())
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    mdd = float((peak - cum).max()) if n else 0.0
    r2 = 0.0
    if n >= 3 and cum.std() > 0:
        x = np.arange(n); r2 = float(np.corrcoef(x, cum)[0, 1] ** 2)
    avg_w = float(a[a > 0].mean()) if (a > 0).any() else 0.0
    avg_l = float(-a[a < 0].mean()) if (a < 0).any() else 0.0
    if avg_l > 0 and avg_w > 0:
        b = avg_w / avg_l
        kelly = win - (1 - win) / b
    elif avg_l == 0:
        kelly = win if win > 0 else 0.0
    else:
        kelly = -1.0
    se = std / math.sqrt(n) if n > 1 else float("inf")
    return dict(n=n, win=win, mean=mean, std=std, sharpe=(mean / std) if std > 0 else 0.0,
                sortino=(mean / down) if down > 0 else 0.0, pf=(gp / gl) if gl > 0 else float("inf"),
                mdd=mdd, r2=r2, kelly=float(kelly), tstat=(mean / se) if se and se != float("inf") and se > 0 else 0.0,
                ci95=(mean - 1.96 * se, mean + 1.96 * se) if n > 1 else (mean, mean), avg_win=avg_w, avg_loss=avg_l,
                skew=float(((a - mean) ** 3).mean() / (std ** 3)) if std > 0 and n > 2 else 0.0)


def monte_carlo(R, horizon=50, n_sims=5000, seed=42):
    """إعادة عينة بإرجاع: ماذا يحدث خلال `horizon` صفقة قادمة إن بقي توزيع النتائج كما هو."""
    a = np.asarray(R, float)
    if len(a) < 8:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_sims, horizon))
    paths = np.cumsum(a[idx], axis=1)
    final = paths[:, -1]
    peak = np.maximum.accumulate(np.concatenate([np.zeros((n_sims, 1)), paths], axis=1), axis=1)[:, 1:]
    mdd = (peak - paths).max(axis=1)
    return dict(horizon=horizon, median=float(np.median(final)), p5=float(np.percentile(final, 5)), p95=float(np.percentile(final, 95)),
                p_loss=float((final < 0).mean()), mdd_med=float(np.median(mdd)), mdd_p95=float(np.percentile(mdd, 95)),
                p_dd10=float((mdd >= 10).mean()))


def sizing(kelly, mean, n):
    """نسبة مخاطرة لكل صفقة من رأس المال. كيلي الكامل خطر مع تقدير ضعيف، فنستخدم ربع كيلي ونسقفه بـ 1%."""
    if n < 20 or mean <= 0 or kelly <= 0:
        return 0.0
    return float(min(max(kelly * 0.25, 0.0025), 0.01))


# ------------------------------------------------------------------ خروج بالتتبّع
def walk_trail(sg, H, L, C, complete, atr, cost_r=0.05, trail_atr=1.0):
    """نفس الدخول والوقف، لكن بعد الهدف1 نحرّك الوقف إلى التعادل ثم نتبع القمة بـ trail_atr×ATR حتى الهدف2 أو نهاية الجلسة.
    يرجع (حالة، R) أو None إن لم تُحسم."""
    dr, entry, risk = sg["dir"], sg["entry"], sg["risk"]
    stop, t1_hit, best = sg["stop"], False, entry
    for h, l, c in zip(H, L, C):
        if not t1_hit:
            if (l <= stop) if dr == 1 else (h >= stop):
                return "SL", -1.0 - cost_r
            if (h >= sg["t1"]) if dr == 1 else (l <= sg["t1"]):
                t1_hit = True; best = h if dr == 1 else l
                stop = entry                                              # التعادل
            continue
        best = max(best, h) if dr == 1 else min(best, l)
        stop = max(stop, best - trail_atr * atr) if dr == 1 else min(stop, best + trail_atr * atr)
        if (h >= sg["t2"]) if dr == 1 else (l <= sg["t2"]):
            return "TP2", abs(sg["t2"] - entry) / risk - cost_r
        if (l <= stop) if dr == 1 else (h >= stop):
            return "TRAIL", (stop - entry) * dr / risk - cost_r
    if complete and len(C):
        return "EXP", (C[-1] - entry) * dr / risk - cost_r
    return None


# ------------------------------------------------------------------ اختبار التوقيت العشوائي
def placebo(P, trades, walk, n_sims=200, seed=7):
    """P: مصفوفات الشموع (prep). trades: صفقات حقيقية فيها i,dir,risk,rr1,name,R.
    لكل صفقة نختار شمعة عشوائية في نفس اليوم بنفس الاتجاه والمخاطرة والهدف ونعيد قياسها.
    يرجع p تجريبي = نسبة المحاكاة التي حققت متوسط R ≥ الفعلي (+1 للتصحيح)."""
    if len(trades) < 10:
        return None
    rng = np.random.default_rng(seed)
    C, H, L, mins, de, ds = P["Close"], P["High"], P["Low"], P["mins"], P["de"], P["ds"]
    pools = []
    for t in trades:
        i = t["i"]; lo, hi = int(ds[i]), int(de[i])
        cand = [j for j in range(lo, hi) if 600 <= mins[j] < 930]
        pools.append(cand or [i])
    complete_of = lambda e: mins[e] >= 945
    null = np.empty(n_sims)
    for s in range(n_sims):
        tot = 0.0
        for t, cand in zip(trades, pools):
            j = cand[int(rng.integers(0, len(cand)))]
            e = int(de[j]); dr, risk = t["dir"], t["risk"]
            ent = C[j]
            fake = {"dir": dr, "entry": ent, "stop": ent - dr * risk, "t1": ent + dr * t["rr1"] * risk, "risk": risk, "rr1": t["rr1"]}
            res = walk(fake, H[j + 1:e + 1], L[j + 1:e + 1], C[j + 1:e + 1], complete_of(e))
            tot += res[1] if res else 0.0
        null[s] = tot / len(trades)
    actual = float(np.mean([t["R"] for t in trades]))
    p = float(((null >= actual).sum() + 1) / (n_sims + 1))
    return dict(actual=actual, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)), p=p, n=len(trades))


def evidence(bt_stat, plc, live_n, live_avg, n_tests):
    """مستوى الدليل لكل سيناريو، على نمط «هرم الأدلة» في بحث options-flow:
    قياس ← ارتباط ← معايرة ← تنفيذ ← إنتاج. نحن لا ندّعي الإنتاج أبداً."""
    if not bt_stat or bt_stat["n"] < 20:
        return "🟤 عيّنة صغيرة", "تحتاج 20 صفقة على الأقل"
    p = plc["p"] if plc else None
    padj = min(1.0, p * n_tests) if p is not None else None
    if bt_stat["avgR"] <= 0:
        return "🔴 بلا أفضلية", "متوسط R سالب أو صفر في الباكتست"
    if p is None or p > 0.20:
        return "🟠 غير مثبت", f"لا يتفوق بوضوح على توقيت عشوائي (p={p:.2f})" if p is not None else "لا اختبار عشوائي"
    if padj is not None and padj <= 0.05 and live_n >= 15 and live_avg > 0:
        return "🟢 مدعوم", f"p={p:.3f} (مصحّح {padj:.2f}) + {live_n} صفقة حية موجبة"
    if padj is not None and padj <= 0.05:
        return "🟡 مرشّح قوي", f"p={p:.3f} (مصحّح {padj:.2f}) لكن لا نتائج حية كافية"
    return "🟡 مرشّح", f"p={p:.3f} لكنه لا يصمد بعد تصحيح المقارنات المتعددة (×{n_tests} → {padj:.2f})"
