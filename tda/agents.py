# -*- coding: utf-8 -*-
"""
وكلاء التداول — نسخة خفيفة من فلسفة TauricResearch/TradingAgents (Apache-2.0) فوق Gemini نفسه.

ما أُخذ منهم (والفكرة، لا الإطار الثقيل):
  • نقاش ثور/دب → مدير أبحاث → لجنة مخاطر (جريء/متحفظ/محايد) → مدير محفظة.
  • تقييم من 5 درجات بمُحلل صارم، و REVIEW بدل Hold مختلق عند غموض الحكم.
  • مخرجات منظمة (JSON) مع تراجع آمن.
  • مبدأ «المصدر المفقود ليس صمتاً» و «المتحدث الأول لا يختلق ردّ خصمه».
  • ذاكرة القرارات: كل نتيجة تُحوَّل إلى درس قصير يُحقن في القرارات القادمة (زمنياً بلا تسريب).
  • تأريض الأسعار: أي مستوى تقترحه اللجنة يُرفض إن لم يكن منطقياً بالنسبة للسعر وATR.
  • الموقف الحالي (position): لا نفترض أن حسابك فارغ إن لم تخبرنا.

دمجتُ «المتداول» في مدير المحفظة ودمجتُ المتحدثين الثلاثة للمخاطر في طلب واحد لتوفير حصة Gemini المجانية.
"""
import html, time, logging
from datetime import datetime

from . import llm
from .rating import RATINGS_5_TIER as RATINGS, RATING_REVIEW, parse_rating

log = logging.getLogger("tdawll.agents")

RATING_AR = {"Buy": "ميل صاعد قوي", "Overweight": "ميل صاعد", "Hold": "محايد/لا أفضلية",
             "Underweight": "ميل هابط", "Sell": "ميل هابط قوي", RATING_REVIEW: "يحتاج مراجعة"}
CONF_AR = {"low": "ثقة منخفضة", "medium": "ثقة متوسطة", "high": "ثقة عالية"}
BAND_AR = {"Bullish": "صاعد", "Mildly Bullish": "صاعد خفيف", "Neutral": "محايد", "Mixed": "متضارب",
           "Mildly Bearish": "هابط خفيف", "Bearish": "هابط"}


def rating_dir(r):
    return {"Buy": 1, "Overweight": 1, "Sell": -1, "Underweight": -1, "Hold": 0}.get(r)


GUIDE = ("You work inside a Telegram trading assistant for the S&P 500 ({label}), 5-minute timeframe.\n"
         "Hard rules:\n"
         "1. Use ONLY the evidence in the CARD. Every price or number you cite must appear in it. Never invent news, levels or data.\n"
         "2. A source marked unavailable is NOT silence: say it is missing and lower confidence accordingly.\n"
         "3. Entry/stop/targets computed by code are fixed. You may judge them, not replace them with invented numbers.\n"
         "4. Weigh arguments on merit, independent of speaking order. Conflict alone is not a reason to Hold: commit to the "
         "stronger side, sized by how decisively it wins. Choose Hold only if the evidence is balanced or too thin.\n"
         "5. A setup's historical stats describe a small sample, not a guarantee. Lessons are single-trade reviews, not proven rules.\n"
         "6. Ratings are about the direction of {label} over roughly the next 2 hours (24 bars): Buy/Overweight = long bias, "
         "Sell/Underweight = short bias, Hold = no edge.")


def _sys(label):
    return GUIDE.format(label=label)


def _opening(text, opponent):
    """المتحدث الأول لا يستلم فراغاً فيختلق حجة خصمه (#1176 في TradingAgents)."""
    t = (text or "").strip()
    return t or f"(The {opponent} has not spoken yet — open the debate with your own case.)"


def _absent(text, source):
    t = (text or "").strip()
    return t or f"(No {source} available: it is missing, not an empty finding.)"


def _enum(vals):
    return {"type": "STRING", "enum": list(vals)}


STR = {"type": "STRING"}
CONF = _enum(["low", "medium", "high"])
NUM = {"type": "NUMBER", "nullable": True}
SCHEMA_LITE = {"type": "OBJECT", "properties": {"bull": STR, "bear": STR, "rating": _enum(RATINGS), "confidence": CONF,
                                                 "verdict": STR, "invalidation": STR,
                                                 "decision": _enum(["approve", "reduce", "reject", "wait"]),
                                                 "regime": _enum(["trend", "range", "high_volatility", "unclear"]),
                                                 "timeframe_alignment": _enum(["aligned", "mixed", "conflicting"]),
                                                 "data_quality": _enum(["good", "degraded", "stale"])},
               "required": ["bull", "bear", "rating", "confidence", "verdict", "invalidation"]}
SCHEMA_PLAN = {"type": "OBJECT", "properties": {"recommendation": _enum(RATINGS), "rationale": STR, "strategic_actions": STR},
               "required": ["recommendation", "rationale", "strategic_actions"]}
SCHEMA_RISK = {"type": "OBJECT", "properties": {"aggressive": STR, "conservative": STR, "neutral": STR},
               "required": ["aggressive", "conservative", "neutral"]}
SCHEMA_PM = {"type": "OBJECT", "properties": {"rating": _enum(RATINGS), "confidence": CONF, "summary": STR, "thesis": STR,
                                               "key_risk": STR, "invalidation": STR, "entry": NUM, "stop": NUM, "target": NUM,
                                               "horizon": STR},
             "required": ["rating", "confidence", "summary", "thesis", "key_risk", "invalidation"]}
SCHEMA_SENT = {"type": "OBJECT", "properties": {"overall_band": _enum(BAND_AR.keys()), "overall_score": {"type": "NUMBER"},
                                                 "confidence": CONF, "narrative": STR},
               "required": ["overall_band", "overall_score", "confidence", "narrative"]}


def _clean_rating(obj, *text_keys):
    """التقييم من JSON، وإلا من النص بالمُحلل الصارم، وإلا REVIEW (لا Hold مختلق)."""
    r = (obj or {}).get("rating") or (obj or {}).get("recommendation")
    if r in RATINGS:
        return r
    return parse_rating(" ".join(str((obj or {}).get(k, "")) for k in text_keys))


def _sig_block(sig):
    if not sig:
        return "No code-computed setup: judge the market direction from the card only."
    return (f"CODE-COMPUTED SETUP (fixed): {sig['name']} | direction {'LONG' if sig['dir'] > 0 else 'SHORT'} | grade {sig['grade']} "
            f"score {sig['score']}/100\nentry {sig['entry']:.2f} stop {sig['stop']:.2f} target1 {sig['t1']:.2f} ({sig['rr1']:.1f}R) "
            f"target2 {sig['t2']:.2f}\ncode reasons: {'; '.join(sig['notes'])}")


# ============================================================ لجنة سريعة (طلب واحد)
def committee_lite(card, label, sig, lessons="", position=""):
    prompt = (f"CARD:\n{card}\n\n{_sig_block(sig)}\n\n{position}\n\n{lessons or 'No past lessons yet.'}\n\n"
              "TASK: run a fast investment committee on this setup, then return JSON.\n"
              "- bull: the strongest honest case FOR taking the trade (Arabic, max 30 words).\n"
              "- bear: the strongest case AGAINST it, attacking this setup's specific weaknesses from the card: daily-trend conflict, "
              "VIX, news, nearby levels, thin or negative backtest, social extremes (Arabic, max 30 words).\n"
              "- rating: your verdict on the market direction (Buy/Overweight/Hold/Underweight/Sell) after weighing both.\n"
              "- confidence: low/medium/high based on how decisively one side wins and on data quality.\n"
              "- decision: exactly approve/reduce/reject/wait; reject or wait if data is stale or the setup is not actionable.\n"
              "- regime: exactly trend/range/high_volatility/unclear.\n"
              "- timeframe_alignment: exactly aligned/mixed/conflicting using daily versus 5-minute evidence.\n"
              "- data_quality: exactly good/degraded/stale; never call delayed or missing data good.\n"
              "- verdict: one actionable Arabic sentence (max 22 words).\n"
              "- invalidation: where the idea is wrong, using a price from the card (Arabic, max 14 words).")
    o = llm.generate_json(_sys(label), prompt, SCHEMA_LITE, deep=False, tokens=420,
                          required=("bull", "bear", "verdict"))
    if not o:
        return None
    r = _clean_rating(o, "verdict")
    d = rating_dir(r)
    agree = None if (d is None or not sig) else (1 if d == sig["dir"] else 0 if d == 0 else -1)
    return {"bull": str(o.get("bull", "")), "bear": str(o.get("bear", "")), "rating": r, "dir": d,
            "confidence": o.get("confidence") if o.get("confidence") in CONF_AR else "low",
            "verdict": str(o.get("verdict", "")), "invalidation": str(o.get("invalidation", "")), "agree": agree,
            "decision": o.get("decision") if o.get("decision") in {"approve", "reduce", "reject", "wait"} else "wait",
            "regime": o.get("regime") if o.get("regime") in {"trend", "range", "high_volatility", "unclear"} else "unclear",
            "timeframe_alignment": o.get("timeframe_alignment") if o.get("timeframe_alignment") in {"aligned", "mixed", "conflicting"} else "mixed",
            "data_quality": o.get("data_quality") if o.get("data_quality") in {"good", "degraded", "stale"} else "degraded"}


def lite_block(cm):
    """نص تيليجرام (HTML آمن) لنتيجة اللجنة السريعة."""
    e = html.escape
    tag = {1: "✅ توافق مع الإشارة", 0: "⚪ محايد", -1: "⛔ تعارض مع الإشارة", None: ""}[cm["agree"]]
    return (f"⚖️ <b>اللجنة</b>: {RATING_AR.get(cm['rating'], cm['rating'])} · {CONF_AR.get(cm['confidence'], '')} · {cm.get('decision', 'wait')} {tag}\n"
            f"🧭 النظام: {cm.get('regime', 'unclear')} · توافق الفريمات: {cm.get('timeframe_alignment', 'mixed')} · جودة البيانات: {cm.get('data_quality', 'degraded')}\n"
            f"🐂 {e(cm['bull'])}\n🐻 {e(cm['bear'])}\n💬 {e(cm['verdict'])}\n🚫 الإبطال: {e(cm['invalidation'])}")


# ============================================================ لجنة كاملة (نقاش حقيقي)
def _say(system, prompt, tokens=380, deep=False):
    t = llm.generate_text(system, prompt, deep=deep, tokens=tokens)
    return t.strip() if t else None


def ground_levels(pm, price, atr, d):
    """أسعار اللجنة تُقبل فقط إن كانت منطقية بالنسبة للسعر وATR (تأريض المتداول في TradingAgents)."""
    out = {"entry": None, "stop": None, "target": None}
    if not d or not price or not atr:
        return out

    def num(x):
        try: return float(x)
        except (TypeError, ValueError): return None
    e, s, t = num(pm.get("entry")), num(pm.get("stop")), num(pm.get("target"))
    if e is not None and abs(e - price) <= atr: out["entry"] = e
    ref = out["entry"] or price
    if s is not None and 0.5 * atr <= (ref - s) * d <= 3.5 * atr: out["stop"] = s
    if t is not None and 0.8 * atr <= (t - ref) * d <= 7 * atr: out["target"] = t
    return out


def full_debate(card, label, price, atr, sig=None, lessons="", position="", rounds=1):
    system = _sys(label)
    ctx = f"CARD:\n{card}\n\n{_sig_block(sig)}\n\n{position}"
    res = {"calls": 0, "errors": [], "history": "", "ts": time.time()}
    hist, last_bull, last_bear = "", "", ""
    for _ in range(rounds):
        bull = _say(system, f"You are the Bull Analyst. Build a strong, evidence-based case for being LONG {label} over the next ~2 hours: "
                            "trend/momentum/structure, supportive macro, news, sentiment and prediction-market data. Directly rebut the bear's "
                            "specific points with data from the card. Conversational, max 170 words.\n\n"
                            f"{ctx}\n\nDEBATE SO FAR:\n{hist or '(none)'}\n\nLAST BEAR ARGUMENT:\n{_opening(last_bear, 'bear analyst')}")
        res["calls"] += 1
        if bull: hist += f"\nBull Analyst: {bull}"; last_bull = bull
        else: res["errors"].append("bull")
        bear = _say(system, f"You are the Bear Analyst. Build a strong, evidence-based case AGAINST being long (or for being SHORT) {label} over "
                            "the next ~2 hours: risks, conflicts between timeframes, weak volume, nearby resistance, macro/news/sentiment threats. "
                            "Directly rebut the bull's specific points with data from the card. Conversational, max 170 words.\n\n"
                            f"{ctx}\n\nDEBATE SO FAR:\n{hist or '(none)'}\n\nLAST BULL ARGUMENT:\n{_opening(last_bull, 'bull analyst')}")
        res["calls"] += 1
        if bear: hist += f"\nBear Analyst: {bear}"; last_bear = bear
        else: res["errors"].append("bear")
    res.update(bull=last_bull, bear=last_bear, history=hist.strip())
    if not (last_bull or last_bear):
        res["errors"].append("debate-empty")
        return res

    plan = llm.generate_json(system, f"{ctx}\n\nDEBATE:\n{_absent(hist, 'debate')}\n\nYou are the Research Manager. Decide which side won and "
                             "give the trader a plan. recommendation must be exactly one of Buy/Overweight/Hold/Underweight/Sell. "
                             "rationale: which arguments decided it (max 90 words). strategic_actions: concrete steps sized against a "
                             "standard allocation (max 70 words).", SCHEMA_PLAN, deep=True, tokens=450,
                             required=("recommendation", "rationale"))
    res["calls"] += 1
    res["plan"] = plan
    plan_txt = f"{plan.get('recommendation')}: {plan.get('rationale')} Actions: {plan.get('strategic_actions')}" if plan else "(Research manager unavailable.)"
    if not plan: res["errors"].append("plan")

    risk = llm.generate_json(system, f"{ctx}\n\nRESEARCH PLAN:\n{plan_txt}\n\nWrite three short voices (max 80 words each) reacting to this plan: "
                             "aggressive (maximize reward; say if the plan is too timid), conservative (protect capital; what can go wrong; "
                             "demand smaller size or skipping), neutral (balanced, practical sizing and timing).", SCHEMA_RISK,
                             deep=False, tokens=520, required=("aggressive", "conservative", "neutral"))
    res["calls"] += 1
    res["risk"] = risk
    if not risk: res["errors"].append("risk")
    risk_txt = (f"Aggressive: {risk['aggressive']}\nConservative: {risk['conservative']}\nNeutral: {risk['neutral']}" if risk
                else "(Risk committee unavailable.)")

    pm = llm.generate_json(system, f"{ctx}\n\nRESEARCH PLAN:\n{plan_txt}\n\nRISK COMMITTEE:\n{risk_txt}\n\n"
                           f"PAST LESSONS:\n{lessons or '(none yet)'}\n\n"
                           "You are the Portfolio Manager (also acting as the trader). Deliver the final decision. "
                           "rating: exactly one of Buy/Overweight/Hold/Underweight/Sell. confidence: low/medium/high. "
                           "summary: the call and how to act on it (Arabic, max 45 words). thesis: the evidence that decided it and what "
                           "would change it (Arabic, max 70 words). key_risk: the single biggest risk (Arabic, max 25 words). "
                           "invalidation: where the idea is wrong, with a price from the card (Arabic, max 18 words). "
                           "entry/stop/target: absolute price levels in dollars chosen from the card's levels or code setup, never percentages; "
                           "use null if you cannot state them. horizon: Arabic, short.", SCHEMA_PM, deep=True, tokens=700,
                           required=("rating", "summary", "thesis"))
    res["calls"] += 1
    res["pm"] = pm
    if not pm:
        res["errors"].append("pm")
        res["rating"], res["dir"] = RATING_REVIEW, None
        return res
    res["rating"] = _clean_rating(pm, "summary", "thesis")
    res["dir"] = rating_dir(res["rating"])
    res["grounded"] = ground_levels(pm, price, atr, res["dir"])
    return res


def debate_text(res, label, price):
    if not res.get("pm"):
        return ("⚖️ <b>لجنة التداول</b>\n⚠️ تعذّر إكمال القرار النهائي (حصة Gemini أو خلل مؤقت). "
                f"الأجزاء الناقصة: {', '.join(res.get('errors', [])) or 'غير معروف'}. أعد المحاولة بعد دقيقة.")
    e, pm = html.escape, res["pm"]
    g = res.get("grounded") or {}
    lv = " · ".join(f"{k} {v:.2f}" for k, v in (("دخول", g.get("entry")), ("وقف", g.get("stop")), ("هدف", g.get("target"))) if v)
    out = [f"⚖️ <b>لجنة التداول</b> · {label} ${price:.2f}",
           f"<b>الحكم النهائي: {RATING_AR.get(res['rating'], res['rating'])}</b> ({res['rating']}) · {CONF_AR.get(pm.get('confidence'), '')}",
           "", e(str(pm.get("summary", ""))), "", f"📌 {e(str(pm.get('thesis', '')))}",
           f"⚠️ أكبر خطر: {e(str(pm.get('key_risk', '')))}", f"🚫 الإبطال: {e(str(pm.get('invalidation', '')))}"]
    if lv: out.append(f"🎯 مستويات اللجنة (تم التحقق منها): {lv}")
    elif pm.get("entry") or pm.get("stop") or pm.get("target"): out.append("🎯 مستويات اللجنة رُفضت لأنها غير منطقية بالنسبة للسعر وATR")
    if pm.get("horizon"): out.append(f"⏱️ الأفق: {e(str(pm['horizon']))}")
    if res["errors"]: out.append(f"\n<i>ملاحظة: أجزاء لم تكتمل ({', '.join(res['errors'])}) فخُفّضت الثقة.</i>")
    out.append(f"\n<i>{res['calls']} طلب Gemini · النقاش الكامل في الملف المرفق</i>")
    return "\n".join(out)


# ============================================================ محلل المزاج
_sent_cache = {}


def sentiment_report(ext_lines, news, label):
    key = "s"
    hit = _sent_cache.get(key)
    if hit and time.time() - hit[0] < 900:
        return hit[1]
    heads = "\n".join(f"- {n['h']} ({n['s']})" for n in news[:8]) or "(no headlines available)"
    prompt = ("You are a market sentiment analyst. Produce a sentiment read for " + label + " from the sources below only.\n"
              "Best practices: read the StockTwits bull/bear ratio as a fast retail signal (>=90/10 may mean over-extension and contrarian "
              "risk; 50/50 is uncertainty; judge sample size). Look for cross-source divergences (news vs retail). Read Reddit for "
              "substance, not engagement. Distinguish events (headlines) from opinions (posts). Name recurring themes, catalysts and risks. "
              "Be honest about gaps: any unavailable source lowers confidence. Past sentiment is not predictive.\n\n"
              "SOURCES:\n" + "\n".join(ext_lines) + f"\n\nHEADLINES:\n{heads}\n\n"
              "overall_band: Bullish/Mildly Bullish/Neutral/Mixed/Mildly Bearish/Bearish (Mixed when sources clearly disagree; "
              "Neutral only when all are genuinely silent). overall_score: 0 (max bearish) to 10 (max bullish), consistent with the band. "
              "narrative: Arabic, max 110 words, source-by-source with divergences and key risks.")
    o = llm.generate_json(_sys(label), prompt, SCHEMA_SENT, deep=False, tokens=450,
                          required=("overall_band", "overall_score", "narrative"))
    if o:
        _sent_cache[key] = (time.time(), o)
    return o


# ============================================================ الذاكرة: دروس من النتائج
def reflect(rec):
    """درس من 2-4 جمل بعد حسم الصفقة (على نمط Reflector في TradingAgents)."""
    prompt = ("You are a trading analyst reviewing your own past alert now that the outcome is known.\n"
              f"Setup: {rec['setup']} | direction {'LONG' if rec['dir'] > 0 else 'SHORT'} | grade {rec['grade']} score {rec['score']}\n"
              f"Code reasons: {'; '.join(rec.get('notes', []))}\n"
              f"Committee verdict at the time: {rec.get('ai_rating') or 'none'}"
              f"{' (agreed)' if rec.get('ai_agree') == 1 else ' (disagreed)' if rec.get('ai_agree') == -1 else ''}\n"
              f"Outcome: {rec['status']} = {rec['R']:+.2f}R | best excursion {rec.get('mfe', 0):+.2f}R | worst {rec.get('mae', 0):+.2f}R | "
              f"{rec.get('bars', 0)} bars to resolution.\n\n"
              "Write exactly 2-4 sentences of plain Arabic prose (no bullets, no markdown). In order: (1) what the outcome shows about the "
              "call, and say plainly if one trade is too little evidence to judge; (2) which part of the thesis it supports or undercuts; "
              "(3) one concrete lesson for the next similar setup. Be terse: this is stored and re-read by future decisions.")
    t = llm.generate_text("You review past trading decisions honestly and tersely.", prompt, deep=False, tokens=260)
    return t.strip()[:700] if t else None


def lessons_context(lessons, setup, n_same=4, n_cross=2):
    """الدروس المحسومة فقط (لا تسريب مستقبلي): نفس السيناريو أولاً ثم دروس عامة."""
    done = [x for x in lessons if x.get("text")]
    if not done:
        return ""
    same = [x for x in reversed(done) if x["setup"] == setup][:n_same]
    cross = [x for x in reversed(done) if x["setup"] != setup][:n_cross]

    def fmt(x):
        return f"- [{x['date']} {x['setup']} {'long' if x['dir'] > 0 else 'short'} {x['grade']}: {x['R']:+.2f}R] {x['text']}"
    parts = []
    if same: parts += [f"Past lessons for {setup} (most recent first):"] + [fmt(x) for x in same]
    if cross: parts += ["Recent lessons from other setups:"] + [fmt(x) for x in cross]
    return "\n".join(parts)


# ============================================================ موقفك الحالي
def position_context(pos, price, atr):
    """لا نفترض أن حسابك فارغ إن لم تُخبرنا (نفس مبدأ PortfolioContext)."""
    if pos is None:
        return ("Position context: not provided. You do not know the trader's current holdings, so do not assume a flat book; "
                "give guidance the trader can apply to their own position.")
    if pos.get("side") == "flat":
        return "Position context: the trader is FLAT (no open position)."
    d = 1 if pos["side"] == "long" else -1
    pts = (price - pos["entry"]) * d
    risk = abs(pos["entry"] - pos["stop"]) if pos.get("stop") else None
    r = f" = {pts / risk:+.2f}R" if risk else ""
    stop = f", stop {pos['stop']:.2f}" if pos.get("stop") else ", no stop stated"
    return (f"Position context: the trader holds a {pos['side'].upper()} from {pos['entry']:.2f}{stop}. "
            f"Open P/L {pts:+.2f} points{r}. Address managing THIS position (hold / trim / exit / move stop), not only new entries.")


# ============================================================ تقرير HTML
def report_html(label, price, res, card, sent=None):
    e = html.escape
    pm = res.get("pm") or {}
    g = res.get("grounded") or {}
    plan = res.get("plan") or {}
    risk = res.get("risk") or {}

    def sec(title, body):
        return f"<section><h2>{e(title)}</h2>{body}</section>" if body else ""

    def p(t):
        return "<p>" + e(str(t)).replace("\n", "<br>") + "</p>" if t else ""
    lv = " · ".join(f"{k} {v:.2f}" for k, v in (("دخول", g.get("entry")), ("وقف", g.get("stop")), ("هدف", g.get("target"))) if v)
    body = [f"<h1>⚖️ لجنة التداول — {e(label)} ${price:.2f}</h1>",
            f"<div class='badge'>{e(RATING_AR.get(res.get('rating'), str(res.get('rating'))))} · {e(CONF_AR.get(pm.get('confidence'), ''))}</div>",
            sec("الملخص", p(pm.get("summary"))), sec("الأطروحة", p(pm.get("thesis"))),
            sec("أكبر خطر", p(pm.get("key_risk"))), sec("الإبطال", p(pm.get("invalidation"))),
            sec("مستويات اللجنة", p(lv)),
            sec("المزاج", p(f"{BAND_AR.get(sent['overall_band'], sent['overall_band'])} ({sent['overall_score']}/10): {sent['narrative']}") if sent else ""),
            "<h2>نص النقاش (بالإنجليزية لجودة الاستدلال)</h2>",
            f"<details open><summary>🐂 المحلل الصاعد</summary>{p(res.get('bull'))}</details>",
            f"<details open><summary>🐻 المحلل الهابط</summary>{p(res.get('bear'))}</details>",
            f"<details><summary>📋 مدير الأبحاث</summary>{p(plan.get('recommendation', '') + ' — ' + plan.get('rationale', '') if plan else '')}{p(plan.get('strategic_actions'))}</details>",
            f"<details><summary>🛡️ لجنة المخاطر</summary>{p('Aggressive: ' + risk.get('aggressive', '')) if risk else ''}"
            f"{p('Conservative: ' + risk.get('conservative', '')) if risk else ''}{p('Neutral: ' + risk.get('neutral', '')) if risk else ''}</details>",
            f"<details><summary>📊 البطاقة الفنية المُدخلة</summary><pre>{e(card)}</pre></details>",
            f"<footer>{res.get('calls', 0)} طلب Gemini · {datetime.now():%Y-%m-%d %H:%M} · تحليل آلي وليس توصية استثمارية</footer>"]
    return ("<!doctype html><html lang='ar' dir='rtl'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>لجنة التداول</title><style>body{font-family:system-ui,Segoe UI,Tahoma,sans-serif;background:#0b0b0f;color:#e8e8ee;"
            "max-width:820px;margin:auto;padding:18px;line-height:1.7}h1{font-size:1.3rem}h2{font-size:1.05rem;color:#9ab;margin-top:1.4rem}"
            ".badge{display:inline-block;background:#1e2a22;color:#6fe3a0;padding:6px 14px;border-radius:999px;font-weight:700}"
            "section,details{background:#14141b;border:1px solid #23232e;border-radius:12px;padding:10px 14px;margin:10px 0}"
            "summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;direction:ltr;text-align:left;font-size:.78rem;color:#aab}"
            "footer{color:#778;font-size:.8rem;margin-top:1.5rem}</style><body>" + "".join(body) + "</body></html>")
