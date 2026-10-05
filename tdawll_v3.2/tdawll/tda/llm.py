# -*- coding: utf-8 -*-
"""
طبقة Gemini الوحيدة في البوت. لا تغيّر النموذج: نفس النماذج ونفس المفتاح (GEMINI_API_KEY)،
والأسماء قابلة للضبط بنفس المتغيرات القديمة (GEMINI_FAST / GEMINI_DEEP).

الإضافات على النسخة السابقة:
  - مخرجات JSON منظمة (responseSchema) مع تراجع تلقائي إن لم يدعمها النموذج.
  - حد معدل للطلبات (LLM_RPM) حتى لا تُستنزف الحصة المجانية أثناء نقاش اللجنة.
"""
import os, re, json, time, logging, threading
from collections import deque

import requests

log = logging.getLogger("tdawll.llm")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FAST_MODELS = [m.strip() for m in os.environ.get("GEMINI_FAST", "gemini-flash-lite-latest,gemini-flash-latest").split(",") if m.strip()]
DEEP_MODELS = [m.strip() for m in os.environ.get("GEMINI_DEEP", "gemini-flash-latest,gemini-2.5-flash,gemini-flash-lite-latest").split(",") if m.strip()]
RPM = int(os.environ.get("LLM_RPM", "9"))            # أقصى طلبات في الدقيقة (الحصة المجانية محدودة)

_calls, _lk = deque(), threading.Lock()
stats = {"calls": 0, "fail": 0}


def _throttle():
    """ينتظر لو اقتربنا من حد الدقيقة، ولا ينتظر أكثر من 25 ثانية."""
    waited = 0.0
    while True:
        with _lk:
            now = time.time()
            while _calls and now - _calls[0] > 60:
                _calls.popleft()
            if len(_calls) < RPM:
                _calls.append(now)
                return
            wait = 60 - (now - _calls[0])
        if waited + wait > 25:
            return
        step = min(wait + 0.1, 5)
        time.sleep(step); waited += step


def extract_json(text):
    """يستخرج أول كائن JSON متوازن من نص النموذج (يتجاوز أسوار ```)."""
    if not text:
        return None
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
        elif ch == '"': instr = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(t[start:i + 1])
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
    return None


def generate(system, contents, deep=False, tokens=500, schema=None, temperature=0.3):
    """يرجع نص النموذج أو None. contents = قائمة رسائل بصيغة Gemini REST."""
    if not GEMINI_API_KEY:
        return None
    budget = 1024 if deep else 0
    base = {"temperature": temperature, "topP": 0.9}
    if schema:
        cfgs = [
            {**base, "maxOutputTokens": tokens + budget, "thinkingConfig": {"thinkingBudget": budget},
             "responseMimeType": "application/json", "responseSchema": schema},
            {**base, "maxOutputTokens": tokens + 600, "responseMimeType": "application/json", "responseSchema": schema},
            {**base, "maxOutputTokens": tokens + 600, "responseMimeType": "application/json"},   # بدون schema
        ]
    else:
        cfgs = [{**base, "maxOutputTokens": tokens + budget, "thinkingConfig": {"thinkingBudget": budget}},
                {**base, "maxOutputTokens": tokens + 600}]
    for mdl in (DEEP_MODELS if deep else FAST_MODELS):
        for ci, cfg in enumerate(cfgs):
            _throttle()
            stats["calls"] += 1
            try:
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent",
                                  headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                                  json={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                                        "generationConfig": cfg}, timeout=60)
                if r.status_code == 200:
                    cd = r.json().get("candidates", [])
                    parts = ((cd[0].get("content") or {}).get("parts", [])) if cd else []
                    txt = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                    if txt:
                        return txt
                    log.warning("gemini EMPTY %s cfg%d", mdl, ci)
                else:
                    log.warning("gemini HTTP%s %s cfg%d: %s", r.status_code, mdl, ci, r.text[:150])
                    if r.status_code in (429, 500, 503):
                        time.sleep(1.2)
                        break                                  # جرّب النموذج التالي
            except Exception as e:
                log.warning("gemini EXC %s: %s", mdl, str(e)[:100])
    stats["fail"] += 1
    return None


def user_msg(text):
    return [{"role": "user", "parts": [{"text": text}]}]


def generate_json(system, prompt, schema, deep=False, tokens=500, required=()):
    """يرجع dict بعد التحقق من الحقول المطلوبة، أو None."""
    txt = generate(system, user_msg(prompt), deep=deep, tokens=tokens, schema=schema)
    obj = extract_json(txt)
    if obj is None:
        return None
    if any(k not in obj for k in required):
        log.warning("json missing keys: %s", [k for k in required if k not in obj])
        return None
    return obj


def generate_text(system, prompt, deep=False, tokens=400):
    return generate(system, user_msg(prompt), deep=deep, tokens=tokens)
