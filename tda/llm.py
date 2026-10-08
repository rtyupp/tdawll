# -*- coding: utf-8 -*-
"""
طبقة LLM متعددة المزودين للبوت.
المحرك الوحيد: Groq المجاني. لا يوجد انتقال إلى Gemini أو OpenRouter.
كل مزود اختياري؛ لا يتم تسجيل المفاتيح أو كشفها في الردود.

الإضافات على النسخة السابقة:
  - مخرجات JSON منظمة (responseSchema) مع تراجع تلقائي إن لم يدعمها النموذج.
  - حد معدل للطلبات (LLM_RPM) حتى لا تُستنزف الحصة المجانية أثناء نقاش اللجنة.
"""
import os, re, json, time, logging, threading
from collections import deque

import requests

log = logging.getLogger("tdawll.llm")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PROVIDER_ORDER = ["groq"]
GROQ_FAST = [m.strip() for m in os.environ.get("GROQ_FAST_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-120b").split(",") if m.strip()]
GROQ_DEEP = [m.strip() for m in os.environ.get("GROQ_DEEP_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-120b").split(",") if m.strip()]
OPENROUTER_MODELS = [m.strip() for m in os.environ.get("OPENROUTER_MODELS", "").split(",") if m.strip()]
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


def _openai_messages(system, contents):
    msgs = [{"role": "system", "content": system}]
    for m in contents:
        role = "assistant" if m.get("role") == "model" else m.get("role", "user")
        text = "".join(p.get("text", "") for p in m.get("parts", []) if isinstance(p, dict))
        if text:
            msgs.append({"role": role, "content": text})
    return msgs

def _compatible_generate(provider, key, models, system, contents, deep, tokens, schema, temperature):
    if not key or not models:
        return None
    messages = _openai_messages(system, contents)
    for mdl in models:
        _throttle(); stats["calls"] += 1
        body = {"model": mdl, "messages": messages, "temperature": temperature,
                "max_completion_tokens": max(64, tokens + (700 if deep else 350)),
                "stream": False}
        if provider == "groq":
            body.update({"reasoning_effort": "medium" if deep else "none", "include_reasoning": False})
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                       "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://spy-bot-live.onrender.com"),
                       "X-OpenRouter-Title": "SPY 5m Arabic Analyst"}
        if schema:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(url, headers=headers, json=body, timeout=20)
            if r.status_code == 200:
                data = r.json(); choices = data.get("choices", [])
                txt = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
                if txt: return txt
                log.warning("%s EMPTY %s", provider, mdl)
            else:
                log.warning("%s HTTP%s %s: %s", provider, r.status_code, mdl, r.text[:160])
                if r.status_code == 429:
                    time.sleep(1.0)
        except Exception as e:
            log.warning("%s EXC %s: %s", provider, mdl, str(e)[:120])
    return None

def generate(system, contents, deep=False, tokens=500, schema=None, temperature=0.3):
    """يرجع نص النموذج أو None. contents = قائمة رسائل بصيغة Gemini REST."""
    # OpenAI-compatible providers are tried first when their keys are configured.
    for provider in PROVIDER_ORDER:
        if provider == "groq":
            txt = _compatible_generate("groq", GROQ_API_KEY, GROQ_DEEP if deep else GROQ_FAST,
                                       system, contents, deep, tokens, schema, temperature)
        elif provider == "openrouter":
            txt = _compatible_generate("openrouter", OPENROUTER_API_KEY, OPENROUTER_MODELS,
                                       system, contents, deep, tokens, schema, temperature)
        else:
            continue
        if txt:
            return txt
    # Groq-only by design: do not wait for or call another provider.
    stats["fail"] += 1
    return None
    budget = 1024 if deep else 0
    base = {"temperature": temperature, "topP": 0.9}
    if schema:
        cfgs = []
        if deep:
            cfgs.append({**base, "maxOutputTokens": tokens + budget,
                         "thinkingConfig": {"thinkingBudget": budget},
                         "responseMimeType": "application/json", "responseSchema": schema})
        cfgs += [{**base, "maxOutputTokens": tokens + 600,
                  "responseMimeType": "application/json", "responseSchema": schema},
                 {**base, "maxOutputTokens": tokens + 600,
                  "responseMimeType": "application/json"}]
    else:
        cfgs = ([{**base, "maxOutputTokens": tokens + budget,
                  "thinkingConfig": {"thinkingBudget": budget}}] if deep else [])
        cfgs.append({**base, "maxOutputTokens": tokens + 600})
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


def provider_status():
    configured = []
    if GROQ_API_KEY: configured.append("groq")
    return {"order": PROVIDER_ORDER, "configured": configured, "calls": stats["calls"], "failures": stats["fail"]}

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
