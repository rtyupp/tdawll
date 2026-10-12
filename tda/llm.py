# -*- coding: utf-8 -*-
"""
طبقة LLM للبوت.
المحرك الوحيد: DeepSeek (واجهة متوافقة مع OpenAI)، ولا يوجد مزود احتياطي.
المفتاح اختياري؛ ولا يُسجَّل أو يُكشف في الردود.

التفاصيل:
  - مخرجات JSON منظمة (response_format=json_object) مع استخراج متوازن للنص.
  - نموذج واحد للنمط السريع و«العميق» (DEEPSEEK_MODEL)، ويمكن ضبط الطلب عبر temperature.
  - حد معدل للطلبات (LLM_RPM) حتى لا يُستنزف الرصيد أثناء نقاش اللجنة.
"""
import os, re, json, time, logging, threading
from collections import deque

import requests

log = logging.getLogger("tdawll.llm")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
PROVIDER_ORDER = ["deepseek"]
RPM = int(os.environ.get("LLM_RPM", "30"))           # أقصى طلبات في الدقيقة

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


def _openai_messages(system, contents, json_mode=False):
    msgs = [{"role": "system", "content": system}]
    for m in contents:
        role = "assistant" if m.get("role") == "model" else m.get("role", "user")
        text = "".join(p.get("text", "") for p in m.get("parts", []) if isinstance(p, dict))
        if text:
            msgs.append({"role": role, "content": text})
    if json_mode:
        # DeepSeek يشترط ذكر كلمة json في الرسائل عند response_format=json_object.
        msgs[-1]["content"] += "\n\nRespond with a single valid JSON object only."
    return msgs


def _compatible_generate(system, contents, deep, tokens, schema, temperature):
    if not DEEPSEEK_API_KEY:
        return None
    messages = _openai_messages(system, contents, json_mode=bool(schema))
    _throttle(); stats["calls"] += 1
    body = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max(64, tokens + (700 if deep else 350)),
            "stream": False}
    if schema:
        body["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(f"{DEEPSEEK_BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                          json=body, timeout=30)
        if r.status_code == 200:
            choices = r.json().get("choices", [])
            txt = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
            if txt:
                return txt
            log.warning("deepseek EMPTY %s", DEEPSEEK_MODEL)
        else:
            log.warning("deepseek HTTP%s %s: %s", r.status_code, DEEPSEEK_MODEL, r.text[:160])
            if r.status_code == 429:
                time.sleep(1.0)
    except Exception as e:
        log.warning("deepseek EXC %s: %s", DEEPSEEK_MODEL, str(e)[:120])
    return None

def generate(system, contents, deep=False, tokens=500, schema=None, temperature=0.3):
    """يرجع نص DeepSeek أو None؛ لا يوجد مزود احتياطي."""
    txt = _compatible_generate(system, contents, deep, tokens, schema, temperature)
    if txt:
        return txt
    stats["fail"] += 1
    return None

def provider_status():
    configured = []
    if DEEPSEEK_API_KEY: configured.append("deepseek")
    return {"order": PROVIDER_ORDER, "configured": configured, "model": DEEPSEEK_MODEL,
            "calls": stats["calls"], "failures": stats["fail"]}

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
