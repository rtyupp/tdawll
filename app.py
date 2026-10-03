def ai_general_reply(user_text, chat_id):
    # ✅ تم الاستبدال: OpenRouter بدلاً من Groq
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    
    if not OPENROUTER_API_KEY:
        return "⚠️ لم يتم ضبط مفتاح OpenRouter. أضفه في إعدادات Render باسم OPENROUTER_API_KEY."
    
    history = chat_memory.get(chat_id, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + \
               [{"role": "user", "content": user_text}]
               
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tdawll.onrender.com",  # مهم جداَ لتسجيل التطبيق
                "X-Title": "SPY Expert Trading Bot"             # يظهر في لوحة تحكم OpenRouter
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",  # ✅ النموذج الأقوى والأكثر استقراراً
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 700
            },
            timeout=25
        )
        
        data = response.json()
        
        if response.status_code != 200 or "choices" not in data:
            error_msg = data.get("error", {}).get("message", str(data))
            return f"❌ خطأ من OpenRouter: {error_msg[:100]}"
            
        reply = data["choices"][0]["message"]["content"]
        
        # حفظ الذاكرة كما كان يعمل سابقاَ
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        chat_memory[chat_id] = history[-20:]
        
        return reply
        
    except Exception as e:
        return f"❌ فشل الاتصال بالذكاء الاصطناعي: {str(e)[:80]}"
