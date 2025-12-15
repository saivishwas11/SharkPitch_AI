import os

# Groq/OpenAI model used for LLM calls
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # replace with your model id if needed
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
# Other helpful defaults (can be overridden via env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ASR defaults
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3")
GROQ_RATE_LIMIT_PER_MINUTE = int(os.getenv("GROQ_RATE_LIMIT_PER_MINUTE", "30"))
