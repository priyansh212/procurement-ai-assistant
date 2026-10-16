"""
Lists the models your Groq API key can use right now.

Groq renames/retires models from time to time, so instead of guessing a
model name we ask Groq directly.

Run:  python -m src.rag.list_groq_models
Then copy one model id into GROQ_MODEL in your .env file.
"""

from openai import OpenAI

from src.config import settings


def main() -> None:
    key = settings.groq_api_key
    if not key or key.startswith("gsk_your"):
        print("GROQ_API_KEY is missing in .env")
        return

    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    models = sorted(m.id for m in client.models.list().data)

    # Chat models that are good for Q&A: skip speech/safety/tts models
    skip = ("whisper", "guard", "tts", "playai", "distil-whisper")
    chat_models = [m for m in models if not any(s in m.lower() for s in skip)]

    llama = [m for m in chat_models if "llama" in m.lower()]
    others = [m for m in chat_models if m not in llama]

    print("Llama models available to your key:")
    for m in llama:
        print("  ", m)
    print("\nOther chat models:")
    for m in others:
        print("  ", m)

    if llama:
        small = [m for m in llama if "8b" in m.lower() or "instant" in m.lower()]
        pick = (small or llama)[0]
        print(f"\nSuggested .env line:\n  GROQ_MODEL={pick}")


if __name__ == "__main__":
    main()
