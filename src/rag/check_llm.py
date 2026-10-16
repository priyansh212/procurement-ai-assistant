"""
Model checker: finds which Hugging Face chat models actually work for
YOUR account right now.

Why this exists: Hugging Face does not run big models itself. It forwards
requests to "inference providers" (Featherless, Together, Novita...), and
which models are available depends on the providers enabled on your
account. This script sends a tiny test message to a list of candidate
models and prints which ones answer.

Run:  python -m src.rag.check_llm
Then copy a working model name into HF_LLM_MODEL in your .env file.
"""

from huggingface_hub import InferenceClient

from src.config import settings

CANDIDATES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "HuggingFaceH4/zephyr-7b-beta",
]


def short_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    for marker in ("model_not_supported", "401", "402", "403", "404", "429"):
        if marker in text:
            return marker
    return text[:90]


def main() -> None:
    token = settings.huggingfacehub_api_token
    if not token or token.startswith("hf_your"):
        print("Token missing in .env (HUGGINGFACEHUB_API_TOKEN).")
        return

    client = InferenceClient(token=token)
    working = []

    print(f"Testing {len(CANDIDATES)} models (about 5-10 sec each)...\n")
    for model in CANDIDATES:
        try:
            out = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                max_tokens=8,
            )
            reply = out.choices[0].message.content.strip()
            print(f"  WORKS   {model}   (reply: {reply!r})")
            working.append(model)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            print(f"  FAILED  {model}   ({short_error(exc)})")

    print()
    if working:
        print("Put one of the WORKS models in .env, for example:")
        print(f"  HF_LLM_MODEL={working[0]}")
    else:
        print("No model worked. Check: token is Read type, and at least one "
              "provider is enabled at huggingface.co/settings/inference-providers")


if __name__ == "__main__":
    main()
