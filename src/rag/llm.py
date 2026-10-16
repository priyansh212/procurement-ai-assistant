"""
LLM factory for the Procurement AI Assistant.

One function, get_llm(), returns a LangChain chat model based on
LLM_PROVIDER in .env:

  - "huggingface": Llama 3 hosted on Hugging Face (needs HF token).
                   Runs on HF's servers, so no GPU is needed on your laptop.
  - "openai":      GPT-4 family (needs OPENAI_API_KEY), optional.

Everything else in the project calls get_llm() and never cares which
provider is behind it. Swapping models = changing one line in .env.
"""

from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def get_llm():
    provider = settings.llm_provider.lower()

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        token = settings.huggingfacehub_api_token
        if not token or token.startswith("hf_your"):
            raise RuntimeError(
                "HUGGINGFACEHUB_API_TOKEN is missing in your .env file. "
                "Create a token at huggingface.co/settings/tokens and paste it."
            )

        print(f"[llm] Using Hugging Face model: {settings.hf_llm_model}")
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_llm_model,
            task="text-generation",
            huggingfacehub_api_token=token,
            max_new_tokens=512,
            temperature=0.1,  # low = more factual, less creative
            do_sample=True,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "groq":
        # Groq serves open Llama models for free (own API key, no credit card).
        # It exposes an OpenAI-compatible API, so we reuse ChatOpenAI.
        from langchain_openai import ChatOpenAI

        if not settings.groq_api_key or settings.groq_api_key.startswith("gsk_your"):
            raise RuntimeError(
                "GROQ_API_KEY is missing in your .env file. "
                "Create one at console.groq.com -> API Keys."
            )

        print(f"[llm] Using Groq model: {settings.groq_model}")
        return ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
            # Reasoning models (gpt-oss, qwen3) spend tokens "thinking" before the
            # answer, so leave plenty of room or the answer can come back empty.
            max_tokens=2048,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
            raise RuntimeError("OPENAI_API_KEY is missing in your .env file.")

        print(f"[llm] Using OpenAI model: {settings.openai_model}")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
