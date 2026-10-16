"""
Embeddings for the Procurement AI Assistant.

We use a local Hugging Face sentence-transformers model
(all-MiniLM-L6-v2 by default) rather than an API-based embedding model.
This matters for this project specifically because:

  1. You're using local/HF Llama 3 as the LLM (no OpenAI key) --
     using local embeddings too means the whole pipeline can run
     with zero paid API calls.
  2. Procurement documents are often confidential -- keeping embedding
     generation local avoids sending contract text to a third-party
     embeddings API.

The model runs on CPU by default and is small enough (~80MB) to load
quickly without a GPU.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    Cached (lru_cache) because loading the sentence-transformers model
    from disk/HF hub is the expensive part -- we want to do it once per
    process, not once per request in the FastAPI app.
    """
    print(f"[embeddings] Loading embedding model: {settings.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine-sim friendly
    )


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.embeddings.embedder
    model = get_embedding_model()
    sample_texts = [
        "Purchases over $50,000 require a competitive bidding process.",
        "The supplier warranty covers manufacturing defects for 12 months.",
    ]
    vectors = model.embed_documents(sample_texts)
    print(f"[embeddings] Generated {len(vectors)} vectors, "
          f"dimension={len(vectors[0])}")
