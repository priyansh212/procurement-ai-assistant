"""
Central configuration for the Procurement AI Assistant.

All environment-driven settings live here so every module (ingestion,
embeddings, RAG chain, API, UI) reads from a single source of truth
instead of scattering os.getenv() calls everywhere.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    llm_provider: str = "groq"  # "groq" | "huggingface" | "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    huggingfacehub_api_token: str = ""
    hf_llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ---
    faiss_index_dir: str = "data/processed/faiss_index"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k_results: int = 4

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"


settings = Settings()
