"""
Document loaders for the Procurement AI Assistant.

Loads contracts, supplier manuals, and policy documents (PDF / DOCX / TXT)
into LangChain `Document` objects, tagging each with metadata that we
later use for citation-backed answers (source filename, doc_type, page).

Design note: we classify `doc_type` (contract / policy / supplier_manual)
from the folder the file lives in, e.g.:
    data/raw/contracts/*.pdf
    data/raw/policies/*.pdf
    data/raw/supplier_manuals/*.docx
This keeps ingestion simple while still giving the LLM useful grounding
context ("this comes from a contract vs. a policy doc").
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _infer_doc_type(file_path: Path, raw_root: Path) -> str:
    """Infer doc_type from the immediate parent folder name, if present."""
    try:
        relative = file_path.relative_to(raw_root)
    except ValueError:
        return "unknown"
    if len(relative.parts) > 1:
        return relative.parts[0]  # e.g. "contracts", "policies"
    return "unknown"


def load_single_file(file_path: Path, raw_root: Path) -> List[Document]:
    """Load one file and attach procurement-relevant metadata."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    doc_type = _infer_doc_type(file_path, raw_root)

    for doc in docs:
        doc.metadata.update(
            {
                "source": file_path.name,
                "source_path": str(file_path),
                "doc_type": doc_type,
            }
        )
    return docs


def load_documents(raw_dir: Path) -> List[Document]:
    """
    Recursively load every supported document under `raw_dir`.

    Returns a flat list of LangChain Documents ready for chunking.
    Skips unsupported file types instead of failing the whole run.
    """
    raw_dir = Path(raw_dir)
    all_docs: List[Document] = []
    skipped: List[str] = []

    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(file_path.name)
            continue
        all_docs.extend(load_single_file(file_path, raw_dir))

    if skipped:
        print(f"[loaders] Skipped {len(skipped)} unsupported file(s): {skipped}")

    print(f"[loaders] Loaded {len(all_docs)} document page(s)/section(s) "
          f"from {raw_dir}")
    return all_docs


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.ingestion.loaders
    from src.config import settings

    documents = load_documents(settings.raw_data_dir)
    for d in documents[:3]:
        print("---")
        print(d.metadata)
        print(d.page_content[:200])
