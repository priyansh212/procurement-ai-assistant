"""
RAG chain for the Procurement AI Assistant.

Flow for one question:
  1. RETRIEVE  - FAISS finds the top-k most relevant chunks
  2. AUGMENT   - chunks are numbered [1], [2]... and placed in the prompt
  3. GENERATE  - the LLM (Llama 3 / GPT) answers using only that context
  4. CITE      - we return the answer AND the source chunks behind it

Run a demo:  python -m src.rag.chain
"""

import re
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.ingestion.chunking import strip_context_header
from src.rag.llm import get_llm
from src.rag.prompts import RAG_PROMPT
from src.vectorstore.faiss_store import load_index, similarity_search

_index = None


def _get_index():
    """Load the FAISS index once and reuse it (loading is slow-ish)."""
    global _index
    if _index is None:
        _index = load_index()
    return _index


def format_context(docs: List[Document]) -> str:
    """Turn retrieved chunks into numbered blocks the LLM can cite."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        dtype = doc.metadata.get("doc_type", "unknown")
        blocks.append(f"[{i}] (source: {src}, type: {dtype})\n{strip_context_header(doc.page_content)}")
    return "\n\n".join(blocks)


def build_sources(docs: List[Document], answer: str) -> List[Dict]:
    """List the retrieved chunks and mark which [n] markers the answer actually used."""
    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [
        {
            "n": i,
            "source": d.metadata.get("source", "unknown"),
            "doc_type": d.metadata.get("doc_type", "unknown"),
            "snippet": strip_context_header(d.page_content)[:200].replace("\n", " "),
            "cited": i in cited_numbers,
        }
        for i, d in enumerate(docs, start=1)
    ]


def answer_question(question: str, k: int = None) -> Dict:
    """
    Answer a procurement question from the indexed documents.

    Returns:
        {"question": str, "answer": str, "sources": [ {n, source, doc_type, snippet} ]}
    """
    docs = similarity_search(_get_index(), question, k=k or settings.top_k_results)
    context = format_context(docs)

    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = build_sources(docs, answer)
    return {"question": question, "answer": answer.strip(), "sources": sources}


if __name__ == "__main__":
    # Demo: python -m src.rag.chain
    demo_questions = [
        "Who needs to approve a purchase of $20,000?",
        "What is the penalty if Acme delivers late?",
        "What is the capital of France?",  # not in docs -> should say not found
    ]
    for q in demo_questions:
        print("=" * 70)
        result = answer_question(q)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print("Sources (* = cited in the answer):")
        for s in result["sources"]:
            mark = "*" if s["cited"] else " "
            print(f" {mark}[{s['n']}] {s['source']} ({s['doc_type']})")
