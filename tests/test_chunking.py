"""Tests for chunking (src/ingestion/chunking.py)."""
from langchain_core.documents import Document

from src.ingestion.chunking import chunk_documents, strip_context_header


def _doc(text, **meta):
    return Document(page_content=text, metadata={"source": "a.txt", "doc_type": "policies", **meta})


def test_chunks_respect_size_limit():
    text = ("Clause about payment terms and delivery. " * 100).strip()
    chunks = chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(strip_context_header(c.page_content)) <= 200 for c in chunks)


def test_chunk_ids_are_sequential():
    text = ("Sentence number one. " * 80).strip()
    chunks = chunk_documents([_doc(text)], chunk_size=150, chunk_overlap=10)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_metadata_is_kept_on_every_chunk():
    text = ("Some contract text. " * 60).strip()
    chunks = chunk_documents([_doc(text, page=3)], chunk_size=150, chunk_overlap=10)
    assert all(c.metadata["source"] == "a.txt" for c in chunks)
    assert all(c.metadata["doc_type"] == "policies" for c in chunks)
    assert all(c.metadata["page"] == 3 for c in chunks)


def test_short_document_stays_one_chunk():
    chunks = chunk_documents([_doc("Short text.")], chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert strip_context_header(chunks[0].page_content) == "Short text."


def test_every_chunk_starts_with_document_header():
    text = ("Article text about delivery. " * 60).strip()
    chunks = chunk_documents([_doc(text)], chunk_size=150, chunk_overlap=10)
    assert all(c.page_content.startswith("[Document: a.txt | Type: policies]\n") for c in chunks)


def test_header_can_be_switched_off():
    chunks = chunk_documents([_doc("Short text.")], add_header=False)
    assert chunks[0].page_content == "Short text."


def test_strip_header_is_safe_without_header():
    assert strip_context_header("plain text") == "plain text"
    assert strip_context_header("[Document: x.txt | Type: t]\nbody") == "body"
