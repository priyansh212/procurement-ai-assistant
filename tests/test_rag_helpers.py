"""Tests for the RAG helper functions and the anti-hallucination prompt."""
from langchain_core.documents import Document

from src.rag.chain import build_sources, format_context
from src.rag.prompts import RAG_PROMPT


def _docs():
    return [
        Document(page_content="Net 45 from invoice receipt.", metadata={"source": "acme.txt", "doc_type": "contracts"}),
        Document(page_content="Director approves $5,000-$50,000.", metadata={"source": "policy.txt", "doc_type": "policies"}),
    ]


def test_format_context_numbers_blocks_from_one():
    ctx = format_context(_docs())
    assert "[1] (source: acme.txt, type: contracts)" in ctx
    assert "[2] (source: policy.txt, type: policies)" in ctx
    assert "Net 45 from invoice receipt." in ctx


def test_build_sources_marks_only_cited_chunks():
    sources = build_sources(_docs(), "Payment terms are Net 45 [1].")
    assert [s["cited"] for s in sources] == [True, False]
    assert sources[0]["source"] == "acme.txt"
    assert sources[1]["doc_type"] == "policies"


def test_build_sources_no_citations():
    sources = build_sources(_docs(), "I could not find this in the provided documents.")
    assert not any(s["cited"] for s in sources)


def test_snippet_is_short_and_single_line():
    long_doc = Document(page_content="line one\n" + "x" * 500, metadata={"source": "s", "doc_type": "d"})
    snippet = build_sources([long_doc], "")[0]["snippet"]
    assert len(snippet) <= 200
    assert "\n" not in snippet


def test_prompt_contains_refusal_rule_and_context():
    messages = RAG_PROMPT.format_messages(context="[1] some text", question="Who approves?")
    system_text = messages[0].content
    assert "I could not find this in the provided documents." in system_text
    assert "[1] some text" in system_text
    assert messages[1].content == "Who approves?"


def test_header_is_hidden_in_prompt_context_and_snippet():
    doc = Document(
        page_content="[Document: globex.docx | Type: contracts]\nThe Supplier guarantees 98% on-time delivery.",
        metadata={"source": "globex.docx", "doc_type": "contracts"},
    )
    ctx = format_context([doc])
    assert "[Document:" not in ctx
    assert "98% on-time" in ctx
    assert build_sources([doc], "")[0]["snippet"].startswith("The Supplier guarantees")
