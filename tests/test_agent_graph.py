"""Tests for the LangGraph agent: conversation memory and query rewriting.
The LLM and the FAISS index are replaced by fakes, so these run offline in about a second."""
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import src.agent.graph as graph

REWRITTEN = "What is the maximum cap on the Acme late-delivery penalty?"


def _fake_llm(prompt_value):
    text = prompt_value.to_string()
    if "Standalone question" in text:      # the rewrite prompt
        return REWRITTEN
    return "The cap is 10% [1]."           # the answer prompt


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    docs = [Document(page_content="Penalty is 1% per day, capped at 10%.",
                     metadata={"source": "acme.txt", "doc_type": "contracts"})]
    monkeypatch.setattr(graph, "get_llm", lambda: RunnableLambda(_fake_llm))
    monkeypatch.setattr(graph, "_get_index", lambda: object())
    monkeypatch.setattr(graph, "similarity_search", lambda index, q, k=None: docs)
    monkeypatch.setattr(graph, "_graph", None)  # fresh graph + fresh memory for every test


def test_first_question_is_not_rewritten():
    r = graph.run_agent("What is the penalty if Acme delivers late?", session_id="s1")
    assert r["standalone_question"] == "What is the penalty if Acme delivers late?"
    assert "10%" in r["answer"]
    assert r["sources"][0]["cited"] is True


def test_follow_up_is_rewritten_using_memory():
    graph.run_agent("What is the penalty if Acme delivers late?", session_id="s1")
    r = graph.run_agent("And what is the cap on that?", session_id="s1")
    assert r["standalone_question"] == REWRITTEN


def test_other_session_has_no_memory():
    graph.run_agent("What is the penalty if Acme delivers late?", session_id="s1")
    r = graph.run_agent("And what is the cap on that?", session_id="s2")
    assert r["standalone_question"] == "And what is the cap on that?"
