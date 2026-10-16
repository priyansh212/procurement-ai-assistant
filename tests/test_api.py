"""Tests for the FastAPI backend. The agent is replaced by a fake, so no LLM key or index is needed."""
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.config import settings


@pytest.fixture
def client():
    # Not used as a context manager on purpose -> the startup (model loading) is skipped.
    return TestClient(api.app)


def _fake_result(question, session_id="default"):
    return {
        "question": question,
        "standalone_question": question,
        "answer": "The Director of Procurement approves it [1].",
        "sources": [{"n": 1, "source": "procurement_policy.txt", "doc_type": "policies", "snippet": "...", "cited": True}],
        "session_id": session_id,
    }


def test_health_reports_missing_index(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "faiss_index_dir", str(tmp_path / "nope"))
    body = client.get("/health").json()
    assert body["status"] == "index_missing"
    assert body["faiss_index_found"] is False


def test_health_ok_when_index_exists(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "faiss_index_dir", str(tmp_path))
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_query_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(api, "run_agent", lambda q, session_id="default", k=None: _fake_result(q, session_id))
    r = client.post("/query", json={"question": "Who approves $20,000?", "session_id": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert "Director of Procurement" in body["answer"]
    assert body["session_id"] == "abc"
    assert body["sources"][0]["cited"] is True


def test_query_passes_top_k_to_agent(client, monkeypatch):
    seen = {}

    def fake(q, session_id="default", k=None):
        seen["k"] = k
        return _fake_result(q, session_id)

    monkeypatch.setattr(api, "run_agent", fake)
    client.post("/query", json={"question": "Any question here", "top_k": 6})
    assert seen["k"] == 6


def test_query_rejects_too_short_question(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422


def test_query_rejects_bad_top_k(client):
    assert client.post("/query", json={"question": "Valid question", "top_k": 99}).status_code == 422


def test_missing_index_gives_503(client, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("No FAISS index found")

    monkeypatch.setattr(api, "run_agent", boom)
    assert client.post("/query", json={"question": "Valid question"}).status_code == 503


def test_unexpected_error_gives_500(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(api, "run_agent", boom)
    r = client.post("/query", json={"question": "Valid question"})
    assert r.status_code == 500
    assert "LLM exploded" in r.json()["detail"]
