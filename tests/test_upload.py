"""Tests for document upload (/upload, /documents) and the helpers in src/vectorstore/indexer.py.
Re-indexing is replaced by a fake, so no embedding model is needed."""
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.config import settings
from src.vectorstore import indexer


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "raw_data_dir", tmp_path)
    monkeypatch.setattr(api, "rebuild_index", lambda: {"documents": 3, "chunks": 7})
    return TestClient(api.app)


def _post(client, name="deal.txt", content=b"Payment is Net 45.", doc_type="contracts"):
    return client.post("/upload", files={"file": (name, content)}, data={"doc_type": doc_type})


def test_upload_saves_file_in_doc_type_folder(client, tmp_path):
    r = _post(client)
    assert r.status_code == 200
    body = r.json()
    assert body == {"filename": "deal.txt", "doc_type": "contracts", "documents_indexed": 3, "chunks_indexed": 7}
    assert (tmp_path / "contracts" / "deal.txt").read_bytes() == b"Payment is Net 45."


def test_unsupported_extension_is_rejected(client):
    r = _post(client, name="virus.exe")
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


def test_unknown_doc_type_is_rejected(client):
    assert _post(client, doc_type="secrets").status_code == 400


def test_empty_file_is_rejected(client):
    assert _post(client, content=b"").status_code == 400


def test_too_large_file_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    assert _post(client, content=b"x" * 50).status_code == 413


def test_path_traversal_name_stays_inside_data_folder(client, tmp_path):
    r = _post(client, name="../../evil.txt")
    assert r.status_code == 200
    assert (tmp_path / "contracts" / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()


def test_indexing_failure_gives_500(client, monkeypatch):
    def boom():
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(api, "rebuild_index", boom)
    r = _post(client)
    assert r.status_code == 500
    assert "indexing failed" in r.json()["detail"]


def test_documents_endpoint_lists_uploaded_files(client):
    _post(client, name="a.txt", doc_type="policies")
    _post(client, name="b.txt", doc_type="contracts")
    docs = client.get("/documents").json()["documents"]
    assert {(d["filename"], d["doc_type"]) for d in docs} == {("a.txt", "policies"), ("b.txt", "contracts")}


@pytest.mark.parametrize("raw,expected", [
    ("report final (1).pdf", "report final _1_.pdf"),
    ("C:\\Users\\me\\contract.docx", "contract.docx"),
    ("../x.txt", "x.txt"),
])
def test_safe_filename(raw, expected):
    assert indexer.safe_filename(raw) == expected


def test_safe_filename_rejects_empty_and_bad_types():
    with pytest.raises(ValueError):
        indexer.safe_filename("")
    with pytest.raises(ValueError):
        indexer.safe_filename("notes.csv")


def test_rebuild_index_saves_and_hot_swaps_into_chain(monkeypatch, tmp_path):
    import src.rag.chain as chain

    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "p.txt").write_text("Approvals above $5,000 need a Director.", encoding="utf-8")
    monkeypatch.setattr(settings, "raw_data_dir", tmp_path)

    fake_index = object()
    saved = {}
    monkeypatch.setattr(indexer, "build_index", lambda chunks: fake_index)
    monkeypatch.setattr(indexer, "save_index", lambda idx: saved.setdefault("idx", idx))
    monkeypatch.setattr(chain, "_index", None)

    stats = indexer.rebuild_index()
    assert stats == {"documents": 1, "chunks": 1}
    assert saved["idx"] is fake_index
    assert chain._index is fake_index


def test_rebuild_index_without_documents_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "raw_data_dir", tmp_path)
    with pytest.raises(ValueError):
        indexer.rebuild_index()
