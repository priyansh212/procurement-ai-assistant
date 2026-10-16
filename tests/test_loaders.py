"""Tests for document loading (src/ingestion/loaders.py). No network, no models."""
import pytest

from src.ingestion.loaders import load_documents, load_single_file


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "contracts").mkdir()
    (tmp_path / "policies").mkdir()
    (tmp_path / "contracts" / "deal.txt").write_text("Payment is Net 45.", encoding="utf-8")
    (tmp_path / "policies" / "policy.txt").write_text("Approvals above $5,000 need a Director.", encoding="utf-8")
    (tmp_path / "policies" / "notes.xyz").write_text("ignored", encoding="utf-8")
    return tmp_path


def test_loads_supported_files_and_skips_others(raw_dir):
    docs = load_documents(raw_dir)
    assert len(docs) == 2
    assert {d.metadata["source"] for d in docs} == {"deal.txt", "policy.txt"}


def test_doc_type_comes_from_folder_name(raw_dir):
    docs = load_documents(raw_dir)
    types = {d.metadata["source"]: d.metadata["doc_type"] for d in docs}
    assert types == {"deal.txt": "contracts", "policy.txt": "policies"}


def test_file_directly_in_root_is_unknown_type(tmp_path):
    f = tmp_path / "loose.txt"
    f.write_text("hello", encoding="utf-8")
    docs = load_single_file(f, tmp_path)
    assert docs[0].metadata["doc_type"] == "unknown"


def test_unsupported_single_file_raises(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        load_single_file(f, tmp_path)


def test_empty_folder_gives_no_documents(tmp_path):
    assert load_documents(tmp_path) == []
