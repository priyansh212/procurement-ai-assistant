"""Small retrieval evaluation (12 questions) on the real sample documents in data/raw (uses the real embedding model).
Skipped automatically if sentence-transformers is not installed."""
import pytest

pytest.importorskip("sentence_transformers")

from src.config import settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import load_documents
from src.vectorstore.faiss_store import build_index

# (question, text that one of the top-3 chunks must contain)
EVAL_SET = [
    ("Who needs to approve a purchase of $20,000?", "Director"),
    ("What is the penalty if Acme delivers late?", "capped at 10%"),
    ("What are the payment terms in the Acme contract?", "Net 45"),
    ("When must we give notice to stop auto-renewal?", "60 days"),
    ("What happens in an emergency purchase?", "Emergency Purchase Order"),
    ("What is the maximum accepted defect rate per delivery lot?", "2% per delivery lot"),
    ("Which ISO certification must suppliers hold?", "ISO 9001"),
    ("What happens to invoices that have no PO number?", "without a PO number"),
    ("Within how many hours must a data breach be reported?", "72 hours"),
    ("How often are Tier 1 critical suppliers reassessed?", "every 12 months"),
    ("What on-time delivery does Globex guarantee?", "98% on-time"),
    ("Which courts have jurisdiction in the Globex agreement?", "Mumbai"),
]


@pytest.fixture(scope="module")
def index():
    chunks = chunk_documents(load_documents(settings.raw_data_dir))
    return build_index(chunks)


def test_retrieval_hit_rate(index):
    hits = 0
    for question, expected in EVAL_SET:
        top = index.similarity_search(question, k=3)
        found = any(expected in d.page_content for d in top)
        print(f"{'HIT ' if found else 'MISS'} | {question}")
        hits += found
    rate = hits / len(EVAL_SET)
    print(f"Retrieval hit rate (top-3): {rate:.0%}")
    assert rate >= 0.8
