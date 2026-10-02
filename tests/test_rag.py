import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rag import chunk_documents, Retriever, generate_answer
from rag.chunker import split_text

DOCS = [
    ("ai.txt", "RAG combines retrieval with a language model. It grounds answers in your documents."),
    ("git.txt", "Git tracks source code changes. Branches and pull requests support collaboration."),
]


def test_chunk_overlap_and_size():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert len(chunks) >= 3


def test_retrieval_picks_right_doc():
    r = Retriever(chunk_documents(DOCS))
    assert r.search("What is RAG retrieval?")[0].chunk.source == "ai.txt"
    assert r.search("git branches")[0].chunk.source == "git.txt"


def test_irrelevant_query_returns_nothing():
    r = Retriever(chunk_documents(DOCS))
    assert r.search("quantum chromodynamics banana") == []


def test_no_hits_means_dont_know(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer, mode = generate_answer("anything", [])
    assert "don't know" in answer and mode == "none"


def test_extractive_answer_cites_source(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = Retriever(chunk_documents(DOCS))
    answer, mode = generate_answer("what do branches do in git", r.search("what do branches do in git"))
    assert mode == "extractive" and "[git.txt]" in answer
