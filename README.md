# RAG AI Knowledge Assistant

A working Retrieval-Augmented Generation (RAG) assistant in Python. It loads your documents, splits them into chunks, retrieves the most relevant passages for a question, and generates a **grounded answer with source citations**. If the answer is not in your documents, it says so instead of guessing.

## Features
- **Loads** `.txt`, `.md` and `.pdf` files from `data/` (recursively)
- **Chunking** with overlap, so retrieval returns passages, not whole files
- **TF-IDF retrieval** (unigrams + bigrams) with a relevance threshold
- **LLM generation** via the Anthropic API when `ANTHROPIC_API_KEY` is set
- **Offline fallback**: with no key, it returns the most relevant sentences, with citations
- **"I don't know"** when nothing relevant is retrieved
- CLI (`app.py`), web UI (`streamlit_app.py`) and unit tests

## Pipeline
```
documents -> loader -> chunker -> TF-IDF index
question  -> retriever (top-k, threshold) -> generator (LLM or extractive) -> answer + sources
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your ANTHROPIC_API_KEY in .env (optional)
```

## Run
```bash
python app.py                    # command line
streamlit run streamlit_app.py   # web UI
pytest                           # tests
```
Add your own files to `data/` and restart.

## Project structure
```
app.py              CLI entry point
streamlit_app.py    Web UI
rag/
  loader.py         read txt/md/pdf
  chunker.py        overlapping word chunks
  retriever.py      TF-IDF search + score threshold
  generator.py      Anthropic LLM answer / extractive fallback
data/               knowledge base
tests/test_rag.py   unit tests
```

## Roadmap
- Swap TF-IDF for embeddings (sentence-transformers + FAISS/Chroma) or hybrid search
- Conversation memory and streaming answers
- Evaluation set to measure retrieval quality
