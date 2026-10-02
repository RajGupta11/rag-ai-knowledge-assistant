# RAG AI Knowledge Assistant

A simple Retrieval-Augmented Generation (RAG) prototype built with Python.

## What problem does it solve?

The project demonstrates how an AI assistant can retrieve relevant information from a
small knowledge base before producing an answer. This is useful when an application
needs to work with domain-specific or private information.

## How it works

1. Text documents are loaded from `data/`.
2. TF-IDF converts the documents and user query into vectors.
3. Cosine similarity retrieves the most relevant documents.
4. The retrieved context is shown as the grounding context for an AI response.

This project focuses on the retrieval and grounding part of a RAG pipeline. An LLM can
be connected to the retrieved context as the generation layer.

## Tech Stack

- Python
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Retrieval-Augmented Generation (RAG) concepts
- Large Language Models (LLMs) as the optional generation layer

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Type a question such as:

`What is RAG?`

Type `exit` to quit.

## Project Structure

```text
rag-ai-knowledge-assistant/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── ai_notes.txt
    └── software_notes.txt
```
