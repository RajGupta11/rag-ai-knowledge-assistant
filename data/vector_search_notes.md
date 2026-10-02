# Embeddings and Vector Search

Embeddings turn text into numeric vectors so that similar meanings end up close together. Vector databases such as FAISS and Chroma store these vectors and find the nearest ones quickly.

Keyword methods like TF-IDF match exact words, while embeddings can match different wording that carries the same meaning. Many production RAG systems combine both approaches, which is called hybrid search.

Chunking splits long documents into smaller passages so that retrieval returns only the relevant part instead of an entire file.
