import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path("data")
documents = []

for path in DOCS_DIR.glob("*.txt"):
    text = path.read_text(encoding="utf-8").strip()
    if text:
        documents.append((path.name, text))

if not documents:
    raise SystemExit("No documents found in data/. Add .txt files and run again.")

texts = [text for _, text in documents]
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

print("RAG AI Knowledge Assistant")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()
    if query.lower() == "exit":
        break
    if not query:
        continue

    q_vector = vectorizer.transform([query])
    scores = cosine_similarity(q_vector, matrix)[0]
    best = scores.argsort()[::-1][:2]

    print("\nRetrieved context:")
    for i in best:
        print(f"- {documents[i][0]} (relevance: {scores[i]:.2f})")
        print(documents[i][1][:500] + "\n")

    print("Answer:")
    print("Use the retrieved context above to formulate the answer. "
          "This prototype demonstrates the retrieval stage of a RAG pipeline.")
    print()
