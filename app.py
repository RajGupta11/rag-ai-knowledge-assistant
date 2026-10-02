"""Command-line RAG knowledge assistant.  Usage: python app.py [--data data] [--top-k 3]"""
import argparse
from dotenv import load_dotenv
from rag import load_documents, chunk_documents, Retriever, generate_answer


def build_retriever(data_dir: str, chunk_size: int = 120, overlap: int = 30) -> Retriever:
    docs = load_documents(data_dir)
    if not docs:
        raise SystemExit(f"No .txt/.md/.pdf documents found in {data_dir}/. Add some and retry.")
    return Retriever(chunk_documents(docs, chunk_size, overlap))


def main():
    load_dotenv()
    p = argparse.ArgumentParser(description="RAG AI Knowledge Assistant")
    p.add_argument("--data", default="data", help="folder with .txt/.md/.pdf files")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    retriever = build_retriever(args.data)
    print(f"RAG AI Knowledge Assistant - {len(retriever.chunks)} chunks indexed. Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        hits = retriever.search(query, args.top_k)
        answer, mode = generate_answer(query, hits)
        print(f"\nAnswer ({mode}):\n{answer}\n")
        if hits:
            print("Sources:")
            for h in hits:
                print(f"  - {h.chunk.source} #{h.chunk.index} (score {h.score:.2f})")
        print()


if __name__ == "__main__":
    main()
