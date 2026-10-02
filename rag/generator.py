import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM = (
    "You are a knowledge assistant. Answer ONLY from the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources in square brackets using the given source names, e.g. [ai_notes.txt]."
)


def _llm_answer(query: str, hits) -> str:
    import anthropic

    context = "\n\n".join(f"[{h.chunk.source}]\n{h.chunk.text}" for h in hits)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _extractive_answer(query: str, hits, n_sentences: int = 3) -> str:
    """Offline fallback: return the most query-relevant sentences from the retrieved chunks."""
    sentences = []
    for h in hits:
        clean = re.sub(r"#+\s*", "", h.chunk.text)
        for s in re.split(r"(?<=[.!?])\s+", clean):
            if len(s.split()) >= 4:
                sentences.append((s.strip(), h.chunk.source))
    if not sentences:
        return NO_ANSWER
    try:
        vec = TfidfVectorizer(stop_words="english").fit([s for s, _ in sentences] + [query])
        scores = cosine_similarity(vec.transform([query]), vec.transform([s for s, _ in sentences]))[0]
    except ValueError:
        return NO_ANSWER
    if scores.max() == 0:
        return NO_ANSWER
    best = sorted(i for i in scores.argsort()[::-1][:n_sentences] if scores[i] > 0)
    return " ".join(f"{sentences[i][0]} [{sentences[i][1]}]" for i in best)


def generate_answer(query: str, hits) -> tuple[str, str]:
    """Return (answer, mode) where mode is 'llm' or 'extractive'."""
    if not hits:
        return NO_ANSWER, "none"
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _llm_answer(query, hits), "llm"
        except Exception as e:  # network/auth/model errors -> degrade gracefully
            return _extractive_answer(query, hits) + f"\n(LLM unavailable: {type(e).__name__})", "extractive"
    return _extractive_answer(query, hits), "extractive"
