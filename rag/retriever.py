from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .chunker import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    """TF-IDF (unigram + bigram) retriever with a relevance threshold."""

    def __init__(self, chunks: list[Chunk], min_score: float = 0.05):
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = chunks
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Top-k chunks scoring at least `min_score`; empty list if nothing is relevant."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= self.min_score]
