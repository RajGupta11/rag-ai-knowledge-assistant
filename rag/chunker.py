import re
from dataclasses import dataclass


@dataclass
class Chunk:
    source: str
    index: int
    text: str


def split_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into word-based chunks (~chunk_size words) with overlap.

    Paragraph boundaries are respected where possible so a chunk reads naturally.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for para in paragraphs:
        words = para.split()
        if current and len(current) + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:]
        current.extend(words)
        while len(current) > chunk_size:  # very long paragraph
            chunks.append(" ".join(current[:chunk_size]))
            current = current[chunk_size - overlap:]
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(docs, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    out = []
    for source, text in docs:
        for i, piece in enumerate(split_text(text, chunk_size, overlap)):
            out.append(Chunk(source, i, piece))
    return out
