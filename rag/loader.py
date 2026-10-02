from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install pypdf to read PDF files: pip install pypdf") from e
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def load_documents(folder: str | Path = "data") -> list[tuple[str, str]]:
    """Return [(filename, text)] for every supported file in `folder` (recursive)."""
    base = Path(folder)
    if not base.is_dir():
        raise FileNotFoundError(f"Knowledge-base folder not found: {base.resolve()}")
    docs = []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="ignore")
        text = text.strip()
        if text:
            docs.append((str(path.relative_to(base)), text))
    return docs
