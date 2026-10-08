# src/lib/chunking.py — il testo in passaggi di ~chunk_size caratteri, con sovrapposizione
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in [". ", ".\n", "? ", "! ", "\n\n"]:
                idx = text.rfind(sep, start, end)
                if idx > start + chunk_size // 2:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks
