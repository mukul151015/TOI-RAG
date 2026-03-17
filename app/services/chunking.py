import re


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_overlap_sentences: int = 2,
) -> list[str]:
    """Sentence-boundary-aware chunking with sentence-level overlap.

    Accumulates sentences until the chunk reaches ``chunk_size`` characters,
    then starts the next chunk by re-including the last
    ``chunk_overlap_sentences`` sentences so context is never split mid-thought.
    Falls back to the old character-window approach only when a single sentence
    exceeds ``chunk_size``.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    # Split on sentence-ending punctuation followed by whitespace or end-of-string.
    raw_sentences = re.split(r"(?<=[.!?])(?:\s+|\n\n)", cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence) + (1 if current else 0)  # +1 for joining space
        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N sentences as seed for the next chunk.
            overlap = current[-chunk_overlap_sentences:] if chunk_overlap_sentences > 0 else []
            current = overlap + [sentence]
            current_len = sum(len(s) + 1 for s in current) - 1
        else:
            current.append(sentence)
            current_len += s_len

        # Safety: if a single sentence itself exceeds chunk_size, hard-split it.
        if current_len > chunk_size * 1.5 and len(current) == 1:
            big = current[0]
            start = 0
            while start < len(big):
                end = min(len(big), start + chunk_size)
                chunks.append(big[start:end].strip())
                if end == len(big):
                    break
                start = max(0, end - chunk_overlap)
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c]
