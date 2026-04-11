def split_into_paragraphs(text, min_length=200):
    """
    Splits text into paragraph-like chunks.
    Ensures chunks are not too small.
    """
    raw_chunks = text.split(". ")
    paragraphs = []

    current = []

    for sentence in raw_chunks:
        current.append(sentence)

        if sum(len(s) for s in current) >= min_length:
            paragraphs.append(". ".join(current).strip() + ".")
            current = []

    if current:
        paragraphs.append(". ".join(current).strip() + ".")

    return paragraphs