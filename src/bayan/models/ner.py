
"""Lab 3 starter: NER label alignment."""


def align_labels(word_ids, word_labels):
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)

        elif word_id != previous_word_id:
            aligned_labels.append(word_labels[word_id])

        else:
            aligned_labels.append(-100)

        previous_word_id = word_id

    return aligned_labels


def prepare_ner_tokens(words, segmentation="none"):
    """Prepare NER tokens while preserving each original token's index."""

    if segmentation == "none":
        return list(words), list(range(len(words)))

    if segmentation == "d3tok":
        from bayan.preprocessing.arabic import segment

        pieces = []
        origins = []

        for word_id, word in enumerate(words):
            word_pieces = segment(word)

            for piece in word_pieces:
                pieces.append(piece)
                origins.append(word_id)

        return pieces, origins

    raise ValueError(f"Unknown segmentation mode: {segmentation}")