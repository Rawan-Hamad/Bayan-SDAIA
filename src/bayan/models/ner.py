
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


