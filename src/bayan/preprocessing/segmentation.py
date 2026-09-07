"""Lab 1 starter: sentence segmentation."""

import spacy
from bayan.preprocessing.core import preprocess

def build_pipeline():
    # TODO(Lab 1): build the spaCy segmentation pipeline.
    # Blank multilingual spaCy pipeline.
    nlp = spacy.blank("xx")

    # Add lightweight sentence segmentation component.
    nlp.add_pipe("sentencizer")  # set end & start of sentence by . ? !

    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    # TODO(Lab 1): preprocess then return non-empty sentence strings.
    # Important: use the same preprocessing contract first.
    text = preprocess(raw)

    doc = nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]
