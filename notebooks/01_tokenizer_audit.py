"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    words = 0
    pieces = 0

    for text in texts:
        text = str(text)

        words += len(text.split()) #"الخدمة ممتازة".split() >> ["الخدمة", "ممتازة"]
        pieces += len(tokenizer.tokenize(text))

    return pieces / max(words, 1) # because division by zero


def main():
    # TODO(Lab 1): load AR/EN slices, audit fertility and sequence lengths,
    # print a table, report p95 per language/tokenizer, and update BENCHMARKS.md.

    # 1) Load Bayan feedback corpus
    corpus = pd.read_csv(DATA)

    # 2) Split Arabic and English text
    ar_texts = corpus[corpus["lang"] == "ar"]["text"].dropna().tolist()
    en_texts = corpus[corpus["lang"] == "en"]["text"].dropna().tolist()

    print(f"Arabic rows:  {len(ar_texts)}")
    print(f"English rows: {len(en_texts)}")
    print()

    # 3) Audit every tokenizer
    for checkpoint, label in CANDIDATES.items():
        print("=" * 60)
        print(label)
        print(checkpoint)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint) #هنا Hugging Face ينزل tokenizer الخاص بالcheckpoint.#CAMeL-Lab/bert-base-arabic-camelbert-mix

        # Fertility
        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)

        # Sequence lengths
        #نحسب عدد التوكنز الناتجة لكل نص عربي من كل النصوص المدخلة، ونخزن هذه الأطوال في list.
        ar_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in ar_texts
        ]

        en_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in en_texts
        ]

        # p95
        #95% من النصوص العربية عندنا طولها 87 token أو أقل.
        # إن reference لـCAMeLBERT العربي حوالي 87 tokens، وبالتالي حد 512 كبير ومريح.
        ar_p95 = np.percentile(ar_lengths, 95)
        en_p95 = np.percentile(en_lengths, 95)

        print(f"AR fertility: {ar_fertility:.2f}")
        print(f"EN fertility: {en_fertility:.2f}")
        print(f"AR p95 length: {ar_p95:.0f} tokens")
        print(f"EN p95 length: {en_p95:.0f} tokens")
        print()



if __name__ == "__main__":
    main()
