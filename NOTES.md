# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: Unicode forms
- Example: Inconsistent character representation (e.g., mixed compatibility variants or normalization issues in Arabic text).
- Why it matters: Causes tokenization mismatch and increases vocabulary fragmentation.
- Decision: Apply standard Unicode normalization (NFC) during preprocessing.

### Defect 2
- Class: Tatweel
- Example: Use of elongation character (ـ) inside words like "مـمتاز".
- Why it matters: Introduces artificial sub-word tokens and spelling variations that confuse the embedding space.
- Decision: Strip all Tatweel characters using regex replacement.

### Defect 3
- Class: Code-switching
- Example: Mixing Arabic and English within the same feedback sentence (e.g., "الخدمة bad جداً").
- Why it matters: Requires robust multilingual or mixed-script handling to prevent UNK tokens.
- Decision: Preserve mixed scripts but ensure the chosen tokenizer handles sub-words efficiently.

### Defect 4
- Class: PII
- Example: Inclusion of sensitive data like phone numbers, emails, or IDs in user feedback.
- Why it matters: Violates privacy policies and introduces unnecessary noisy tokens.
- Decision: Mask all PII entities using regex patterns during the preprocessing pipeline.

### Defect 5
- Class: Emojis
- Example: Presence of graphical emoticons (e.g., 😊, 🚀) in the text.
- Why it matters: Carries emotional signal but can create out-of-vocabulary artifacts if unhandled.
- Decision: Standardize or safely handle emojis depending on downstream task requirements.

### Defect 6
- Class: HTML remnants
- Example: Leftover web tags or entities (e.g., `<br>`, `&amp;`) scraped from interfaces.
- Why it matters: Adds structural noise that distorts linguistic tokenization.
- Decision: Clean and strip all HTML tags and entities using BeautifulSoup or regex parsing.


# Lab 1 - Step 4: Sentence Segmentation Notes
## Verification of Long Examples & Edge Cases:
- Standard Sentences & Punctuation: The sentencizer pipeline correctly identified sentence boundaries based on ending punctuation (., ?, !) after applying the standard preprocessing and PII masking contract.
- PII Masking Integration: Masking sensitive data (such as phone numbers to <PHONE>) prior to segmentation successfully prevented digits or special characters from causing premature or faulty sentence splits.
- Complex Text and Lists: Tested structural text blocks, handling them cleanly without breaking internal list markers or abbreviations improperly.
- Arabic Flow & Normalization: Unicode NFC normalization and Tatweel removal ensured consistent character mapping, yielding accurate and clean sentence tokens.


## Lab 2 - Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes |
|---|---|---|---|
| mBERT | 177,853,440 | 51.85% (92,208,384) | Larger vocab size (multilingual tax) |
| CamelBERT | 109,081,344 | 21.49% (23,436,288) | Targeted Arabic vocab size |

* **Why is the embedding share different?**  
The embedding share differs due to the significantly larger vocabulary size of the multilingual model (mBERT has ~119k tokens vs. CamelBERT's smaller Arabic-specific vocabulary), which directly increases the size of the token embedding matrix and incurs the 'multilingual tax'.
-Decoder-style causal attention

## Lab 2 - Step 5: Attention-map Diagnostics & Pad Leakage
* **Causal Mask Verification:** Confirmed future attention weights are strictly zero (`True`), ensuring proper autoregressive masking.
* **Pad-Attention Leakage Analysis:** 
  * Without a padding mask, a significant attention mass leaks into the `[PAD]` token (`0.8093`).
  * With the correct attention mask, the pad attention mass drops completely to `0.0`, confirming that the leakage is successfully eliminated (`Pad leak removed: True`).


## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
