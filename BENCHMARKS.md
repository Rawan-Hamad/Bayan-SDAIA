# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---|---|---|---|---|
| mBERT | 2.15 | 1.51 | 27 | 25 | 0.00 |
| XLM-R | 1.67 | 1.43 | 21 | 23 | 0.00 |
| CAMeLBERT | 1.41 | 2.70 | 20 | 38 | 0.00 |
| DistilBERT | 4.53 | 1.30 | 47 | 21 | 0.00 |
- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 2 Benchmark: Pad-Attention Leakage Results
| Metric / Condition | Value / Status |
|---|---|
| Pad Attention Mass (Without Mask) | `0.8093` |
| Pad Attention Mass (With Mask) | `0.0` |
| Pad Leak Removed | `True` |
| Numerical Equivalence | `True` |
| Causal Masking (Future Attention == 0) | `True` |

## Lab 3 — Models

| Model | Metric | Validation | Frozen Test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | - |
| Topic classifier (CAMeLBERT) | macro-F1 | 1.0000 | 1.0000 | 269.82 s |
| NER | entity-F1 | 1.0000 | 1.0000 | 208.46 s |
| QA | span/null smoke |  |  |  |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |


### Lab 4 — Arabic clitic segmentation

| Segmentation | LOCATION Recall | Delta | Test F1 | Test Accuracy |
|---|---:|---:|---:|---:|
| Baseline (no segmentation) | 1.0000 | — | 1.0000 | 1.0000 |
| D3Tok | 1.0000 | +0.0000 | 1.0000 | 1.0000 |

**Segmentation scheme:** CAMeL Tools D3Tok  
**LOCATION recall delta:** 1.0000 - 1.0000 = **0.0000 (no regression)**  
**D3Tok training runtime:** 6384.24 s (CPU)

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 0.0026 | 0.0026 | 16.86 ms |
| + cross-encoder rerank | 0.0179 | 0.0244 | 56.25 ms |
| cross-lingual slice | N/A | N/A | N/A |

- no-answer empty-correct: 20 / 20
- cross-lingual gap: N/A

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

## Lab 3A — TF-IDF Baseline

| Model | Metric | Validation | Frozen Test |
|---|---|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 |
