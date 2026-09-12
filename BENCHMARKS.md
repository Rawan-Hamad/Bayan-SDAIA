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
| fp32 torch @512 padded | 736.10 ms | 1343.33 ms | Not measured yet | 416.20 MiB |
| fp32 torch @128 dynamic | 42.07 ms | 70.92 ms | Not measured yet | 416.20 MiB |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

## Lab 3A — TF-IDF Baseline

| Model | Metric | Validation | Frozen Test |
|---|---|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 |

### Lab 6 - Step 3: Behavioural suite

Model: local CAMeLBERT topic classifier (CPU, shared preprocessing).
Evidence: [case results](data/eval/behavioural_results.json).

| Suite | Cases | Unique cases | Passed | Failed | Skipped | Pass rate | Failure rate | Target |
|---|---:|---:|---:|---:|---:|---|---|---|
| invariance | 200 | 10 | 200 | 0 | 0 | 100.0% | 0.0% | 95% met |
| directional | 200 | 10 | 0 | 0 | 200 | N/A | N/A | N/A |
| mft | 16 | 16 | 14 | 2 | 0 | 87.5% | 12.5% | 90% NOT met |

Invariance covers whitespace changes only (10 unique pairs repeated across 200 cases).
MFT fell below 90%: two English cases predicted billing instead of licensing, and parks instead of roads.

Re-run: `python -m bayan.evaluation.behavioural --topic-model artifacts/topic_classifier --output data/eval/behavioural_results.json`

### Lab 6 - Step 4: Error analysis

Assistant draft: 120 fixture errors; park paths 55, irrigation 33, playgrounds 32.
[Histogram, case evidence and three proposed fixes](docs/ERROR_REVIEW_ANALYSIS.md).
Metric deltas are conditional planning scenarios, not measured improvements.

### Lab 7 - Step 1: CPU baseline

CPU fp32, batch=1, threads=4, warm-up=20 per rung; all 2000 supplied texts.
Model forward time only; preprocessing, tokenization and HTTP excluded.
At batch=1, dynamic padding uses each text's actual token length.
Dynamic/128 speed-up: p50 17.50x; p99 18.94x.
Texts truncated at 128: 0. Quality change: not measured yet.
Single sequential run; CPU contention and run order can affect timing.
[Raw timings and environment](data/serving/baseline_benchmark.json).

Re-run: `python scripts/benchmark_inference.py --threads 4 --warmup 20`
