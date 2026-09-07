# Decision Records

## tokenizer
- Chosen checkpoint(s): `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- Arabic fertility evidence: 1.41 (lowest fragmentation for Arabic text)
- English fertility evidence: 2.70
- p95 length evidence: AR p95 = 20 tokens, EN p95 = 38 tokens (well within the 512 context limit)
- Operational trade-off / rationale: CAMeLBERT is chosen because Bayan's core use case prioritizes Arabic feedback processing; its superior Arabic fertility (1.41) outweighs the higher English fertility trade-off, ensuring lower computational overhead and cleaner tokenization.

## arabic-model

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
