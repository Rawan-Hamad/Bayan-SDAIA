# Model Card — Bayan two-stage search

## Intended use
استرجاع الحالات المشابهة وإعادة ترتيبها.

## Artefact / data versions
- Checkpoint / implementation: src/bayan/search/index.py + service.py; measured weights revision not recorded
- Preprocessing: 1.2.0 (current code; historical run version not recorded)
- Data snapshot: data/search/bayan_cases.csv (SHA256 13dc7df7eed8f1be3d741495f91a063999e6c5372ca3d987d661222843d77f62) (current local snapshot)

## Metrics
Recorded in BENCHMARKS.md; not rerun by this generator.

| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 0.0026 | 0.0026 | 16.86 ms |
| + cross-encoder rerank | 0.0179 | 0.0244 | 56.25 ms |
| cross-lingual slice | N/A | N/A | N/A |

- no-answer empty-correct: 20 / 20
- cross-lingual gap: N/A

## Slice metrics
لم يتم قياسه بعد.

## Behavioural tests
لم يتم قياسه بعد.

## Known limitations
نتائج البحث المسجلة أقل من أهداف اللاب. الفرق بين اللغات لم يتم قياسه بعد. لا تكفي نتيجة الاستعلامات بلا إجابة وحدها للحكم على جودة الاسترجاع. نسخة أوزان النماذج المستخدمة في القياس المسجل غير موثقة هنا.

## Evidence
[Benchmarks](../../BENCHMARKS.md) | [Evaluation](../../EVALUATION_REPORT.md)

## Contact / owner
Bayan project team.
