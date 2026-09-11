# Model Card — Bayan NER

## Intended use
استخراج الكيانات من النص.

## Artefact / data versions
- Checkpoint / implementation: scripts/train_ner.py; local trained artefact not supplied
- Preprocessing: 1.2.0 (current code; historical run version not recorded)
- Data snapshot: data/models/bayan_ner.conll (SHA256 4d0adae95c44d92ea81c08ffdb2763772f3bd6602a4d94485e9fa4acab070784) (current local snapshot)

## Metrics
Recorded in BENCHMARKS.md; not rerun by this generator.
| Model | Metric | Validation | Frozen Test | Train time |
|---|---|---:|---:|---|
| NER | entity-F1 | 1.0000 | 1.0000 | 208.46 s |

## Slice metrics
لم يتم قياسه بعد.

## Behavioural tests
لم يتم قياسه بعد.

## Known limitations
نتيجة NER من البنش مارك المسجّل؛ لم يُعَد تشغيل النموذج محليًا في هذه الخطوة. تقييم الشرائح والاختبارات السلوكية لم يتم قياسه بعد. الأداء على نصوص جديدة يحتاج تحققًا مستقلًا.

## Evidence
[Benchmarks](../../BENCHMARKS.md) | [Evaluation](../../EVALUATION_REPORT.md)

## Contact / owner
Bayan project team.
