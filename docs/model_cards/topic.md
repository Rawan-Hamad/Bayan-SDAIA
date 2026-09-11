# Model Card — CAMeLBERT topic classifier

## Intended use
تصنيف موضوع ملاحظات المستفيدين.

## Artefact / data versions
- Checkpoint / implementation: artifacts/topic_classifier; CAMeL-Lab/bert-base-arabic-camelbert-mix
- Preprocessing: 1.2.0 (current code; historical run version not recorded)
- Data snapshot: data/raw/bayan_feedback.csv (SHA256 6ecc9edc4a30440fe5d04ea45d04346715b97d6f3600faea06980d947fc0ae15) (current local snapshot)

## Metrics
Recorded in BENCHMARKS.md; not rerun by this generator.
| Model | Metric | Validation | Frozen Test | Train time |
|---|---|---:|---:|---|
| Topic classifier (CAMeLBERT) | macro-F1 | 1.0000 | 1.0000 | 269.82 s |

## Slice metrics
لم يتم قياسه بعد.

## Behavioural tests
| Suite | Passed | Failed | Skipped | Pass rate |
|---|---:|---:|---:|---|
| invariance | 200 | 0 | 0 | 100.0% |
| directional | 0 | 0 | 200 | لم يتم قياسه بعد. |
| mft | 14 | 2 | 0 | 87.5% |

## Known limitations
اختبار الثبات يغطي المسافات فقط. نتيجة MFT أقل من الهدف، والخطآن في نصين إنجليزيين. تقييم الشرائح الخاص بهذا النموذج لم يتم قياسه بعد. نتائج ملف التوقعات المرفق تخص مصدرًا غير محدد، فلا ننسبها إلى هذا النموذج.

## Evidence
[Benchmarks](../../BENCHMARKS.md) | [Evaluation](../../EVALUATION_REPORT.md)

## Contact / owner
Bayan project team.
