# EVALUATION REPORT — Bayan

## Executive headline

On the supplied prediction fixture (2400 rows), accuracy is 0.8750 and macro-F1 is 0.8333, with 14 evaluated slices. The lowest-accuracy slice is y_true=parks (0.0000, n=300); these fixture results do not establish the performance of the trained topic and dialect-aware models.

## Recorded model metrics

Source: BENCHMARKS.md. Historical results; not rerun in Step 5.

| Model | Metric | Validation | Frozen Test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | - |
| Topic classifier (CAMeLBERT) | macro-F1 | 1.0000 | 1.0000 | 269.82 s |
| NER | entity-F1 | 1.0000 | 1.0000 | 208.46 s |
| QA | span/null smoke |  |  |  |

## Sliced metrics with bootstrap CIs

Source: `data/eval/validation_predictions.csv` (supplied fixture; model identity is not provided).

Percentile CIs: 95%; resamples=2000; seed=42.
Small slices have n < 30; they are flagged, not excluded.
Macro-F1 uses a fixed global label set, including absent classes as zero; class slices therefore require care when comparing F1.
Rows are resampled as paired true/predicted labels, assuming independent observations; intervals are pointwise and do not account for citizen-group dependence.
مقارنة شرائح النموذجين: لم يتم قياسه بعد.

| Dimension | Slice | n | Accuracy [CI] | Macro-F1 [CI] | Small slice |
|---|---|---:|---|---|---|
| overall | all | 2400 | 0.8750 [0.8617, 0.8888] | 0.8333 [0.8289, 0.8375] | no |
| lang | ar | 1200 | 0.7500 [0.7250, 0.7733] | 0.3750 [0.3750, 0.3750] | no |
| lang | en | 1200 | 1.0000 [1.0000, 1.0000] | 0.5000 [0.5000, 0.5000] | no |
| dialect_region | MSA | 1200 | 0.7500 [0.7250, 0.7742] | 0.3750 [0.3750, 0.3750] | no |
| dialect_region | NA | 1200 | 1.0000 [1.0000, 1.0000] | 0.5000 [0.5000, 0.5000] | no |
| y_true | billing | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| y_true | digital_services | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| y_true | licensing | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| y_true | lighting | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| y_true | parks | 300 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | no |
| y_true | roads | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| y_true | waste | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| y_true | water | 300 | 1.0000 [1.0000, 1.0000] | 0.1250 [0.1250, 0.1250] | no |
| length_bucket | medium | 1646 | 0.8894 [0.8742, 0.9046] | 0.8459 [0.8417, 0.8498] | no |
| length_bucket | short | 754 | 0.8435 [0.8170, 0.8674] | 0.6250 [0.6250, 0.6250] | no |

## Behavioural suite

| Suite | Passed | Failed | Skipped | Pass rate |
|---|---:|---:|---:|---|
| invariance | 200 | 0 | 0 | 100.0% |
| directional | 0 | 0 | 200 | لم يتم قياسه بعد. |
| mft | 14 | 2 | 0 | 87.5% |

Source: data/eval/behavioural_results.json; local topic classifier.
الثبات يختبر المسافات فقط (10 أزواج فريدة). MFT: ‏87.5%، أقل من هدف 90%.

## Error taxonomy

| Category | Count |
|---|---:|
| خلط تصنيفي: ري الحديقة | 33 |
| خلط تصنيفي: صيانة ألعاب الحديقة | 32 |
| خلط تصنيفي: ممرات الحديقة | 55 |

Review status: assistant_draft: 120.

[Case evidence](data/eval/error_review_120.csv).

[Analysis and histogram](docs/ERROR_REVIEW_ANALYSIS.md).

## Top fixes

| الأولوية | التغيير المقترح ودليله | الأثر المتوقع المشروط |
|---|---|---|
| 1 | تدريب أمثلة متقابلة تفرق بين خدمة داخل الحديقة وخدمة الطريق العام، مع توضيح أين يصنّف ري الحدائق. الشواهد: FB-000488 وFB-005608 وFB-002288. تُكتب أمثلة تدريب جديدة؛ لا تُنقل حالات التحقق إلى التدريب. | إذا صحّح هذا التغيير 12–24 حالة من العينة دون إفساد أي توقع آخر، تزيد Accuracy على ملف الـ2400 بمقدار 0.50–1.00 نقطة مئوية. |
| 2 | تبديل أسماء الأماكن في أمثلة التدريب وإضافة اختبار ثبات لها: نفس طلب صيانة الألعاب مع اسم حي ثم اسم شارع. الشواهد: FB-002288 وFB-004568 وFB-007408. الهدف التأكد أن اسم الطريق لا يطغى على الخدمة المطلوبة. | إذا صُحّحت 6–12 حالة دون أخطاء جديدة، تكون الزيادة 0.25–0.50 نقطة مئوية على الملف نفسه. |
| 3 | مقارنة النص الأصلي بنسخة مصححة إملائيًا، ثم تجربة تطبيع عربي مناسب أو تنويعات تدريب إذا ثبت التحسن. الشواهد: FB-001768 وFB-004048 وFB-002528. لا نغيّر المعالجة المشتركة دون فحص تأثيرها على بقية الحالات. | إذا صُحّحت 1–3 حالات دون أخطاء جديدة، تكون الزيادة نحو 0.04–0.13 نقطة مئوية على الملف نفسه. |

هذه الأرقام سيناريوهات تخطيطية وليست مكاسب مقاسة أو توقعات إحصائية مستنتجة من التجربة. الحساب هو عدد التصحيحات المفترض ÷ 2400 × 100؛ لا نفترض انتقال الأثر إلى باقي البيانات. قد تكون النتيجة صفرًا أو تراجعًا، وقد تتداخل الحالات بين التحسينات، لذلك لا تُجمع المكاسب. يُقاس الأثر الفعلي على بيانات تحقق مستقلة مع مراجعة Macro-F1 وفترات الثقة أيضًا.

## Retrieval quality

Source: BENCHMARKS.md; recorded Lab 5 run.

| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 0.0026 | 0.0026 | 16.86 ms |
| + cross-encoder rerank | 0.0179 | 0.0244 | 56.25 ms |
| cross-lingual slice | N/A | N/A | N/A |

- no-answer empty-correct: 20 / 20
- cross-lingual gap: N/A

الفرق بين اللغات: لم يتم قياسه بعد.

## Model cards

- [CAMeLBERT topic classifier](docs/model_cards/topic.md)
- [Bayan NER](docs/model_cards/ner.md)
- [Bayan two-stage search](docs/model_cards/search.md)

## Known limitations

ملف التوقعات المرفق لا يحدد النموذج الذي أنتجه. شرائح اللهجات محدودة، ولا توجد شريحة خليجية فيه. فترات الثقة تفترض استقلال الصفوف، والصياغات المتكررة تحد من تنوع البيانات. تحليل الأخطاء أعدّه المساعد. اختبارات المشاعر والمقارنة بين النموذجين لم يتم قياسها بعد. آثار التحسينات المقترحة تقديرية ولم تُختبر.

## Reproduce

`python scripts/evaluation_report.py`

Checks existing outputs without writing: `python scripts/evaluation_report.py --check`
