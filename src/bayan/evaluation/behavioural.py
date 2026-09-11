"""Lab 6: bilingual invariance, directional and minimum-functionality checks.

Predictors accept one raw string. Topic predictors return a topic label;
sentiment predictors return a finite score where higher means more positive.
Missing predictors are skipped, never counted as passes.
"""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/eval/behavioural_templates.csv"

MFT_CASES = {
    "billing": ("My service bill contains an incorrect charge.", "\u0641\u0627\u062a\u0648\u0631\u0629 \u0627\u0644\u062e\u062f\u0645\u0629 \u0641\u064a\u0647\u0627 \u0645\u0628\u0644\u063a \u062e\u0627\u0637\u0626."),
    "digital_services": ("The government website login page will not load.", "\u0635\u0641\u062d\u0629 \u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644 \u0641\u064a \u0627\u0644\u0645\u0648\u0642\u0639 \u0627\u0644\u062d\u0643\u0648\u0645\u064a \u0644\u0627 \u062a\u0641\u062a\u062d."),
    "licensing": ("I need to renew my business license.", "\u0623\u062d\u062a\u0627\u062c \u062a\u062c\u062f\u064a\u062f \u0631\u062e\u0635\u0629 \u0627\u0644\u0645\u062d\u0644."),
    "lighting": ("The street lights are broken at night.", "\u0625\u0646\u0627\u0631\u0629 \u0627\u0644\u0634\u0627\u0631\u0639 \u0645\u0639\u0637\u0644\u0629 \u0641\u064a \u0627\u0644\u0644\u064a\u0644."),
    "parks": ("The public park playground needs maintenance.", "\u0623\u0644\u0639\u0627\u0628 \u0627\u0644\u062d\u062f\u064a\u0642\u0629 \u0627\u0644\u0639\u0627\u0645\u0629 \u062a\u062d\u062a\u0627\u062c \u0635\u064a\u0627\u0646\u0629."),
    "roads": ("There is a large pothole in the road.", "\u062a\u0648\u062c\u062f \u062d\u0641\u0631\u0629 \u0643\u0628\u064a\u0631\u0629 \u0641\u064a \u0627\u0644\u0637\u0631\u064a\u0642."),
    "waste": ("The garbage bins have not been emptied.", "\u0644\u0645 \u064a\u062a\u0645 \u062a\u0641\u0631\u064a\u063a \u062d\u0627\u0648\u064a\u0627\u062a \u0627\u0644\u0646\u0641\u0627\u064a\u0627\u062a."),
    "water": ("The water supply to my house has stopped.", "\u0627\u0646\u0642\u0637\u0639\u062a \u0627\u0644\u0645\u064a\u0627\u0647 \u0639\u0646 \u0645\u0646\u0632\u0644\u064a."),
}


def build_cases(data=DEFAULT_DATA):
    """Expand supplied skeletons; use whitespace invariance and positive/negative pairs.

    Directional templates already contain negation. Remove that known negation
    for the baseline instead of introducing a semantically ambiguous double negative.
    """
    with Path(data).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cases, ids = [], set()
    for row in rows:
        required = ("test_id", "test_type", "lang", "template", "term", "expected_relation")
        if any(not row.get(key) for key in required):
            raise ValueError("Incomplete behavioural template")
        if row["test_id"] in ids:
            raise ValueError("Duplicate test_id")
        ids.add(row["test_id"])
        text = row["template"].format(term=row["term"])
        kind = row["test_type"]
        if kind == "invariance" and row["expected_relation"] == "topic unchanged":
            base, changed = text, "  " + text.replace(" ", "   ") + "\n"
        elif kind == "directional" and row["expected_relation"] == "sentiment must not improve after negation":
            token = {"en": "is not working", "ar": "\u0644\u0627 \u062a\u0639\u0645\u0644"}.get(row["lang"])
            replacement = {"en": "is working", "ar": "\u062a\u0639\u0645\u0644"}.get(row["lang"])
            if token is None or text.count(token) != 1:
                raise ValueError("Unsupported negation template")
            base, changed = text.replace(token, replacement, 1), text
        else:
            raise ValueError("Unsupported behavioural relation")
        cases.append(dict(test_id=row["test_id"], test_type=kind, lang=row["lang"],
                          baseline=base, text=changed))
    for topic, texts in MFT_CASES.items():
        for lang, text in zip(("en", "ar"), texts):
            test_id = f"MFT-{topic}-{lang}"
            if test_id in ids:
                raise ValueError("Duplicate MFT test_id")
            cases.append(dict(test_id=test_id, test_type="mft", lang=lang,
                              text=text, expected=topic))
    return cases


def run_behavioural_suite(topic_predictor=None, sentiment_predictor=None, *,
                          data=DEFAULT_DATA):
    """Return case evidence, coverage and rates (fractions, not percentages).

    Exceptions/invalid predictions count as failures. Target attainment requires
    full category coverage. Repeated skeletons are retained and unique cases
    reported, so repeated rows cannot be mistaken for independent coverage.
    """
    results = []
    for case in build_cases(data):
        result = dict(case)
        kind = case["test_type"]
        predictor = sentiment_predictor if kind == "directional" else topic_predictor
        if predictor is None:
            result.update(status="skipped", reason="sentiment predictor unavailable" if
                          kind == "directional" else "topic predictor unavailable")
        else:
            try:
                prediction = predictor(case["text"])
                baseline = predictor(case["baseline"]) if kind != "mft" else None
                if kind == "directional":
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                           not math.isfinite(v) for v in (prediction, baseline)):
                        raise ValueError("Sentiment must return a finite numeric score")
                    passed = prediction <= baseline
                else:
                    if not isinstance(prediction, str) or not prediction.strip():
                        raise ValueError("Topic must return a nonempty label")
                    if kind == "invariance" and (not isinstance(baseline, str) or not baseline.strip()):
                        raise ValueError("Baseline topic must return a nonempty label")
                    passed = prediction == (case["expected"] if kind == "mft" else baseline)
                result.update(status="passed" if passed else "failed",
                              prediction=prediction, baseline_prediction=baseline)
            except Exception as exc:
                result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        results.append(result)
    summary = {}
    for kind, target in (("invariance", .95), ("directional", None), ("mft", .90)):
        group = [r for r in results if r["test_type"] == kind]
        counts = {status: sum(r["status"] == status for r in group)
                  for status in ("passed", "failed", "skipped")}
        evaluated = counts["passed"] + counts["failed"]
        rate = counts["passed"] / evaluated if evaluated else None
        summary[kind] = dict(total=len(group), evaluated=evaluated, **counts,
                             unique_cases=len({(r.get("baseline"), r["text"], r.get("expected"))
                                               for r in group}),
                             pass_rate=rate,
                             failure_rate=counts["failed"] / evaluated if evaluated else None,
                             target=target,
                             target_met=(rate >= target if target is not None and
                                         evaluated == len(group) and evaluated else None))
    return dict(summary=summary, results=results)


def local_predictor(path, *, sentiment=False):
    """Load only an explicitly provided local Hugging Face checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    from bayan.preprocessing.core import preprocess

    path = Path(path)
    if not (path / "config.json").is_file():
        raise ValueError(f"No local checkpoint config at {path}")
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    infer = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    cache = {}

    def predict(text):
        if text not in cache:
            output = infer(preprocess(text), truncation=True,
                           max_length=model.config.max_position_embeddings, top_k=None)
            if sentiment:
                probabilities = {item["label"].lower(): item["score"] for item in output}
                if not set(probabilities).issubset({"positive", "negative", "neutral"}) or not {
                        "positive", "negative"}.issubset(probabilities):
                    raise ValueError("Sentiment labels must be positive/negative[/neutral]")
                cache[text] = float(probabilities["positive"] - probabilities["negative"])
            else:
                cache[text] = max(output, key=lambda item: item["score"])["label"]
        return cache[text]
    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--topic-model", type=Path)
    parser.add_argument("--sentiment-model", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON case evidence")
    args = parser.parse_args()
    report = run_behavioural_suite(
        local_predictor(args.topic_model) if args.topic_model else None,
        local_predictor(args.sentiment_model, sentiment=True) if args.sentiment_model else None,
        data=args.data)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
