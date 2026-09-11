"""Build Lab 6 evidence and three cards without running training or frozen tests."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import Environment, StrictUndefined
from bayan.evaluation.slices import sliced_report, render_report
from bayan.preprocessing.core import PREPROC_VERSION

NOT_MEASURED = "لم يتم قياسه بعد."


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def section(document, heading):
    match = re.search(r"^## " + re.escape(heading) + r"\n(.*?)(?=^## |\Z)",
                      document, flags=re.M | re.S)
    if not match:
        raise ValueError(f"Missing evidence section: {heading}")
    return match.group(1).strip()


def snapshot(path):
    return path + " (SHA256 " + hashlib.sha256((ROOT / path).read_bytes()).hexdigest() + ")"


def behavioural_table(report):
    lines = ["| Suite | Passed | Failed | Skipped | Pass rate |",
             "|---|---:|---:|---:|---|"]
    for kind in ("invariance", "directional", "mft"):
        rows = [r for r in report["results"] if r["test_type"] == kind]
        counts = Counter(r["status"] for r in rows)
        if set(counts) - {"passed", "failed", "skipped"}:
            raise ValueError("Unknown behavioural result status")
        supplied = report["summary"][kind]
        if supplied["total"] != len(rows) or any(
                supplied[k] != counts[k] for k in ("passed", "failed", "skipped")):
            raise ValueError("Behavioural summary does not match cases")
        n = counts["passed"] + counts["failed"]
        rate = f"{counts['passed']/n:.1%}" if n else NOT_MEASURED
        lines.append(f"| {kind} | {counts['passed']} | {counts['failed']} | {counts['skipped']} | {rate} |")
    return "\n".join(lines)


def build():
    benchmarks = read("BENCHMARKS.md")
    model_metrics = section(benchmarks, "Lab 3 — Models")
    retrieval = section(benchmarks, "Lab 5 — Search")
    sliced = sliced_report()
    headline, slices = render_report(sliced, "data/eval/validation_predictions.csv")
    slices = slices.replace(
        "Separate predictions for the trained topic and dialect-aware models are unavailable locally; their sliced comparison remains pending.",
        "مقارنة شرائح النموذجين: " + NOT_MEASURED)
    behavioural = json.loads(read("data/eval/behavioural_results.json"))
    behaviour = behavioural_table(behavioural)
    with (ROOT / "data/eval/error_review_120.csv").open(encoding="utf-8-sig", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    if len(reviews) != 120 or len({r["feedback_id"] for r in reviews}) != 120:
        raise ValueError("Expected 120 unique error annotations")
    if any(not r["primary_category"] or not r["evidence"] for r in reviews):
        raise ValueError("Incomplete error annotations")
    counts = Counter(r["primary_category"] for r in reviews)
    taxonomy = "\n".join(["| Category | Count |", "|---|---:|"] +
                          [f"| {k} | {v} |" for k, v in sorted(counts.items())])
    statuses = ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["review_status"] for r in reviews).items()))
    taxonomy += "\n\nReview status: " + statuses + ".\n\n[Case evidence](data/eval/error_review_120.csv)."
    analysis = read("docs/ERROR_REVIEW_ANALYSIS.md")
    fixes = section(analysis, "أهم ثلاثة تحسينات")
    template = Environment(undefined=StrictUndefined, autoescape=False,
                           keep_trailing_newline=True).from_string(read("templates/model_card.md.j2"))
    outputs = {}
    card_links = []
    specs = [
        ("topic", "CAMeLBERT topic classifier", "تصنيف موضوع ملاحظات المستفيدين.",
         "artifacts/topic_classifier; CAMeL-Lab/bert-base-arabic-camelbert-mix",
         "data/raw/bayan_feedback.csv", "Topic classifier (CAMeLBERT)", behaviour),
        ("ner", "Bayan NER", "استخراج الكيانات من النص.",
         "scripts/train_ner.py; local trained artefact not supplied",
         "data/models/bayan_ner.conll", "NER", NOT_MEASURED),
        ("search", "Bayan two-stage search", "استرجاع الحالات المشابهة وإعادة ترتيبها.",
         "src/bayan/search/index.py + service.py; measured weights revision not recorded",
         "data/search/bayan_cases.csv", None, NOT_MEASURED),
    ]
    for slug, name, use, checkpoint, data, metric_name, behaviour_card in specs:
        if metric_name:
            metric_row = next(line for line in model_metrics.splitlines()
                              if line.startswith("| " + metric_name + " |"))
            metrics = "\n".join(["Recorded in BENCHMARKS.md; not rerun by this generator.",
                                  "| Model | Metric | Validation | Frozen Test | Train time |",
                                  "|---|---|---:|---:|---|", metric_row])
        else:
            metrics = "Recorded in BENCHMARKS.md; not rerun by this generator.\n\n" + retrieval
        card = template.render(
            model_name=name, intended_use=use, checkpoint=checkpoint,
            preproc_version=PREPROC_VERSION + " (current code; historical run version not recorded)",
            data_version=snapshot(data) + " (current local snapshot)",
            metrics_table=metrics, slices_table=NOT_MEASURED,
            behavioural_table=behaviour_card,
            limitations=read(f"docs/model_cards/{slug}_limitations.md").strip(),
            evidence="[Benchmarks](../../BENCHMARKS.md) | [Evaluation](../../EVALUATION_REPORT.md)")
        target = f"docs/model_cards/{slug}.md"
        outputs[target] = card
        card_links.append(f"- [{name}]({target})")
    document = [
        "# EVALUATION REPORT — Bayan", "## Executive headline", headline,
        "## Recorded model metrics",
        "Source: BENCHMARKS.md. Historical results; not rerun in Step 5.\n\n" + model_metrics,
        "## Sliced metrics with bootstrap CIs", slices,
        "## Behavioural suite", behaviour,
        "Source: data/eval/behavioural_results.json; local topic classifier.\n"
        "الثبات يختبر المسافات فقط (10 أزواج فريدة). MFT: ‏87.5%، أقل من هدف 90%.",
        "## Error taxonomy", taxonomy,
        "[Analysis and histogram](docs/ERROR_REVIEW_ANALYSIS.md).",
        "## Top fixes", fixes,
        "## Retrieval quality", "Source: BENCHMARKS.md; recorded Lab 5 run.\n\n" + retrieval,
        "الفرق بين اللغات: " + NOT_MEASURED,
        "## Model cards", "\n".join(card_links),
        "## Known limitations", read("docs/evaluation_limitations.md").strip(),
        "## Reproduce", "`python scripts/evaluation_report.py`\n\n"
        "Checks existing outputs without writing: `python scripts/evaluation_report.py --check`",
    ]
    outputs["EVALUATION_REPORT.md"] = "\n\n".join(document) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()
    if args.check:
        stale = [p for p, text in outputs.items()
                 if not (ROOT / p).exists() or read(p) != text]
        if stale:
            raise SystemExit("Stale or missing outputs: " + ", ".join(stale))
        print("Verified report and 3 model cards.")
    else:
        for path, text in outputs.items():
            destination = ROOT / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        print("Generated EVALUATION_REPORT.md and 3 model cards.")


if __name__ == "__main__":
    main()
