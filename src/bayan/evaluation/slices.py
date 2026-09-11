"""Lab 6 step 2: classification slices with percentile bootstrap intervals."""

import argparse
import csv
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/eval/validation_predictions.csv"


def sliced_report(data=DEFAULT_DATA, *, n_boot=2000, seed=42, alpha=0.05, min_size=30):
    """Return overall, language, dialect, true-class and length metrics.

    Accept a CSV path or iterable of row dictionaries. Macro-F1 uses the same
    label set (union of true/predicted labels across the input) in every slice,
    with zero F1 for undefined classes. Recompute F1 for each paired resample
    of true/predicted labels; do not bootstrap averages of per-row F1 values.
    CIs assume independent rows and are pointwise, not simultaneous intervals.
    """
    if type(n_boot) is not int or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if type(min_size) is not int or min_size <= 0:
        raise ValueError("min_size must be a positive integer")
    if not isinstance(alpha, (float, int)) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(data, (str, Path)):
        with Path(data).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = list(data)
    if not rows:
        raise ValueError("No prediction rows supplied")
    columns = ("lang", "dialect_region", "y_true", "length_bucket", "y_pred")
    cleaned = []
    for row in rows:
        if not isinstance(row, dict) or not set(columns).issubset(row):
            raise ValueError(f"Every row must contain {columns}")
        item = {key: str(row[key]).strip() if row[key] is not None else "" for key in columns}
        if not item["y_true"] or not item["y_pred"]:
            raise ValueError("True and predicted labels must be non-empty")
        for key in ("lang", "dialect_region", "length_bucket"):
            item[key] = item[key] or "Unknown"
        cleaned.append(item)
    labels = sorted({row[key] for row in cleaned for key in ("y_true", "y_pred")})
    label_ids = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    groups = [("overall", "all", cleaned)]
    for dimension in ("lang", "dialect_region", "y_true", "length_bucket"):
        for value in sorted({row[dimension] for row in cleaned}):
            groups.append((dimension, value, [row for row in cleaned if row[dimension] == value]))
    rng = np.random.default_rng(seed)

    def scores(codes):
        matrix = np.bincount(codes, minlength=n_labels ** 2).reshape(n_labels, n_labels)
        tp = matrix.diagonal()
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(2.0 * tp, denominator, out=np.zeros(n_labels), where=denominator != 0)
        return float(tp.sum() / matrix.sum()), float(f1.mean())

    results = []
    for dimension, value, group in groups:
        codes = np.array([label_ids[row["y_true"]] * n_labels + label_ids[row["y_pred"]]
                          for row in group], dtype=np.int64)
        accuracy, macro_f1 = scores(codes)
        boot = np.empty((n_boot, 2))
        for iteration in range(n_boot):
            boot[iteration] = scores(codes[rng.integers(0, len(codes), size=len(codes))])
        lower, upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
        results.append({"dimension": dimension, "value": value, "n": len(group),
                        "small_slice": len(group) < min_size,
                        "accuracy": accuracy, "accuracy_ci": [float(lower[0]), float(upper[0])],
                        "macro_f1": macro_f1, "macro_f1_ci": [float(lower[1]), float(upper[1])]})
    return {"labels": labels, "n_boot": n_boot, "seed": seed, "alpha": alpha,
            "min_size": min_size, "n_slices": len(results) - 1, "slices": results}


def render_report(report, source):
    overall = report["slices"][0]
    worst = min(report["slices"][1:], key=lambda row: row["accuracy"])
    headline = (
        f"On the supplied prediction fixture ({overall['n']} rows), accuracy is {overall['accuracy']:.4f} "
        f"and macro-F1 is {overall['macro_f1']:.4f}, with {report['n_slices']} evaluated slices. "
        f"The lowest-accuracy slice is {worst['dimension']}={worst['value']} "
        f"({worst['accuracy']:.4f}, n={worst['n']}); these fixture results do not establish "
        "the performance of the trained topic and dialect-aware models."
    )
    lines = [f"Source: `{source}` (supplied fixture; model identity is not provided).", "",
             f"Percentile CIs: {(1-report['alpha'])*100:.0f}%; resamples={report['n_boot']}; seed={report['seed']}.",
             f"Small slices have n < {report['min_size']}; they are flagged, not excluded.",
             "Macro-F1 uses a fixed global label set, including absent classes as zero; class slices therefore require care when comparing F1.",
             "Rows are resampled as paired true/predicted labels, assuming independent observations; intervals are pointwise and do not account for citizen-group dependence.",
             "Separate predictions for the trained topic and dialect-aware models are unavailable locally; their sliced comparison remains pending.", "",
             "| Dimension | Slice | n | Accuracy [CI] | Macro-F1 [CI] | Small slice |",
             "|---|---|---:|---|---|---|"]
    for row in report["slices"]:
        acc_lo, acc_hi = row["accuracy_ci"]
        f1_lo, f1_hi = row["macro_f1_ci"]
        value = row["value"].replace("|", "\\|")
        lines.append(f"| {row['dimension']} | {value} | {row['n']} | "
                     f"{row['accuracy']:.4f} [{acc_lo:.4f}, {acc_hi:.4f}] | "
                     f"{row['macro_f1']:.4f} [{f1_lo:.4f}, {f1_hi:.4f}] | "
                     f"{'YES' if row['small_slice'] else 'no'} |")
    return headline, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--report-path", type=Path, help="Update only headline and sliced-metrics sections")
    args = parser.parse_args()
    report = sliced_report(args.data, n_boot=args.n_boot, seed=args.seed, min_size=args.min_size)
    headline, table = render_report(report, args.data.as_posix())
    if args.report_path:
        document = args.report_path.read_text(encoding="utf-8")
        for heading, content in (("Executive headline", headline), ("Sliced metrics with bootstrap CIs", table)):
            pattern = rf"(^## {re.escape(heading)}\r?\n).*?(?=^## |\Z)"
            document, count = re.subn(pattern, lambda match: match.group(1) + content + "\n\n",
                                      document, flags=re.MULTILINE | re.DOTALL)
            if count != 1:
                raise ValueError(f"Expected exactly one report section: {heading}")
        args.report_path.write_text(document, encoding="utf-8")
    print(headline + "\n\n" + table)


if __name__ == "__main__":
    main()
