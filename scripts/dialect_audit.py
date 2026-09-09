"""Lab 4: describe supplied dialect labels in the Arabic slice."""

import argparse
from collections import Counter
import csv
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/raw/bayan_feedback.csv"


def audit_dialects(path: Path) -> tuple[int, Counter]:
    """Count Arabic records, including unlabelled rows in the denominator."""
    counts = Counter()
    total = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"lang", "dialect_region"}.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV must contain lang and dialect_region columns")
        for row in reader:
            total += 1
            if (row.get("lang") or "").strip().lower() == "ar":
                counts[(row.get("dialect_region") or "").strip() or "Unknown"] += 1
    return total, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    total, counts = audit_dialects(args.data)
    arabic_total = sum(counts.values())
    print(f"Source: {args.data}")
    print("Method: supplied dialect_region labels; descriptive audit across all splits.")
    print(f"Total records: {total}; Arabic records: {arabic_total}")
    if not arabic_total:
        print("No Arabic records; dialect proportions and MSA coverage are undefined.")
        return
    print("\n| Dialect / region | Count | % of Arabic |")
    print("|---|---:|---:|")
    for dialect, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"| {dialect} | {count} | {100 * count / arabic_total:.2f}% |")
    msa = sum(count for label, count in counts.items() if label.casefold() == "msa")
    print(
        f"\nMSA-only evaluation covers {100 * msa / arabic_total:.2f}% of Arabic "
        "records and cannot establish performance on the remaining dialects; "
        "report Gulf/MSA metrics separately. These counts do not measure model quality."
    )


if __name__ == "__main__":
    main()
