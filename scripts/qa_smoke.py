"""Lab 3B: run the QA smoke set."""

import json
from pathlib import Path


SMOKE_FILE = Path("data/eval/qa_smoke_set.json")


def load_smoke_set(path):
    """
    نقرأ ملف الـQA smoke set.
    الملف بصيغة SQuAD-style:
    data -> paragraphs -> context + qas
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_smoke_set(SMOKE_FILE)

    answerable_total = 0
    unanswerable_total = 0

    # نمر على كل الأسئلة
    for item in data["data"]:
        for paragraph in item["paragraphs"]:

            context = paragraph["context"]

            for qa in paragraph["qas"]:

                question = qa["question"]
                is_impossible = qa["is_impossible"]

                print("=" * 70)
                print("Question:", question)
                print("Context :", context)

                # في الـsmoke set نعرف مسبقاً الـexpected answer
                # عشان نتحقق من سلوك الـpipeline
                if is_impossible:
                    expected = None
                    unanswerable_total += 1
                else:
                    expected = qa["answers"][0]["text"]
                    answerable_total += 1

                print("Expected:", expected)

    print("\nQA Smoke Summary")
    print("Answerable questions   :", answerable_total)
    print("Unanswerable questions :", unanswerable_total)


if __name__ == "__main__":
    main()