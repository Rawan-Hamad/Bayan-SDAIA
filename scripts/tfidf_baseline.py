"""Lab 3A starter: TF-IDF + LinearSVC baseline."""

import pandas as pd

# TF-IDF يحول النص إلى numeric features
from sklearn.feature_extraction.text import TfidfVectorizer

# LinearSVC هو الـclassical classifier اللي راح يتعلم الـtopics
from sklearn.svm import LinearSVC

# Pipeline تربط TF-IDF والـclassifier في مسار واحد
from sklearn.pipeline import Pipeline

# Macro-F1 هو الـmetric المطلوبة في اللاب
from sklearn.metrics import f1_score

# نفس preprocessing حق Lab 1
# عشان ما يصير train/serve preprocessing skew
from bayan.preprocessing.core import preprocess


def main():
    # ---------------------------------------------------------
    # 1) LOAD DATA
    # ---------------------------------------------------------
    # نقرأ Dataset الخاصة بـBayan topic classification
    df = pd.read_csv("data/raw/bayan_feedback.csv")
    print(f"Total rows: {len(df)}")
    print("\nSupplied splits:")
    print(df["split"].value_counts())

    # ---------------------------------------------------------
    # 2) APPLY THE SAME PREPROCESSING CONTRACT
    # ---------------------------------------------------------
    # نستخدم نفس preprocess اللي بنيناه في Lab 1
    # بدل ما يكون لكل موديل تنظيف مختلف.
    df["text"] = df["text"].astype(str).map(preprocess)

    # ---------------------------------------------------------
    # 3) USE THE SUPPLIED SPLIT
    # ---------------------------------------------------------
    # هنا ما نسوي split جديد.
    # Step 1 مطلوب يستخدم supplied split الموجودة في CSV.
    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    print("\nRows per split:")
    print("Train:", len(train_df))
    print("Validation:", len(validation_df))
    print("Test:", len(test_df))

    # ---------------------------------------------------------
    # 4) BUILD THE BASELINE
    # ---------------------------------------------------------
    # Pipeline:
    #
    # Text
    #   ↓
    # TF-IDF
    #   ↓
    # Numeric features
    #   ↓
    # LinearSVC
    #   ↓
    # Topic prediction

    baseline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(),
        ),
        (
            "classifier",
            LinearSVC(random_state=42),
        ),
    ])

    # ---------------------------------------------------------
    # 5) TRAIN
    # ---------------------------------------------------------
    # X = text
    # y = topic
    #
    # مثال:
    # "الفاتورة مرتفعة" → billing
    baseline.fit(train_df["text"], train_df["topic"])
    # ---------------------------------------------------------
    # 6) VALIDATION PREDICTION
    # ---------------------------------------------------------
    validation_predictions = baseline.predict(validation_df["text"])

    # نحسب Macro-F1 على validation
    validation_f1 = f1_score(
        validation_df["topic"],
        validation_predictions,
        average="macro",
    )

    # ---------------------------------------------------------
    # 7) FROZEN TEST PREDICTION
    # ---------------------------------------------------------
    test_predictions = baseline.predict(
        test_df["text"]
    )

    # نحسب Macro-F1 على test
    test_f1 = f1_score(
        test_df["topic"],
        test_predictions,
        average="macro",
    )

    # ---------------------------------------------------------
    # 8) PRINT RESULTS
    # ---------------------------------------------------------
    print("\nTF-IDF + LinearSVC BASELINE")
    print("--------------------------------")
    print(f"Validation macro-F1 : {validation_f1:.4f}")
    print(f"Frozen test macro-F1: {test_f1:.4f}")

    # ---------------------------------------------------------
    # 9) APPEND RESULT TO BENCHMARKS.md
    # ---------------------------------------------------------
    # مهم:
    # نحفظ OUR RUN، مو ننسخ الـreference من السلايد.

    with open("BENCHMARKS.md", "a", encoding="utf-8") as file:
        file.write("\n## Lab 3A — TF-IDF Baseline\n\n")
        file.write("| Model | Metric | Validation | Frozen Test |\n")
        file.write("|---|---|---:|---:|\n")
        file.write(
            f"| TF-IDF + LinearSVC | macro-F1 | "
            f"{validation_f1:.4f} | {test_f1:.4f} |\n"
        )


if __name__ == "__main__":
    main()
