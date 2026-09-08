"""Lab 3 starter: dataset construction and split integrity."""


import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

from bayan.preprocessing.core import preprocess


TOPICS = [
    "roads",
    "lighting",
    "waste",
    "water",
    "billing",
    "digital_services",
    "licensing",
    "parks",
]


def build_topic_dataset(csv_path: str = "data/raw/bayan_feedback.csv",seed: int = 42,) -> DatasetDict:
    #seed = رقم نثبّت فيه العشوائية
    # ---------------------------------------------------------
    # 1) LOAD DATA
    # ---------------------------------------------------------
    df = pd.read_csv(csv_path)

    # ---------------------------------------------------------
    # 2) SAME PREPROCESSING AS LAB 1
    # ---------------------------------------------------------
    df["text"] = df["text"].astype(str).map(preprocess)

    # ---------------------------------------------------------
    # 3) CONVERT TOPIC NAME TO NUMERIC LABEL
    # ---------------------------------------------------------
    # Example:
    # roads -> 0
    # lighting -> 1
    # ...
    df["label"] = df["topic"].map(
        {topic: i for i, topic in enumerate(TOPICS)}
    )

    # ---------------------------------------------------------
    # 4) FIRST GROUPED SPLIT
    #
    # 80% -> temporary train
    # 20% -> frozen test
    #
    # Group by citizen_group_id
    # ---------------------------------------------------------
    #GroupShuffleSplit = أداة تقسيم من scikit-learn
    #n-splits سوِّ لي تقسيم واحد فقط train , test
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_idx, test_idx = next(
        first_split.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # ---------------------------------------------------------
    # 5) SECOND GROUPED SPLIT
    #
    # نأخذ من الـ80% جزء للـvalidation.
    #
    # test_size=0.125 من الـ80%
    # = تقريبًا 10% من البيانات الأصلية.
    #
    # Final:
    # Train      ≈ 70%
    # Validation ≈ 10%
    # Test       ≈ 20%
    # ---------------------------------------------------------
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=seed,
    )

    final_train_idx, validation_idx = next(
        second_split.split(
            train_df,
            groups=train_df["citizen_group_id"],
        )
    )

    final_train_df = train_df.iloc[final_train_idx].copy()
    validation_df = train_df.iloc[validation_idx].copy()

    # ---------------------------------------------------------
    # 6) VERIFY ZERO GROUP OVERLAP نتحقق ان كل سبليت مايحتوي على اي دي قروب ثاني
    # ---------------------------------------------------------
    train_groups = set(final_train_df["citizen_group_id"])
    validation_groups = set(validation_df["citizen_group_id"])
    test_groups = set(test_df["citizen_group_id"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    # ---------------------------------------------------------
    # 7) RETURN HUGGING FACE DATASETDICT
    # ---------------------------------------------------------
    return DatasetDict(
        train=Dataset.from_pandas(
            final_train_df,
            preserve_index=False,
        ),
        validation=Dataset.from_pandas(
            validation_df,
            preserve_index=False,
        ),
        test=Dataset.from_pandas(
            test_df,
            preserve_index=False,
        ),
    )