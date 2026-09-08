"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from bayan.models.data import TOPICS, build_topic_dataset


CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    accuracy = (predictions == labels).mean()

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Training on CPU")

    # 1. Load tokenizer/checkpoint chosen from Lab 1
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    # 2. Load grouped dataset from Step 2
    dataset = build_topic_dataset(
        "data/raw/bayan_feedback.csv",
        seed=args.seed,
    )

    print(dataset)

    # 3. Tokenize text
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
    )

    # 4. Load pretrained Transformer + classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(TOPICS),
        id2label={
            i: topic
            for i, topic in enumerate(TOPICS)
        },
        label2id={
            topic: i
            for i, topic in enumerate(TOPICS)
        },
    )

    # 5. Training configuration
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=4,

        warmup_ratio=0.1,
        weight_decay=0.01,

        fp16=torch.cuda.is_available(),

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,

        # Avoid non-contiguous tensor error with safetensors
        save_safetensors=False,

        logging_steps=50,

        seed=args.seed,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # 6. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    # 7. Fine-tune
    trainer.train()

    # 8. Evaluate on frozen test set
    test_results = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print("\nFINAL TEST RESULTS")
    print(test_results)

    # 9. Save final model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved artefact to: {output_dir}")


if __name__ == "__main__":
    main()