"""Lab 3B: fine-tune token classification with correct alignment."""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bayan.models.ner import align_labels


CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

NER_FILE = "data/models/bayan_ner.conll"

LABELS = [
    "O",
    "B-SERVICE",
    "I-SERVICE",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE",
    "B-REFERENCE",
    "I-REFERENCE",
    "B-ORG",
    "I-ORG",
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def read_conll(path):
    """
    Read CoNLL-style data.

    Expected format:
        token label
        token label

        token label
        ...

    Blank line = end of sentence.
    """

    sentences = []
    tags = []

    current_tokens = []
    current_tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Blank line = sentence boundary
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    tags.append(current_tags)

                    current_tokens = []
                    current_tags = []

                continue

            # Ignore comments / document markers
            if line.startswith("#") or line.startswith("-DOCSTART-"):
                continue

            parts = line.split()

            # First column = token
            # Last column = BIO label
            token = parts[0]
            label = parts[-1]

            current_tokens.append(token)
            current_tags.append(label2id[label])

    # Last sentence if file does not end with blank line
    if current_tokens:
        sentences.append(current_tokens)
        tags.append(current_tags)

    return sentences, tags


def build_dataset(path, seed):
    tokens, labels = read_conll(path)

    dataset = Dataset.from_dict(
        {
            "tokens": tokens,
            "ner_tags": labels,
        }
    )

    # 80% train, 20% temporary
    split1 = dataset.train_test_split(
        test_size=0.20,
        seed=seed,
    )

    # Split temporary 50/50 -> 10% validation, 10% test
    split2 = split1["test"].train_test_split(
        test_size=0.50,
        seed=seed,
    )

    return DatasetDict(
        {
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"],
        }
    )


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=256,
        is_split_into_words=True,
    )

    all_labels = []

    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)

        aligned = align_labels(
            word_ids,
            word_labels,
        )

        all_labels.append(aligned)

    tokenized["labels"] = all_labels

    return tokenized


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        sentence_predictions = []
        sentence_labels = []

        for pred_id, label_id in zip(prediction, label):

            # Ignore continuation subwords / special tokens
            if label_id == -100:
                continue

            sentence_predictions.append(
                id2label[int(pred_id)]
            )

            sentence_labels.append(
                id2label[int(label_id)]
            )

        true_predictions.append(sentence_predictions)
        true_labels.append(sentence_labels)

    return {
        "precision": precision_score(
            true_labels,
            true_predictions,
        ),
        "recall": recall_score(
            true_labels,
            true_predictions,
        ),
        "f1": f1_score(
            true_labels,
            true_predictions,
        ),
        "accuracy": accuracy_score(
            true_labels,
            true_predictions,
        ),
    }


def main():
    args = parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------
    # 1. Load NER dataset
    # -------------------------

    dataset = build_dataset(
        NER_FILE,
        args.seed,
    )

    print(dataset)

    # -------------------------
    # 2. Load tokenizer
    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    # -------------------------
    # 3. Tokenize + align labels
    # -------------------------

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(
            batch,
            tokenizer,
        ),
        batched=True,
    )

    # -------------------------
    # 4. Load pretrained model
    # -------------------------

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # -------------------------
    # 5. Training configuration
    # -------------------------

    training_args = TrainingArguments(
        output_dir="runs/ner",

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=4,

        weight_decay=0.01,
        warmup_ratio=0.1,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=2,

        fp16=torch.cuda.is_available(),

        logging_steps=50,

        seed=args.seed,
        report_to="none",

        # CAMeLBERT workaround
        save_safetensors=False,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )

    # -------------------------
    # 6. Trainer
    # -------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        tokenizer=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # -------------------------
    # 7. Fine-tune
    # -------------------------

    trainer.train()

    # -------------------------
    # 8. Frozen test evaluation
    # -------------------------

    test_metrics = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print("\nNER Test Results:")
    print(test_metrics)

    # -------------------------
    # 9. Save artefact
    # -------------------------

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSaved NER artefact to: {output_dir}")


if __name__ == "__main__":
    main()
    
    