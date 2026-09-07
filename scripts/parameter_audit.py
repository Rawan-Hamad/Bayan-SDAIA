"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""

"""Lab 2 starter: parameter audit script."""

import torch
from transformers import AutoModel


def audit(checkpoint_name):
    print(f"\n==========================================")
    print(f" Auditing Checkpoint: {checkpoint_name}")
    print(f"==========================================")

    # تحميل النموذج من Hugging Face
    model = AutoModel.from_pretrained(checkpoint_name)

    total_params = 0
    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0
    }

    for name, param in model.named_parameters():
        num_el = param.numel()
        total_params += num_el

        # تصنيف المعلمات إلى فئات (Buckets)
        if "embeddings" in name:
            buckets["embeddings"] += num_el
        elif "attention" in name or "attn" in name:
            buckets["attention"] += num_el
        elif "intermediate" in name or "output" in name or "ffn" in name:
            buckets["ffn"] += num_el
        elif "LayerNorm" in name or "norm" in name:
            buckets["norms"] += num_el
        elif "pooler" in name:
            buckets["pooler"] += num_el
        else:
            buckets["other"] += num_el

    print(f"Total Parameters: {total_params:,}\n")
    print("Parameter Buckets:")
    for bucket_name, count in buckets.items():
        percentage = (count / total_params) * 100 if total_params > 0 else 0
        print(f"  - {bucket_name.capitalize():<12}: {count:,} ({percentage:.2f}%)")

    return total_params, buckets


def main():
    checkpoints = [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    ]

    for cp in checkpoints:
        audit(cp)


if __name__ == "__main__":
    main()