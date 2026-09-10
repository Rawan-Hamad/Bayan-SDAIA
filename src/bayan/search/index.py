"""Lab 5, step 1: build and persist a versioned bilingual FAISS index."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data/search/bayan_cases.csv"


def build_index(
    prefix: str,
    limit: int | None = None,
    *,
    corpus_path: str | Path = DEFAULT_CORPUS,
    model: str = DEFAULT_MODEL,
    revision: str = "main",
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """Save <prefix>.faiss, <prefix>_metadata.json and <prefix>_manifest.json.

    FAISS row IDs match metadata list positions. Encode queries with the same
    pinned model and preprocessing, then L2-normalise them before searching;
    inner-product scores then equal cosine similarity. ``limit`` is useful for
    the supplied smoke contract; omit it when building the full corpus.
    """
    import faiss
    import numpy as np
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    corpus_path = Path(corpus_path)
    metadata = []
    texts = []
    seen_ids = set()
    with corpus_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"case_id", "case_text"}.issubset(reader.fieldnames or []):
            raise ValueError("Corpus must contain case_id and case_text columns")
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            text = preprocess(row.get("case_text") or "")
            if not case_id or case_id in seen_ids:
                raise ValueError(f"Empty or duplicate case_id: {case_id!r}")
            if not text:
                raise ValueError(f"Empty case_text for {case_id}")
            seen_ids.add(case_id)
            metadata.append({**row, "vector_id": len(metadata), "indexed_text": text})
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    if not texts:
        raise ValueError("Corpus contains no cases")

    # Resolve moving branch/tag names to an immutable Hub commit before loading.
    model_revision = model_info(model, revision=revision).sha
    if not model_revision:
        raise ValueError("Could not resolve the model to an immutable revision")
    encoder = SentenceTransformer(model, revision=model_revision, device=device)
    vectors = np.array(
        encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        ),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(metadata) or vectors.shape[1] == 0:
        raise ValueError("Encoder returned an invalid embedding shape")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Embeddings contain invalid or zero-length vectors")
    faiss.normalize_L2(vectors)
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6):
        raise ValueError("L2 normalisation failed")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    index_path = Path(f"{prefix}.faiss")
    metadata_path = Path(f"{prefix}_metadata.json")
    manifest_path = Path(f"{prefix}_manifest.json")
    # A stale manifest must not mark an interrupted rebuild as complete.
    manifest_path.unlink(missing_ok=True)
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def sha256(path: Path) -> str:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    manifest = {
        "schema_version": 1,
        "model": model,
        "model_revision": model_revision,
        "preproc_version": PREPROC_VERSION,
        "preprocessing": "bayan.preprocessing.core.preprocess",
        "preproc_sha256": sha256(Path(__file__).resolve().parents[1] / "preprocessing/core.py"),
        "n_vectors": int(index.ntotal),
        "dim": int(index.d),
        "index_type": "IndexFlatIP",
        "metric": "cosine",
        "l2_normalized": True,
        "text_field": "case_text",
        "max_seq_length": int(encoder.max_seq_length),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": sha256(corpus_path),
        "limit": limit,
        "index_file": index_path.name,
        "metadata_file": metadata_path.name,
        "index_sha256": sha256(index_path),
        "metadata_sha256": sha256(metadata_path),
    }
    # Write the manifest last, after both persisted artifacts are complete.
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="artifacts/search/case_index_v1")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(build_index(**vars(args)), ensure_ascii=False, indent=2))
