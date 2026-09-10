"""Lab 5, step 2: version-checked, two-stage bilingual case search."""

import hashlib
import json
import math
from pathlib import Path

from bayan.preprocessing import core


class CaseSearch:
    def __init__(
        self,
        prefix: str,
        *,
        reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        reranker_revision: str = "main",
        device: str | None = None,
    ):
        import faiss
        import numpy as np
        from huggingface_hub import model_info
        from sentence_transformers import CrossEncoder, SentenceTransformer

        prefix = Path(prefix)
        manifest_path = Path(f"{prefix}_manifest.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = self.manifest
        required = {
            "schema_version", "model", "model_revision", "preproc_version",
            "preprocessing", "preproc_sha256", "n_vectors", "dim", "index_type",
            "metric", "l2_normalized", "text_field", "max_seq_length",
            "index_file", "metadata_file", "index_sha256", "metadata_sha256",
        }
        if not isinstance(manifest, dict) or not required.issubset(manifest):
            raise ValueError("Incomplete index manifest; rebuild using Lab 5 step 1")
        if (
            manifest["schema_version"] != 1
            or manifest["index_type"] != "IndexFlatIP"
            or manifest["metric"] != "cosine"
            or manifest["l2_normalized"] is not True
            or manifest["text_field"] != "case_text"
        ):
            raise ValueError("Unsupported index format or embedding contract")

        def sha256(path: Path) -> str:
            with path.open("rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()

        if (
            manifest["preproc_version"] != core.PREPROC_VERSION
            or manifest["preprocessing"] != "bayan.preprocessing.core.preprocess"
            or manifest["preproc_sha256"] != sha256(Path(core.__file__))
        ):
            raise ValueError("Preprocessing differs from the index; rebuild the index")
        revision = manifest["model_revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise ValueError("Manifest must pin an immutable model commit")
        for key in ("n_vectors", "dim", "max_seq_length"):
            if type(manifest[key]) is not int or manifest[key] <= 0:
                raise ValueError(f"Invalid manifest value: {key}")

        paths = {}
        for kind, expected in (
            ("index", Path(f"{prefix}.faiss")),
            ("metadata", Path(f"{prefix}_metadata.json")),
        ):
            if manifest[f"{kind}_file"] != expected.name:
                raise ValueError(f"Unexpected {kind} filename in manifest")
            if sha256(expected) != manifest[f"{kind}_sha256"]:
                raise ValueError(f"{kind} checksum mismatch; rebuild the index")
            paths[kind] = expected

        self.index = faiss.read_index(str(paths["index"]))
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            not isinstance(self.index, faiss.IndexFlatIP)
            or self.index.metric_type != faiss.METRIC_INNER_PRODUCT
            or self.index.ntotal != manifest["n_vectors"]
            or self.index.d != manifest["dim"]
            or not isinstance(self.metadata, list)
            or len(self.metadata) != self.index.ntotal
        ):
            raise ValueError("Index dimensions/count or metadata disagree with manifest")
        seen = set()
        for vector_id, row in enumerate(self.metadata):
            if not isinstance(row, dict):
                raise ValueError("Invalid metadata row")
            case_id = row.get("case_id")
            if (
                row.get("vector_id") != vector_id
                or not isinstance(case_id, str) or not case_id.strip()
                or case_id in seen
                or not isinstance(row.get("case_text"), str)
                or not isinstance(row.get("indexed_text"), str)
                or not row["indexed_text"]
                or row["indexed_text"] != core.preprocess(row["case_text"])
            ):
                raise ValueError(f"Invalid metadata alignment at vector {vector_id}")
            seen.add(case_id)
        # Check actual vectors as well as the manifest's normalisation flag.
        for start in range(0, self.index.ntotal, 4096):
            vectors = self.index.reconstruct_n(start, min(4096, self.index.ntotal - start))
            if not np.isfinite(vectors).all() or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6
            ):
                raise ValueError("Index contains non-finite or non-unit vectors")

        self.encoder = SentenceTransformer(
            manifest["model"], revision=revision, device=device
        )
        if self.encoder.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError("Encoder dimension does not match index")
        self.encoder.max_seq_length = manifest["max_seq_length"]
        self.reranker_model = reranker_model
        self.reranker_revision = model_info(reranker_model, revision=reranker_revision).sha
        if not self.reranker_revision:
            raise ValueError("Could not resolve the reranker revision")
        self.reranker = CrossEncoder(
            reranker_model, revision=self.reranker_revision, device=device
        )
        if self.reranker.config.num_labels != 1:
            raise ValueError("Reranker must produce a single relevance score per pair")

    def search(
        self, query: str, k: int = 5, candidates: int = 50, min_score: float = 0.25
    ) -> list[dict]:
        """Return metadata plus score and bi_encoder_score, sorted by relevance.

        ``score`` is sigmoid-transformed cross-encoder relevance, not a calibrated
        probability. Tune ``min_score`` on labelled no-answer validation queries
        in step 3. Candidate cosine scores are retained for retrieval analysis.
        """
        import faiss
        import numpy as np
        import torch

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        if type(candidates) is not int or candidates < k:
            raise ValueError("candidates must be an integer greater than or equal to k")
        if not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be finite and between 0 and 1")
        query = core.preprocess(query)
        if not query:
            return []
        vector = np.array(
            self.encoder.encode([query], convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32, order="C", copy=True,
        )
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid query embedding norm")
        faiss.normalize_L2(vector)
        similarities, ids = self.index.search(vector, min(candidates, self.index.ntotal))
        retrieved = [
            (int(vector_id), float(score))
            for vector_id, score in zip(ids[0], similarities[0])
            if vector_id >= 0
        ]
        if not retrieved:
            return []
        pairs = [(query, self.metadata[i]["indexed_text"]) for i, _ in retrieved]
        scores = np.asarray(self.reranker.predict(
            pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
            activation_fn=torch.nn.Sigmoid(),
        )).reshape(-1)
        if len(scores) != len(retrieved) or not np.isfinite(scores).all():
            raise ValueError("Reranker returned invalid scores")
        results = [
            {**self.metadata[vector_id], "score": float(score), "bi_encoder_score": cosine}
            for (vector_id, cosine), score in zip(retrieved, scores)
            if score >= min_score
        ]
        results.sort(key=lambda row: (-row["score"], -row["bi_encoder_score"], row["vector_id"]))
        return results[:k]
