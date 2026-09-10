"""Lab 5 step 3: labelled retrieval evaluation (run explicitly in Colab)."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


def load_queries(path, metadata):
    cases = {row["case_id"]: row for row in metadata}
    queries = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not {
                "query_id", "query", "lang", "relevant_case_ids", "no_answer"
            }.issubset(row):
                raise ValueError(f"Missing query fields at line {line_number}")
            if not isinstance(row["query_id"], str) or not row["query_id"] or row["query_id"] in seen:
                raise ValueError(f"Invalid or duplicate query ID at line {line_number}")
            if not isinstance(row["query"], str) or not preprocess(row["query"]):
                raise ValueError(f"Empty query at line {line_number}")
            if not isinstance(row["lang"], str) or not row["lang"].strip():
                raise ValueError(f"Missing query language at line {line_number}")
            relevant = row["relevant_case_ids"]
            if not isinstance(relevant, list) or not all(isinstance(x, str) for x in relevant):
                raise ValueError(f"Invalid relevance labels at line {line_number}")
            if type(row["no_answer"]) is not bool or row["no_answer"] != (len(relevant) == 0):
                raise ValueError(f"Inconsistent no-answer labels at line {line_number}")
            missing = set(relevant) - cases.keys()
            if missing:
                raise ValueError(f"Relevant cases missing from index: {sorted(missing)}; build the full index")
            row["lang"] = row["lang"].strip().lower()
            if relevant:
                langs = {str(cases[x].get("lang", "")).strip().lower() for x in relevant}
                if "" in langs:
                    raise ValueError("Relevant case metadata is missing language")
                row["language_slice"] = (
                    "same_language" if langs == {row["lang"]}
                    else "cross_language" if row["lang"] not in langs else "mixed_language"
                )
            else:
                row["language_slice"] = "no_answer"
            seen.add(row["query_id"])
            queries.append(row)
    if not queries:
        raise ValueError("No labelled queries found")
    return queries


def metrics(rows, stage):
    answerable = [row for row in rows if not row["no_answer"]]
    if not answerable:
        return {"n": 0, "recall_at_10": None, "mrr_at_10": None}
    recalls, reciprocal_ranks = [], []
    for row in answerable:
        relevant = set(row["relevant_case_ids"])
        ids = row[stage][:10]
        recalls.append(len(set(ids) & relevant) / len(relevant))
        reciprocal_ranks.append(next((1 / rank for rank, cid in enumerate(ids, 1) if cid in relevant), 0.0))
    return {"n": len(answerable), "recall_at_10": statistics.mean(recalls),
            "mrr_at_10": statistics.mean(reciprocal_ranks)}


def threshold_metrics(rows, threshold):
    nulls = [r for r in rows if r["no_answer"]]
    answers = [r for r in rows if not r["no_answer"]]
    empty_correct = sum(not r["scores"] or r["scores"][0] < threshold for r in nulls)
    false_empty = sum(not r["scores"] or r["scores"][0] < threshold for r in answers)
    filtered = [{**r, "filtered": [cid for cid, score in zip(r["reranked"], r["scores"])
                                    if score >= threshold]} for r in rows]
    return {"threshold": threshold, "no_answer_n": len(nulls), "empty_correct": empty_correct,
            "empty_correct_rate": empty_correct / len(nulls) if nulls else None,
            "answerable_n": len(answers), "answerable_false_empty": false_empty,
            "answerable_false_empty_rate": false_empty / len(answers) if answers else None,
            "retrieval_after_threshold": metrics(filtered, "filtered")}


def evaluate(search, queries, candidates, min_score):
    import faiss
    import numpy as np
    import torch

    def synchronize():
        if search.encoder.device.type == "cuda" or search.reranker.model.device.type == "cuda":
            torch.cuda.synchronize()

    # Warm up models; loading/downloads and this warm-up are excluded from timings.
    search.search(queries[0]["query"], k=10, candidates=candidates, min_score=0.0)
    rows = []
    for query in queries:
        synchronize()
        start = perf_counter()
        text = preprocess(query["query"])
        vector = np.array(search.encoder.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        ), dtype=np.float32, order="C", copy=True)
        if vector.shape != (1, search.index.d) or not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid query embedding norm")
        faiss.normalize_L2(vector)
        similarities, ids = search.index.search(vector, min(candidates, search.index.ntotal))
        retrieved = [(int(i), float(s)) for i, s in zip(ids[0], similarities[0]) if i >= 0]
        synchronize()
        bi_end = perf_counter()
        pairs = [(text, search.metadata[i]["indexed_text"]) for i, _ in retrieved]
        scores = np.asarray(search.reranker.predict(
            pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
            activation_fn=torch.nn.Sigmoid(),
        )).reshape(-1)
        if len(scores) != len(retrieved) or not np.isfinite(scores).all():
            raise ValueError("Invalid reranker scores")
        ranked = sorted(zip(retrieved, scores), key=lambda pair: (-float(pair[1]), -pair[0][1], pair[0][0]))
        synchronize()
        end = perf_counter()
        rows.append({**query,
                     "bi_encoder": [search.metadata[i]["case_id"] for i, _ in retrieved],
                     "reranked": [search.metadata[i]["case_id"] for (i, _), _score in ranked],
                     "scores": [float(score) for _, score in ranked],
                     "bi_encoder_ms": (bi_end - start) * 1000,
                     "reranker_ms": (end - bi_end) * 1000,
                     "two_stage_ms": (end - start) * 1000})
    groups = {"all": rows}
    groups.update({f"query_lang={lang}": [r for r in rows if r["lang"] == lang]
                   for lang in sorted({r["lang"] for r in rows})})
    groups.update({name: [r for r in rows if r["language_slice"] == name]
                   for name in ("same_language", "cross_language", "mixed_language")})
    slices = {name: {stage: metrics(group, stage) for stage in ("bi_encoder", "reranked")}
              for name, group in groups.items()}
    gaps = {}
    for stage in ("bi_encoder", "reranked"):
        same, cross = slices["same_language"][stage], slices["cross_language"][stage]
        gaps[stage] = {metric: same[metric] - cross[metric]
                       if same[metric] is not None and cross[metric] is not None else None
                       for metric in ("recall_at_10", "mrr_at_10")}
    latency = {stage: {"p50": float(np.percentile([r[stage] for r in rows], 50)),
                       "p95": float(np.percentile([r[stage] for r in rows], 95))}
               for stage in ("bi_encoder_ms", "reranker_ms", "two_stage_ms")}
    all_metrics = slices["all"]
    before, after = all_metrics["bi_encoder"]["mrr_at_10"], all_metrics["reranked"]["mrr_at_10"]
    return {"n_queries": len(rows), "candidates": candidates, "slices": slices,
            "mrr_lift": after - before if before is not None and after is not None else None,
            "cross_lingual_gap_same_minus_cross": gaps, "latency_ms": latency,
            "selected_threshold": threshold_metrics(rows, min_score),
            "threshold_sweep_exploratory": [threshold_metrics(rows, float(t)) for t in np.linspace(0, 1, 21)],
            "queries": rows}


def markdown_report(report):
    lines = ["# Lab 5 retrieval evaluation", "", *report["notes"], "",
             "| Slice | Stage | Answerable n | Recall@10 | MRR@10 |",
             "|---|---|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    for name, stages in report["slices"].items():
        for stage, values in stages.items():
            lines.append(f"| {name} | {stage} | {values['n']} | {fmt(values['recall_at_10'])} | {fmt(values['mrr_at_10'])} |")
    lines += ["", f"MRR lift: {fmt(report['mrr_lift'])}", "", "## Latency (milliseconds)", ""]
    for stage, values in report["latency_ms"].items():
        lines.append(f"- {stage}: p50={values['p50']:.2f}, p95={values['p95']:.2f}")
    lines += ["", "## Cross-lingual gap (same-language minus cross-language)", ""]
    for stage, values in report["cross_lingual_gap_same_minus_cross"].items():
        lines.append(f"- {stage}: Recall@10={fmt(values['recall_at_10'])}, MRR@10={fmt(values['mrr_at_10'])}")
    chosen = report["selected_threshold"]
    lines += ["", f"Selected threshold: {chosen['threshold']}; empty-correct: {chosen['empty_correct']}/{chosen['no_answer_n']}; answerable false-empty: {chosen['answerable_false_empty']}/{chosen['answerable_n']}.",
              "", "## Exploratory threshold sweep", "",
              "| Threshold | Empty-correct | Answerable false-empty | Recall@10 | MRR@10 |",
              "|---:|---:|---:|---:|---:|"]
    for item in report["threshold_sweep_exploratory"]:
        values = item["retrieval_after_threshold"]
        lines.append(f"| {item['threshold']:.2f} | {item['empty_correct']}/{item['no_answer_n']} | {item['answerable_false_empty']}/{item['answerable_n']} | {fmt(values['recall_at_10'])} | {fmt(values['mrr_at_10'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=ROOT / "artifacts/search/case_index_v1")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/search/bayan_queries.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/search/evaluation")
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.candidates < 10:
        parser.error("--candidates must be at least 10")
    if not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1:
        parser.error("--min-score must be between 0 and 1")
    search = CaseSearch(str(args.prefix), device=args.device)
    queries = load_queries(args.queries, search.metadata)
    report = evaluate(search, queries, args.candidates, args.min_score)
    report["index_manifest"] = search.manifest
    report["reranker"] = {"model": search.reranker_model, "revision": search.reranker_revision}
    report["device"] = str(search.encoder.device)
    report["query_source"] = str(args.queries.resolve())
    report["notes"] = [
        "Recall@10 is the fraction of labelled relevant cases retrieved, averaged over answerable queries; MRR@10 uses the first relevant rank.",
        "Main retrieval metrics use unfiltered rankings. No-answer queries are evaluated separately.",
        "Cross-language means all labelled relevant cases have a different language from the query; mixed-language queries are separate. Missing slices are N/A.",
        "Threshold sweep uses this same query set and is exploratory, not held-out evidence. Freeze a threshold on separate validation data before final evaluation.",
        "Latencies include preprocessing/retrieval and reranking as labelled, after warm-up; downloads and model loading are excluded.",
        "Only the two report files in output-dir are written (replaced on rerun). BENCHMARKS.md, source data and index artifacts are not modified.",
    ]
    markdown = markdown_report(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "retrieval_eval.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Reports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
