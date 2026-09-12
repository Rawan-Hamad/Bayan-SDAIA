"""Lab 7 Step 1: CPU fp32 latency, batch size one, using the supplied text mix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def benchmark(predict, inputs, *, warmup=20):
    """Time only predict(input); preparation and warm-up are excluded."""
    import numpy as np
    if not inputs or warmup < 1:
        raise ValueError("Inputs and positive warmup required")
    for i in range(warmup):
        predict(inputs[i % len(inputs)])
    times = []
    for i, item in enumerate(inputs, 1):
        start = perf_counter()
        predict(item)
        times.append((perf_counter() - start) * 1000)
        if i % 200 == 0:
            print(f"Measured {i}/{len(inputs)}", flush=True)
    return {"n": len(times), "p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99)), "latencies_ms": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--onnx", action="store_true", help="Benchmark exported ONNX using saved Step 1 baseline")
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 1:
        parser.error("threads and warmup must be positive")
    if args.onnx:
        from export_onnx import configure, benchmark_onnx
        configure(args.threads)
        benchmark_onnx(args.threads, args.warmup)
        return
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from bayan.preprocessing.core import preprocess, PREPROC_VERSION
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    path = ROOT / "artifacts/topic_classifier"
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).float().cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mix_path = ROOT / "data/serving/bench_mix.npy"
    texts = np.load(mix_path, allow_pickle=False)
    if texts.ndim != 1 or texts.dtype.kind != "U" or not len(texts):
        raise ValueError("Expected a nonempty array of text")
    texts = [preprocess(str(text)) for text in texts]
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts]
    report = {"device": "cpu", "dtype": "float32", "batch_size": 1,
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "warmup_per_rung": args.warmup, "samples_per_rung": len(texts),
              "timing_scope": "model forward only; preprocessing, tokenization and HTTP excluded",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "cpu": platform.processor(), "preprocessing": PREPROC_VERSION,
              "mix_sha256": hashlib.sha256(mix_path.read_bytes()).hexdigest(),
              "token_lengths": {"min": min(lengths), "max": max(lengths),
                               "p50": float(np.percentile(lengths,50)), "p95": float(np.percentile(lengths,95))},
              "quality": "Not measured in Step 1", "rungs": {}}
    with (path / "pytorch_model.bin").open("rb") as f:
        report["weights_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    with torch.inference_mode():
        for name, maximum, padding in (("fp32 torch @512 padded", 512, "max_length"),
                                        ("fp32 torch @128 dynamic", 128, True)):
            print(name, flush=True)
            inputs = [tokenizer(t, return_tensors="pt", max_length=maximum,
                                truncation=True, padding=padding) for t in texts]
            result = benchmark(lambda item: model(**item), inputs, warmup=args.warmup)
            result["truncated_texts"] = sum(n > maximum for n in lengths)
            result["input_length_min"] = min(x["input_ids"].shape[1] for x in inputs)
            result["input_length_max"] = max(x["input_ids"].shape[1] for x in inputs)
            report["rungs"][name] = result
    base, dynamic = report["rungs"].values()
    report["speedup_p50"] = base["p50_ms"] / dynamic["p50_ms"]
    report["speedup_p99"] = base["p99_ms"] / dynamic["p99_ms"]
    out = ROOT / "data/serving/baseline_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    bp = ROOT / "BENCHMARKS.md"
    lines = bp.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        for name, values in report["rungs"].items():
            if line.startswith("| " + name + " |"):
                size = (path / "pytorch_model.bin").stat().st_size / (1024 ** 2)
                lines[i] = f"| {name} | {values['p50_ms']:.2f} ms | {values['p99_ms']:.2f} ms | Not measured yet | {size:.2f} MiB |"
    heading = "### Lab 7 - Step 1: CPU baseline"
    text = "\n".join(lines) + "\n"
    if heading in text:
        start = text.index(heading)
        end = text.find("\n##", start + len(heading))
        text = text[:start] + (text[end:] if end != -1 else "")
    text += (f"\n{heading}\n\n"
             f"CPU fp32, batch=1, threads={args.threads}, warm-up={args.warmup} per rung; all {len(texts)} supplied texts.\n"
             "Model forward time only; preprocessing, tokenization and HTTP excluded.\n"
             "At batch=1, dynamic padding uses each text's actual token length.\n"
             f"Dynamic/128 speed-up: p50 {report['speedup_p50']:.2f}x; p99 {report['speedup_p99']:.2f}x.\n"
             f"Texts truncated at 128: {dynamic['truncated_texts']}. Quality change: not measured yet.\n"
             "Single sequential run; CPU contention and run order can affect timing.\n"
             "[Raw timings and environment](data/serving/baseline_benchmark.json).\n\n"
             "Re-run: `python scripts/benchmark_inference.py --threads 4 --warmup 20`\n")
    bp.write_text(text, encoding="utf-8")
    print(json.dumps({k: {m:v[m] for m in ('p50_ms','p99_ms','truncated_texts')}
                      for k,v in report["rungs"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
