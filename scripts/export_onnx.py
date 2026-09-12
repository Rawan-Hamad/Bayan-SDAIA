"""Lab 7 Step 2: fp32 ONNX export and paired validation; original weights retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = ROOT / "artifacts/topic_classifier"
OUTPUT = ROOT / "artifacts/onnx/topic_classifier.onnx"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def configure(threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def session(path, threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def export(threads):
    import numpy as np
    import torch
    import onnx
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    before = digest(MODEL / "pytorch_model.bin")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, attn_implementation="eager").float().cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    class Logits(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.base(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids).logits

    wrapper = Logits(model).eval()
    names = ["input_ids", "attention_mask", "token_type_ids"]
    sample = tokenizer("The park needs maintenance.", return_tensors="pt")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_name("topic_classifier.pending.onnx")
    with torch.inference_mode():
        torch.onnx.export(wrapper, tuple(sample[n] for n in names), str(temp),
                          input_names=names, output_names=["logits"], opset_version=17,
                          dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names},
                                        "logits": {0: "batch"}},
                          dynamo=False, external_data=False)
    onnx.checker.check_model(str(temp))
    runtime = session(temp, threads)
    max_error = 0.0
    # Exercise exported axes and masks, not just the tracing example.
    for batch, length in ((1, 8), (2, 32), (1, 128), (1, 512)):
        ids = torch.full((batch, length), 100, dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[:, length // 2:] = 0
        types = torch.zeros_like(ids)
        with torch.inference_mode():
            reference = wrapper(ids, mask, types).numpy()
        actual = runtime.run(None, dict(zip(names, [ids.numpy(), mask.numpy(), types.numpy()])))[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
        max_error = max(max_error, float(np.abs(actual-reference).max()))
    del runtime
    assert digest(MODEL / "pytorch_model.bin") == before
    temp.replace(OUTPUT)
    manifest = {"format": "ONNX fp32", "opset": 17, "threads": threads,
                "weights_sha256": before, "onnx_sha256": digest(OUTPUT),
                "rollback": "artifacts/topic_classifier (original model and tokenizer retained)",
                "onnx_bytes": OUTPUT.stat().st_size, "dynamic_axes_checked": [[1,8],[2,32],[1,128],[1,512]],
                "max_smoke_logit_error": max_error}
    print("Export and dynamic shape checks passed; rollback weights unchanged.", flush=True)
    return manifest


def quality(manifest, threads):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from bayan.models.data import build_topic_dataset, TOPICS
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True).float().cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert [model.config.id2label[i] for i in range(len(TOPICS))] == TOPICS
    runtime = session(OUTPUT, threads)
    data = build_topic_dataset(str(ROOT / "data/raw/bayan_feedback.csv"), seed=42)["validation"]
    records = []
    max_error = 0.0
    for i, row in enumerate(data, 1):
        # Dataset builder applies the shared preprocessing once.
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=128)
        with torch.inference_mode():
            reference = model(**inputs).logits.numpy()
        actual = runtime.run(None, {k: v.numpy() for k,v in inputs.items()})[0]
        np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-4)
        max_error = max(max_error, float(np.abs(actual-reference).max()))
        records.append({"feedback_id":row["feedback_id"], "y_true":int(row["label"]),
                        "fp32":int(reference.argmax()), "onnx":int(actual.argmax())})
        if i % 200 == 0:
            print(f"Paired validation {i}/{len(data)}", flush=True)
    y = np.array([r["y_true"] for r in records])
    a = np.array([r["fp32"] for r in records])
    b = np.array([r["onnx"] for r in records])
    def f1(truth, pred):
        matrix = np.bincount(truth*len(TOPICS)+pred, minlength=len(TOPICS)**2).reshape(len(TOPICS),len(TOPICS))
        den = matrix.sum(0)+matrix.sum(1)
        return float(np.divide(2*matrix.diagonal(),den,out=np.zeros(len(TOPICS)),where=den!=0).mean())
    tax = f1(y,a)-f1(y,b)
    rng=np.random.default_rng(42)
    taxes=[]
    for _ in range(2000):
        indices=rng.integers(0,len(y),len(y))
        taxes.append(f1(y[indices],a[indices])-f1(y[indices],b[indices]))
    manifest["quality"]={"n":len(y), "split":"build_topic_dataset(seed=42).validation; frozen test not evaluated",
                         "max_length":128, "fp32_macro_f1":f1(y,a), "onnx_macro_f1":f1(y,b),
                         "fp32_accuracy":float((y==a).mean()), "onnx_accuracy":float((y==b).mean()),
                         "prediction_disagreements":int((a!=b).sum()), "max_logit_error":max_error,
                         "quality_tax":tax, "tax_ci95":np.quantile(taxes,[.025,.975]).tolist(),
                         "ci_method":"2000 paired row bootstrap resamples, seed 42; independent rows assumed",
                         "records":records}
    manifest["data_sha256"]=digest(ROOT / "data/raw/bayan_feedback.csv")
    (ROOT / "data/serving/onnx_export_report.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Paired validation saved.",flush=True)


def benchmark_onnx(threads, warmup=20):
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from bayan.preprocessing.core import preprocess
    from benchmark_inference import benchmark
    report=json.loads((ROOT/"data/serving/onnx_export_report.json").read_text())
    baseline=json.loads((ROOT/"data/serving/baseline_benchmark.json").read_text())
    assert digest(OUTPUT)==report["onnx_sha256"]
    assert digest(MODEL/"pytorch_model.bin")==report["weights_sha256"]==baseline["weights_sha256"]
    mix=ROOT/"data/serving/bench_mix.npy"
    assert digest(mix)==baseline["mix_sha256"] and threads==baseline["threads"]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    runtime=session(OUTPUT,threads)
    texts=np.load(mix,allow_pickle=False)
    inputs=[{k: np.asarray(v, dtype=np.int64) for k,v in
             tokenizer(preprocess(str(t)),return_tensors="np",truncation=True,max_length=128).items()} for t in texts]
    timing=benchmark(lambda x:runtime.run(None,x),inputs,warmup=warmup)
    timing.update(threads=threads,interop_threads=1,batch_size=1,warmup=warmup,
                  runtime=ort.__version__,device="CPUExecutionProvider",onnx_sha256=report["onnx_sha256"],
                  scope="forward only; tokenization/preprocessing excluded",
                  mix_sha256=digest(mix))
    timing["speedup_p99_vs_512"]=baseline["rungs"]["fp32 torch @512 padded"]["p99_ms"]/timing["p99_ms"]
    timing["speedup_p99_vs_dynamic"]=baseline["rungs"]["fp32 torch @128 dynamic"]["p99_ms"]/timing["p99_ms"]
    (ROOT/"data/serving/onnx_benchmark.json").write_text(json.dumps(timing,indent=2),encoding="utf-8")
    q=report["quality"]
    bp=ROOT/"BENCHMARKS.md"
    lines=bp.read_text(encoding="utf-8").splitlines()
    for i,line in enumerate(lines):
        if line.startswith("| ONNX fp32 @128 |"):
            lines[i]=f"| ONNX fp32 @128 | {timing['p50_ms']:.2f} ms | {timing['p99_ms']:.2f} ms | F1 {q['onnx_macro_f1']:.4f}; tax {q['quality_tax']*100:.4f} pp | {OUTPUT.stat().st_size/1024**2:.2f} MiB |"
    heading="### Lab 7 - Step 2: ONNX fp32"
    text="\n".join(lines)+"\n"
    if heading in text:
        start=text.index(heading); end=text.find("\n##",start+len(heading))
        text=text[:start]+(text[end:] if end!=-1 else "")
    text+=f"""
{heading}

CPU, {threads} intra-op threads, 1 inter-op thread, batch 1; {len(texts)} texts, {warmup} warm-up requests.
p99 speed-up vs saved Step 1: {timing['speedup_p99_vs_512']:.2f}x vs 512; {timing['speedup_p99_vs_dynamic']:.2f}x vs dynamic/128.
Timings exclude preprocessing, tokenization and HTTP. Baseline is a previous run; CPU conditions may differ.
Paired validation: {q['n']} rows; fp32 macro-F1={q['fp32_macro_f1']:.4f}, ONNX={q['onnx_macro_f1']:.4f}.
Tax (fp32 minus ONNX): {q['quality_tax']*100:.4f} percentage points; 95% CI [{q['tax_ci95'][0]*100:.4f}, {q['tax_ci95'][1]*100:.4f}].
Prediction disagreements: {q['prediction_disagreements']}; maximum logit difference: {q['max_logit_error']:.8f}.
CI uses paired rows (2000 resamples, seed 42); citizen-group dependence is not modelled.
Original fp32 weights retained unchanged in artifacts/topic_classifier.
[Export and paired predictions](data/serving/onnx_export_report.json) | [Raw timings](data/serving/onnx_benchmark.json).

Re-run: `python scripts/export_onnx.py --threads 4`, then `python scripts/benchmark_inference.py --onnx --threads 4`.
"""
    bp.write_text(text,encoding="utf-8")
    print(json.dumps({k:v for k,v in timing.items() if k!="latencies_ms"},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    configure(args.threads)
    quality(export(args.threads),args.threads)


if __name__=="__main__":
    main()
