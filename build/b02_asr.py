"""B02 - transcribe every kept clip with two unrelated ASR families.

Qwen3-ASR-1.7B (Qwen family, forced Chinese, no context prompt: on Kurisu a context prompt
made it echo the prompt on 163 clips) and Paraformer-zh (FunASR family, the Seed-TTS-eval
zh scorer). Comparing in pinyin space happens in b03; this step only writes hypotheses.

Needs qwen_asr (pins transformers 4.57.6) and funasr, e.g. the `.venv-curation` env.
Resumable: <WORK>/b02_asr.jsonl is extended, never rewritten from scratch.

    python build/b02_asr.py [--only qwen|para] [--limit N]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def load(out: Path) -> dict:
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.setdefault(r["id"], {}).update(r)
    return done


def save(done: dict, out: Path) -> None:
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["qwen", "para"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qbatch", type=int, default=8,
                    help="Qwen3-ASR batch; 32 oversubscribed a 16 GB card (WDDM spilled to shared memory)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r.get("keep")]
    rows.sort(key=lambda r: -r["dur"])          # bounded padding per batch
    if args.limit:
        rows = rows[:args.limit]
    out = C.WORK / "b02_asr.jsonl"
    done = load(out)

    if args.only in (None, "qwen"):
        todo = [r for r in rows if done.get(r["id"], {}).get("qwen") is None]
        if todo:
            from huggingface_hub import snapshot_download
            from qwen_asr import Qwen3ASRModel
            model = Qwen3ASRModel.from_pretrained(
                snapshot_download("Qwen/Qwen3-ASR-1.7B"), dtype=torch.bfloat16, device_map="cuda:0",
                attn_implementation="sdpa", max_inference_batch_size=args.qbatch, max_new_tokens=256)
            for i, chunk in enumerate(batches(todo, args.qbatch)):
                res = model.transcribe([str(C.WORK / r["audio"]) for r in chunk], language="Chinese")
                for r, h in zip(chunk, res):
                    done.setdefault(r["id"], {"id": r["id"]})["qwen"] = h.text.strip()
                if i % 100 == 0:
                    print(f"[qwen-asr] {min((i + 1) * args.qbatch, len(todo))}/{len(todo)}", flush=True)
                    save(done, out)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            save(done, out)

    if args.only in (None, "para"):
        todo = [r for r in rows if done.get(r["id"], {}).get("para") is None]
        if todo:
            from funasr import AutoModel
            model = AutoModel(model="paraformer-zh", disable_update=True, device="cuda:0")
            for i, chunk in enumerate(batches(todo, 64)):
                res = model.generate(input=[str(C.WORK / r["audio"]) for r in chunk], batch_size=64)
                for r, h in zip(chunk, res):
                    done.setdefault(r["id"], {"id": r["id"]})["para"] = h["text"].replace(" ", "").strip()
                if i % 25 == 0:
                    print(f"[paraformer] {min((i + 1) * 64, len(todo))}/{len(todo)}", flush=True)
                    save(done, out)
            save(done, out)
    print(f"[b02] {len(done)} clips with hypotheses -> {out}")


if __name__ == "__main__":
    main()
