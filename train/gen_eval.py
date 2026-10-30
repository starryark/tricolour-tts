"""Generate the T1 evaluation suites for one or more checkpoints (research/T1_preregistration.md,
Amendment 1).

  S1  held-out test-split lines (<= 20 per character), no instruction
  S2  the 30 neutral probes x every character, no instruction
  S3  S2 texts x {开心, 生气, 伤心} instructions (only when the checkpoint has CustomVoice presets)
language="Chinese", default sampling, non-streaming, max_new_tokens 400. Items are packed
across characters (and instructions) into batches of --batch in a fixed order; each batch
is seeded from (group, batch index), so every checkpoint sees identical batches and seeds.
Checkpoints are processed in the order given; resumable per batch.

    python gen_eval.py --suites suites.json --ckpt <dir> [--ckpt <dir> ...] --out <dir>
Output: <out>/<run>/<ckpt>/{S1,S2,S3}/<speaker>_<emotion>_<text id>.flac, index.jsonl, DONE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

EMOTIONS = {"happy": "用开心的语气说", "angry": "用生气的语气说", "sad": "用伤心的语气说"}
KW = dict(max_new_tokens=400, non_streaming_mode=True)


def seed_of(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def items_for(suites: dict, has_instruct: bool) -> dict[str, list[dict]]:
    plain, instr = [], []
    for spk, s1 in suites["S1"].items():
        plain += [{"suite": "S1", "speaker": spk, "emotion": "", "text_id": x["id"], "text": x["text"]} for x in s1]
        plain += [{"suite": "S2", "speaker": spk, "emotion": "", "text_id": f"p{i:02d}", "text": t}
                  for i, t in enumerate(suites["S2"])]
        if has_instruct:
            for emo in EMOTIONS:
                instr += [{"suite": "S3", "speaker": spk, "emotion": emo, "text_id": f"p{i:02d}", "text": t}
                          for i, t in enumerate(suites["S2"])]
    return {"plain": plain, "instruct": instr} if has_instruct else {"plain": plain}


def main() -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", required=True, help="json {S1: {speaker: [{id, text}]}, S2: [text]}")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="items per group (smoke tests)")
    args = ap.parse_args()
    suites = json.loads(Path(args.suites).read_text(encoding="utf-8"))

    for ck in map(Path, args.ckpt):
        out = Path(args.out) / ck.parent.name / ck.name
        if (out / "DONE").exists():
            print(f"[gen] {ck.parent.name}/{ck.name} already done", flush=True)
            continue
        idx_path = out / "index.jsonl"
        done = set()
        if idx_path.exists():
            done = {(r["group"], r["batch"]) for r in map(json.loads, idx_path.read_text(encoding="utf-8").splitlines())}
        cfg = json.loads((ck / "config.json").read_text(encoding="utf-8"))["talker_config"]
        groups = items_for(suites, "vivian" in cfg["spk_id"])      # S3 only for CustomVoice-derived
        w = Qwen3TTSModel.from_pretrained(str(ck), dtype=torch.bfloat16, device_map="cuda:0",
                                          attn_implementation="sdpa")
        for group, items in groups.items():
            if args.limit:
                items = items[:args.limit]
            for b in range(0, len(items), args.batch):
                if (group, b) in done:
                    continue
                chunk = items[b:b + args.batch]
                t0 = time.time()
                torch.manual_seed(seed_of(group, str(b)))
                wavs, sr = w.generate_custom_voice(
                    text=[x["text"] for x in chunk], speaker=[x["speaker"] for x in chunk],
                    language=["Chinese"] * len(chunk),
                    instruct=[EMOTIONS[x["emotion"]] for x in chunk] if group == "instruct" else None, **KW)
                rows = []
                for x, wav in zip(chunk, wavs):
                    (out / x["suite"]).mkdir(parents=True, exist_ok=True)
                    p = out / x["suite"] / f"{x['speaker']}_{x['emotion'] or 'none'}_{x['text_id']}.flac"
                    sf.write(p, np.asarray(wav, dtype=np.float32), sr, format="FLAC")
                    dur = len(wav) / sr
                    rows.append({**x, "group": group, "batch": b,
                                 "audio": str(p.relative_to(out)).replace("\\", "/"),
                                 "dur": round(dur, 3), "frames": int(round(dur * 12.5))})
                with idx_path.open("a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                dt = time.time() - t0
                print(f"[gen] {ck.parent.name}/{ck.name} {group} {b + len(chunk)}/{len(items)}  "
                      f"{len(chunk)} clips in {dt:.0f}s ({len(chunk) / dt:.1f} clip/s, max {max(r['frames'] for r in rows)} frames)",
                      flush=True)
        del w
        torch.cuda.empty_cache()
        (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
