"""NOT RUNNABLE AS WRITTEN (2026-10-30): its cells come from emotion2vec top-1 at p >= 0.6,
and D4 measured that label's precision on this cast at 0.08 (surprised) to 0.40 (angry), so
the "emotion" centroids would be mostly neutral clips. Revive only with human-labelled cells.

D3 - does emotion live in the x-vector, and can it be added without moving identity?

arXiv 2606.05367 (Qwen3-TTS-1.7B-Base) puts emotional prosody mainly in the x-vector:
x = x(char, neutral) + alpha * tau_emo.  Here, on our cast:
  1. Base speaker-encoder vectors for the mains' real clips whose emotion2vec top-1 is
     confident (>= 0.6) in neutral / happy / angry / sad (B05 labels).
  2. tau_emo = mean over the 6 mains of (centroid(char, emo) - centroid(char, neutral)),
     pooled so side characters with few emotional clips are not needed.
  3. Compare |tau_emo| with the distance from each character's row to its nearest other
     row (raw and centred): an offset as large as a neighbour gap would move identity.
  4. Optional --generate: Base with x = row + alpha*tau on the 30 neutral probes, alpha in
     {0, 0.5, 1, 1.5}, for the 8 D2 characters; score with eval/judge.py (identity + e2v).

    python eval/d3_xvec_emotion.py [--generate]
Output: <WORK>/d3/stats.json, <WORK>/d3/<emo>_a<alpha>/<speaker>_<idx>.wav, d3/index.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MAINS = ["moxiaoju", "wenzhi", "luoyan", "chiguo", "chiyao", "yexin"]
EMOS = ["happy", "angry", "sad"]
CONF = 0.6
PER_CELL = 80


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from xvectors import speaker_embedding
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()
    out = C.WORK / "d3"
    out.mkdir(parents=True, exist_ok=True)

    emo = {}
    for l in (C.WORK / "b05_emotion.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("e2v"):
            emo[r["id"]] = r["e2v"]
    cells = defaultdict(list)
    for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        if not d.get("keep") or d["speaker"] not in MAINS or d["id"] not in emo or not 1.5 <= d["core_dur"] <= 8:
            continue
        sc = emo[d["id"]]
        lab = max(sc, key=sc.get)
        if lab in EMOS + ["neutral"] and sc[lab] >= CONF:
            cells[(d["speaker"], lab)].append((C.stable_unit("d3:" + d["id"]), C.WORK / d["audio"]))

    w = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=True),
                                      dtype=torch.bfloat16, device_map="cuda:0")
    cent, counts = {}, {}
    for (spk, lab), items in cells.items():
        paths = [p for _, p in sorted(items)[:PER_CELL]]
        counts[f"{spk}:{lab}"] = len(paths)
        if len(paths) >= 10:
            cent[(spk, lab)] = speaker_embedding(w.model, paths, "cuda:0", torch.bfloat16)[0]
    tau = {}
    for e in EMOS:
        diffs = [cent[(s, e)] - cent[(s, "neutral")] for s in MAINS if (s, e) in cent and (s, "neutral") in cent]
        if diffs:
            tau[e] = torch.stack(diffs).mean(0)
    rows = torch.load(C.WORK / "xvec_rows.pt")
    table = rows["table"]
    mu = table.mean(0)
    stats = {"cell_counts": counts, "tau_norm": {e: round(float(t.norm()), 3) for e, t in tau.items()},
             "tau_cos_between": {f"{a}:{b}": round(float(torch.nn.functional.cosine_similarity(tau[a], tau[b], 0)), 3)
                                 for i, a in enumerate(tau) for b in list(tau)[i + 1:]},
             "per_character": {}}
    for i, s in enumerate(rows["names"]):
        d_raw = min(float((table[i] - table[j]).norm()) for j in range(len(table)) if j != i)
        stats["per_character"][s] = {"row_norm": round(float(table[i].norm()), 3),
                                     "nearest_row_dist": round(d_raw, 3),
                                     **{f"tau_{e}/nearest": round(float(tau[e].norm()) / d_raw, 3) for e in tau}}
    # does each character's own emotion offset agree with the pooled one?
    stats["own_vs_pooled_cos"] = {f"{s}:{e}": round(float(torch.nn.functional.cosine_similarity(
        cent[(s, e)] - cent[(s, "neutral")], tau[e], 0)), 3)
        for s in MAINS for e in tau if (s, e) in cent and (s, "neutral") in cent}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save({"tau": tau}, out / "tau.pt")
    print(json.dumps(stats, ensure_ascii=False, indent=1))

    if args.generate:
        import soundfile as sf
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        texts = [t.strip() for t in (Path(__file__).parent / "probes_neutral_zh.txt").read_text(encoding="utf-8").splitlines() if t.strip()]
        chars = MAINS + ["wenzhou", "zhute"]
        vec = {n: table[i] for i, n in enumerate(rows["names"])}
        with (out / "index.jsonl").open("w", encoding="utf-8") as f:
            for e in EMOS:
                for alpha in (0.5, 1.0, 1.5):
                    cond = f"{e}_a{alpha}"
                    (out / cond).mkdir(exist_ok=True)
                    for spk in chars:
                        x = (vec[spk] + alpha * tau[e]).to("cuda:0").to(torch.bfloat16)
                        item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=x, x_vector_only_mode=True, icl_mode=False)
                        torch.manual_seed(1000 + C.ROSTER.index(spk))
                        wavs, sr = w.generate_voice_clone(text=texts, language=["Chinese"] * len(texts),
                                                          voice_clone_prompt=[item] * len(texts),
                                                          max_new_tokens=400, non_streaming_mode=True)
                        for i, (t, wav) in enumerate(zip(texts, wavs)):
                            p = out / cond / f"{spk}_{i:02d}.wav"
                            sf.write(p, np.asarray(wav, np.float32), sr, subtype="PCM_16")
                            dur = len(wav) / sr
                            f.write(json.dumps({"cond": cond, "emotion": e, "alpha": alpha, "speaker": spk, "idx": i,
                                                "audio": str(p.relative_to(C.WORK)).replace("\\", "/"), "dur": round(dur, 3),
                                                "runaway": bool(dur * 12.5 >= 399 or dur > 5 * C.cjk_count(t) / 4.5)},
                                               ensure_ascii=False) + "\n")
                    print(f"[d3] {cond}", flush=True)


if __name__ == "__main__":
    main()
