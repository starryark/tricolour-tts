"""B05 - audio emotion signals for every kept clip.

Labels must describe how a line was *delivered*, not what its words imply: an emotion
control trained on text-derived labels is redundant with the text and cannot override it.
Text may tag a clip (audio says angry, text reads neutral = an override-teaching example)
but never gates it.
  * emotion2vec+ large (9 classes, utterance scores): the audio label.
  * SenseVoiceSmall's emotion tag: a diagnostic only. Its SER labels were derived from
    emotion2vec+, so it is NOT an independent second vote.
TRI's sprite face codes (b03) are the independent, author-given signal. The emotion2vec
labels' accuracy on these clips is measured against human labels (D4) before any is used.

Needs funasr (e.g. the `.venv-curation` env). Resumable: <WORK>/b05_emotion.jsonl.

    python build/b05_emotion_audio.py [--only e2v|sv] [--limit N]
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from b02_asr import batches, load, save  # noqa: E402

SV_TAG = re.compile(r"<\|([A-Z_]+)\|>")
SV_EMO = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["e2v", "sv"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r.get("keep")]
    rows.sort(key=lambda r: -r["dur"])
    if args.limit:
        rows = rows[:args.limit]
    out = C.WORK / "b05_emotion.jsonl"
    done = load(out)

    from funasr import AutoModel
    if args.only in (None, "e2v"):
        todo = [r for r in rows if done.get(r["id"], {}).get("e2v") is None]
        if todo:
            model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True, device="cuda:0")
            for i, chunk in enumerate(batches(todo, 32)):
                res = model.generate([str(C.WORK / r["audio"]) for r in chunk], granularity="utterance",
                                     extract_embedding=False, batch_size=32)
                for r, h in zip(chunk, res):
                    labels = [l.split("/")[-1] for l in h["labels"]]
                    done.setdefault(r["id"], {"id": r["id"]})["e2v"] = {
                        k: round(float(s), 4) for k, s in zip(labels, h["scores"])}
                if i % 50 == 0:
                    print(f"[emotion2vec] {min((i + 1) * 32, len(todo))}/{len(todo)}", flush=True)
                    save(done, out)
            save(done, out)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if args.only in (None, "sv"):
        todo = [r for r in rows if done.get(r["id"], {}).get("sv") is None]
        if todo:
            model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device="cuda:0")
            for i, chunk in enumerate(batches(todo, 64)):
                res = model.generate([str(C.WORK / r["audio"]) for r in chunk], language="zh",
                                     use_itn=False, batch_size=64)
                for r, h in zip(chunk, res):
                    tags = SV_TAG.findall(h["text"])
                    emo = next((t for t in tags if t in SV_EMO), None)
                    event = next((t for t in tags if t not in SV_EMO and t.isupper()
                                  and t not in ("ZH", "EN", "JA", "KO", "YUE", "NOSPEECH", "WITHITN", "WOITN")), None)
                    done.setdefault(r["id"], {"id": r["id"]})["sv"] = {"emo": emo, "event": event}
                if i % 50 == 0:
                    print(f"[sensevoice] {min((i + 1) * 64, len(todo))}/{len(todo)}", flush=True)
                    save(done, out)
            save(done, out)
    print(f"[b05] {len(done)} clips -> {out}")


if __name__ == "__main__":
    main()
