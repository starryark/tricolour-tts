"""B06 - the final per-clip manifest: decisions, transcripts, emotion signals and splits.

Drops on top of b01:
  * transcript_mismatch: both ASR families further from the line than 1-in-100 random lines
    of the same speaker (b04 `mismatch`), or best toneless-pinyin error >= 0.4 on a line of
    >= 4 syllables (unspoken narration, truncated or substituted takes).
Splits:
  * test = ~3% of 40-line script blocks (scene + script order // 40), chosen by a seeded hash;
           characters left with fewer than 8 test clips get further blocks until they have 8.
           Blocks, not whole scenes: a TRI "scene" is a whole KAG file, and holding out one
           took up to 41% of a side character's lines.
  * val  = the dataset's own scene-held-out split, capped per character at max(12, 4%) of
           that character's clips; the excess returns to train. Val only monitors training
           loss, so its scene overlap with train is harmless; test is the clean held-out set.
Emotion: emotion2vec+ scores and TRI face codes are attached as signals. No emotion label is
assigned here: emotion2vec's accuracy on this cast is unknown until the human calibration (D4).

    python build/b06_manifest.py
Output: <WORK>/manifest.jsonl, <WORK>/b06_report.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TEST_SHARE = 0.03
TEST_MIN = 8
BLOCK = 40
VAL_CAP_MIN, VAL_CAP_SHARE = 12, 0.04


def jl(path: Path) -> dict:
    out = {}
    for l in path.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r
    return out


def main() -> None:
    dec = {k: v for k, v in jl(C.WORK / "b01_decisions.jsonl").items() if v.get("keep")}
    asr, ver, emo = jl(C.WORK / "b02_asr.jsonl"), jl(C.WORK / "b04_verify.jsonl"), jl(C.WORK / "b05_emotion.jsonl")
    faces = jl(C.WORK / "b03_faces.jsonl")
    with (C.DATASET / "splits" / "val.csv").open(encoding="utf-8") as f:
        val_ids = {Path(r["path"]).stem for r in csv.DictReader(f, delimiter="|")}

    rows, dropped = [], Counter()
    for k, d in dec.items():
        v = ver[k]
        if v["mismatch"] or (v["per_best"] >= 0.4 and v["n_syl"] >= 4):
            dropped["transcript_mismatch"] += 1
            continue
        e = emo.get(k, {}).get("e2v") or {}
        fc = faces.get(k, {})
        rows.append({
            "id": k, "speaker": d["speaker"], "series": d["series"], "scene": d["scene"], "tier": d["tier"],
            "audio": d["audio"], "dur": d["dur"], "core_dur": d["core_dur"],
            "text": d["text"], "text_src": d["text_src"],
            "asr_qwen": asr[k]["qwen"], "asr_paraformer": asr[k]["para"],
            "pinyin_err_best": v["per_best"],
            "gain_db": d["gain_db"], "peak_limited": d.get("peak_limited", False),
            "e2v": e, "e2v_top": max(e, key=e.get) if e else None,
            "face": fc.get("face"), "face_brow": fc.get("brow"), "face_blush": fc.get("blush"),
            "face_tears": fc.get("tears"), "sprite_motion": fc.get("motion"),
            "split": "val" if k in val_ids else None,
        })

    order = {}
    for l in (C.DATASET / "metadata.jsonl").open(encoding="utf-8"):
        m = json.loads(l)
        order[m["clip_id"]] = m["order"]
    for r in rows:
        r["block"] = f"{r['scene']}#{order[r['id']] // BLOCK}"

    # cap val per character; the excess (seeded) returns to train
    n_spk = Counter(r["speaker"] for r in rows)
    vals = defaultdict(list)
    for r in rows:
        if r["split"] == "val":
            vals[r["speaker"]].append(r)
    for spk, vs in vals.items():
        cap = max(VAL_CAP_MIN, int(VAL_CAP_SHARE * n_spk[spk]))
        for r in sorted(vs, key=lambda r: C.stable_unit("val:" + r["id"]))[cap:]:
            r["split"] = None

    # test blocks: seeded share of the non-val blocks, then top up small characters
    blocks = sorted({r["block"] for r in rows if r["split"] is None}, key=lambda b: C.stable_unit("test:" + b))
    test_blocks = set(blocks[:max(1, int(len(blocks) * TEST_SHARE))])
    by_block = defaultdict(list)
    for r in rows:
        if r["split"] is None:
            by_block[r["block"]].append(r)
    for spk in C.ROSTER:
        have = sum(1 for b in test_blocks for r in by_block[b] if r["speaker"] == spk)
        for b in blocks:
            if have >= TEST_MIN:
                break
            if b in test_blocks or not any(r["speaker"] == spk for r in by_block[b]):
                continue
            test_blocks.add(b)
            have += sum(1 for r in by_block[b] if r["speaker"] == spk)
    # a test block must not share a scene block with val-returned or train clips: it doesn't,
    # since blocks partition clips; it may share a whole scene with train (by design, see doc)
    for r in rows:
        if r["split"] is None:
            r["split"] = "test" if r["block"] in test_blocks else "train"

    rows.sort(key=lambda r: (C.ROSTER.index(r["speaker"]), r["id"]))
    with (C.WORK / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = defaultdict(lambda: Counter())
    mins = defaultdict(float)
    for r in rows:
        per[r["speaker"]][r["split"]] += 1
        mins[r["speaker"]] += r["dur"] / 60
    rep = {"clips": len(rows), "hours": round(sum(r["dur"] for r in rows) / 3600, 3), "dropped": dict(dropped),
           "splits": dict(Counter(r["split"] for r in rows)), "test_blocks": len(test_blocks),
           "per_speaker": {s: {**per[s], "minutes": round(mins[s], 1)} for s in C.ROSTER}}
    (C.WORK / "b06_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
