"""B09 - training rows, effect exclusions and screened reference clips (report s15.3-15.4).

Effect flags (any of):
  * the speech-only band-limit or cepstral-echo detector (research/notes/fx_pilot_feats.csv:
    `nb`, `echo`), or the speaker-relative LTAS phone test (b07b: both edges < -10 dB);
  * a human effect label (tts_work/fx/labels_final, 100-clip blind check).
Flags propagate to the whole 40-line script block of that speaker when >= 30% of the
speaker's lines in the block are flagged (effects run scene by scene).
Training and val rows drop every flagged clip. Reference clips (the locked speaker vector)
are stricter: they also drop the `tail` and `basscut` flags, cost-free since false positives
only shrink a pool of hundreds.

    python build/b09_train_sets.py
Output: <WORK>/qwen/{train,val}_rows.jsonl (trainer rows, no codes yet),
        <WORK>/refs/<speaker>.txt (60 reference paths each), <WORK>/b09_report.json
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NOTES = Path(r"F:\Tricolour\research\notes")
BLOCK_SHARE = 0.30
N_REF = 60


def main() -> None:
    man = [json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")]
    fx = {r["clip_id"]: set(f for f in r["flags"].split(";") if f)
          for r in csv.DictReader((NOTES / "fx_pilot_feats.csv").open(encoding="utf-8"))}
    ltas = {json.loads(l)["id"]: json.loads(l) for l in (C.WORK / "b07b_ltas.jsonl").open(encoding="utf-8")}
    fmap = json.loads((C.WORK / "fx" / "map.json").read_text(encoding="utf-8"))
    human = {}
    for f in glob.glob(str(C.WORK / "fx" / "labels_final" / "labels" / "*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        d = d.get("data", d)
        human[fmap[Path(f).stem]["clip_id"]] = not d.get("clean", False)

    def strong(cid: str) -> list[str]:
        why = []
        flags = {g for f in fx.get(cid, ()) for g in f.split("+")}
        if "nb" in flags:
            why.append("bandlimit")
        if "echo" in flags:
            why.append("echo")
        L = ltas.get(cid)
        if L and L["hi_dev"] < -10 and L["lo_dev"] < -10:
            why.append("ltas_phone")
        if human.get(cid):
            why.append("human_effect")
        return why

    flagged = {r["id"]: strong(r["id"]) for r in man}
    blocks = defaultdict(list)
    for r in man:
        blocks[(r["speaker"], r["block"])].append(r["id"])
    block_flag = {k for k, ids in blocks.items()
                  if sum(bool(flagged[i]) for i in ids) / len(ids) >= BLOCK_SHARE and len(ids) >= 2}
    drop = {}
    for r in man:
        why = list(flagged[r["id"]])
        if (r["speaker"], r["block"]) in block_flag and not why:
            why.append("flagged_block")
        if why:
            drop[r["id"]] = why

    out = C.WORK / "qwen"
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": []}
    for r in man:
        if r["split"] not in rows or r["id"] in drop:
            continue
        rows[r["split"]].append({"id": r["id"], "audio": str(C.WORK / r["audio"]), "text": r["text"],
                                 "language": "chinese", "speaker": r["speaker"], "split": r["split"],
                                 "dur": r["dur"], "series": r["series"]})
    for split, rs in rows.items():
        with (out / f"{split}_rows.jsonl").open("w", encoding="utf-8") as f:
            for x in rs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # reference clips: train split, tier A, 2.5-8 s core, no flag of any kind, no flagged block
    refdir = C.WORK / "refs"
    refdir.mkdir(exist_ok=True)
    pool = defaultdict(lambda: defaultdict(list))
    for r in man:
        weak = {g for f in fx.get(r["id"], ()) for g in f.split("+")}
        if (r["split"] == "train" and r["tier"] == "A" and 2.5 <= r["core_dur"] <= 8.0
                and r["id"] not in drop and not weak and r["id"] not in human):
            pool[r["speaker"]][r["series"]].append(r)
    refs = {}
    for spk in C.ROSTER:
        # a seeded draw over the whole pool, so each game is represented in proportion to
        # its share of the character's lines (the vector stands for the data it keys)
        allr = [r for v in pool[spk].values() for r in v]
        refs[spk] = sorted(allr, key=lambda r: C.stable_unit("ref:" + r["id"]))[:N_REF]
        (refdir / f"{spk}.txt").write_text("\n".join(str(C.WORK / r["audio"]) for r in refs[spk]) + "\n", encoding="utf-8")

    (out / "dropped_ids.json").write_text(json.dumps(drop, ensure_ascii=False, indent=0), encoding="utf-8")
    reasons = Counter(w for v in drop.values() for w in v)
    rep = {"manifest": len(man), "dropped": len(drop), "drop_reasons": dict(reasons),
           "flagged_blocks": len(block_flag),
           "dropped_by_split": dict(Counter(r["split"] for r in man if r["id"] in drop)),
           "dropped_by_speaker": dict(Counter(r["speaker"] for r in man if r["id"] in drop)),
           "rows": {k: len(v) for k, v in rows.items()},
           "train_hours": round(sum(x["dur"] for x in rows["train"]) / 3600, 3),
           "train_by_speaker": dict(Counter(x["speaker"] for x in rows["train"])),
           "refs": {s: len(v) for s, v in refs.items()},
           "ref_pool": {s: sum(len(v) for v in pool[s].values()) for s in C.ROSTER},
           "ref_series": {s: dict(Counter(r["series"] for r in refs[s])) for s in C.ROSTER}}
    (C.WORK / "b09_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
