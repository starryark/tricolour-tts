"""Fixed T1 evaluation texts: S1 = up to 20 effect-free test-split lines per character
(seeded), with the real clip ids so the real recordings give the reference points;
S2 = the 30 probes.

    python train/make_suites.py
Output: <WORK>/qwen/suites.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402


def main() -> None:
    man = [json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")]
    drop = set(json.loads((C.WORK / "qwen" / "dropped_ids.json").read_text(encoding="utf-8")))
    train_texts = {r["text"] for r in man if r["split"] == "train"}     # no S1 line is also trained on
    s1 = {}
    for spk in C.ROSTER:
        cand = [r for r in man if r["speaker"] == spk and r["split"] == "test" and r["id"] not in drop
                and 4 <= C.cjk_count(r["text"]) and not re.search(r"[A-Za-z]", r["text"])
                and C.lexical_count(r["text"]) >= 3 and r["text"] not in train_texts]
        cand.sort(key=lambda r: C.stable_unit("s1:" + r["id"]))
        s1[spk] = [{"id": r["id"], "text": r["text"]} for r in cand[:20]]
    probes = [t.strip() for t in (Path(__file__).resolve().parents[1] / "eval" / "probes_neutral_zh.txt")
              .read_text(encoding="utf-8").splitlines() if t.strip()]
    out = C.WORK / "qwen" / "suites.json"
    out.write_text(json.dumps({"S1": s1, "S2": probes}, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in s1.items()}, "S2", len(probes), "->", out)


if __name__ == "__main__":
    main()
