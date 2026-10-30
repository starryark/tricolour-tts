"""B08 - AudioSet channel/room tags for every clip (CED-base), a trained detector for the
echo / telephone / noise effects heard in the game audio.

Scores (max over <= 10 s windows) for the classes below are written per clip. They are a
ranking signal, not labels: CED was trained on AudioSet, not on acted game speech, so
thresholds are set later against a human-labelled sample.

Reuses the Kurisu project's tagger wrapper (kurisu_v2/pod/ced_tagger.py, pinned revision).

    python build/b08_ced_channel.py
Output: <WORK>/b08_ced.jsonl (resumable)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("KURISU_POD", r"E:\STEINS;GATEVoice\kurisu_v2\pod"))
import common as C  # noqa: E402

CLASSES = ["Echo", "Reverberation", "Telephone", "Radio", "Television", "Static", "Noise", "Distortion",
           "White noise", "Hum", "Environmental noise", "Inside, small room", "Inside, large room or hall",
           "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural",
           "Speech synthesizer", "Music", "Hubbub, speech noise, speech babble", "Speech"]


def main() -> None:
    from ced_tagger import CED
    out = C.WORK / "b08_ced.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")]
    todo = [r for r in rows if r["id"] not in done]
    ced = CED()
    ids = ced.label_ids(CLASSES)
    names = [ced.labels[i] for i in ids]
    with out.open("a", encoding="utf-8") as f:
        for n, r in enumerate(todo):
            p = ced.probs(str(C.WORK / r["audio"]))
            f.write(json.dumps({"id": r["id"], **{nm: round(float(p[i]), 4) for nm, i in zip(names, ids)}},
                               ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                f.flush()
                print(f"[b08] {n}/{len(todo)}", flush=True)
    print(f"[b08] done -> {out}")


if __name__ == "__main__":
    main()
