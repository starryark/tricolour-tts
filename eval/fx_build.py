"""FX check - blind human labels for production effects on a 100-clip sample.

The sample and its strata come from research/notes/fx_label_sample.csv (50 detector-flagged,
20 unflagged neighbours of flagged clips, 30 unflagged random), drawn before any listening.
Audio is the *source* game file (untrimmed: the dataset export cut reverb tails at -45 dBFS),
mono, at its native rate. The stratum and detector flags stay in fx_label_key.csv, local only.

    python eval/fx_build.py
Output: <WORK>/fx/index.html, <WORK>/fx/audio/f###.mp3, <WORK>/fx/map.json (local)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

NOTES = Path(r"F:\Tricolour\research\notes")
ROOTS = {
    "TRI": ["F:/Tricolour/Tricolour_Lovestory/_extracted/voice/",
            "F:/Tricolour/Tricolour_Lovestory/_extracted/append_zero/voice/",
            "F:/Tricolour/Tricolour_Lovestory/_extracted/append_animate/voice/"],
    "SLS": ["F:/Tricolour/SL-extracted/_extracted/"],
    "HM": ["F:/Tricolour/HeartfulMemories/_extracted/"],
}


def main() -> None:
    meta = {}
    for l in (C.DATASET / "metadata.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        meta[r["clip_id"]] = r
    sample = list(csv.DictReader((NOTES / "fx_label_sample.csv").open(encoding="utf-8-sig")))
    out = C.WORK / "fx"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    clips, mapping, missing = [], {}, []
    for row in sample:
        cid = row["clip_id"]
        m = meta[cid]
        src = next((os.path.join(r, m["source"]) for r in ROOTS[m["series"]]
                    if os.path.exists(os.path.join(r, m["source"]))), None)
        if src is None:
            missing.append(cid)
            continue
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        x = x.mean(1)
        n = f"f{int(row['n']):03d}"
        sf.write(str(out / "audio" / f"{n}.mp3"), x, sr, format="MP3")
        clips.append({"n": n, "who": m["speaker"], "text": m["text"]})
        mapping[n] = {"clip_id": cid, "source": src, "sr": sr, "dur_src": round(len(x) / sr, 2)}
    page = (Path(__file__).with_name("fx_page.html")).read_text(encoding="utf-8")
    (out / "index.html").write_text(page.replace("/*__CLIPS__*/[]", json.dumps(clips, ensure_ascii=False)),
                                    encoding="utf-8")
    (out / "map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(clips)} clips written, {len(missing)} missing: {missing}")


if __name__ == "__main__":
    main()
