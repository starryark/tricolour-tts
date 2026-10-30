"""B03 - the sprite expression each TRI line was spoken with (author-given emotion signal).

TRI (Kirikiri/KAG) scripts set a character's face with `[墨小菊 f417 ...]` and hide the
sprite with `消`/`hide`. A face code is f<brow><eyes><mouth>[h<blush>][t<tears/sweat>]; the
image is fgimage/<char>_*/..._face_<BB><EE><MM>... (faces.tjs lists the irregular ones).
Brow parts seen on 墨小菊: 01 soft, 02 lowered toward the centre (frown), 03 raised (surprise),
04 asymmetric. Emotion names are assigned later from emotion2vec agreement (b05), not here.

For each kept TRI clip: replay its script up to the clip's line and record the speaker's
current face code (or `hidden`), plus whether a shake (`vibration=`) or `action=` was applied
to the speaker since their previous line.

    python build/b03_faces.py
Output: <WORK>/b03_faces.jsonl
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TRI = Path(r"F:\Tricolour\Tricolour_Lovestory\_extracted")
NAMES = {"moxiaoju": "墨小菊", "wenzhi": "文芷", "luoyan": "骆衍", "chiyao": "迟耀", "chiguo": "迟菓",
         "moshu": "墨叔", "zhute": "朱特", "dinglaoshi": "丁老师", "wenzhou": "文绉", "linjie": "琳姐",
         "ciertou": "刺儿头"}
FACE = re.compile(r"^f(\d)(\d+)((?:h\d)?(?:t\d)?)$")


def parse_face(tok: str) -> dict | None:
    m = FACE.match(tok)
    if not m:
        return None
    rest = m.group(2)
    return {"code": tok, "brow": int(m.group(1)), "eyes_mouth": rest,
            "blush": "h" in m.group(3), "tears": "t" in m.group(3)}


def main() -> None:
    src = {}
    with (TRI / "dataset" / "voice_transcripts.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            src[r["voice_file"]] = (r["script"], int(r["line"]))
    meta = {}
    for l in (C.DATASET / "metadata.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["series"] == "TRI":
            meta[r["clip_id"]] = r
    kept = [json.loads(l) for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8")]
    kept = [d for d in kept if d.get("keep") and d["series"] == "TRI"]

    want = defaultdict(list)                 # script -> [(line, clip_id, speaker)]
    missing = 0
    for d in kept:
        r = meta[d["id"]]
        key = src.get(r["source"])
        if key is None:
            missing += 1
            continue
        want[key[0]].append((key[1], d["id"], d["speaker"]))

    name_re = re.compile(r"\[(" + "|".join(NAMES.values()) + r")(\s[^\]]*)?\]")
    by_name = {v: k for k, v in NAMES.items()}
    out, stats = [], Counter()
    for script, items in want.items():
        path = TRI / script
        lines = path.read_bytes().decode("utf-16", "ignore").splitlines()
        items.sort()
        state = {}                            # speaker -> face dict | "hidden"
        moved = defaultdict(set)              # speaker -> {"shake", "action:<x>"} since last line
        j = 0
        for ln, text in enumerate(lines, start=1):
            while j < len(items) and items[j][0] == ln:
                _, cid, spk = items[j]
                face = state.get(spk)
                rec = {"id": cid, "speaker": spk, "script": script, "line": ln,
                       "face": None if face in (None, "hidden") else face["code"],
                       "brow": None if face in (None, "hidden") else face["brow"],
                       "blush": None if face in (None, "hidden") else face["blush"],
                       "tears": None if face in (None, "hidden") else face["tears"],
                       "sprite": "hidden" if face == "hidden" else ("none" if face is None else "shown"),
                       "motion": sorted(moved[spk])}
                out.append(rec)
                stats[rec["sprite"]] += 1
                moved[spk].clear()
                j += 1
            if text.lstrip().startswith(";"):
                continue
            for m in name_re.finditer(text):
                spk = by_name[m.group(1)]
                body = (m.group(2) or "").split()
                for tok in body:
                    if tok in ("消", "hide"):
                        state[spk] = "hidden"
                    f = parse_face(tok)
                    if f:
                        state[spk] = f
                    if tok.startswith("vibration="):
                        moved[spk].add("shake")
                    if tok.startswith("action="):
                        moved[spk].add("action:" + tok.split("=", 1)[1])
        if j < len(items):
            stats["line_past_eof"] += len(items) - j

    with (C.WORK / "b03_faces.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    codes = Counter(r["face"] for r in out if r["face"])
    rep = {"tri_kept": len(kept), "no_source_row": missing, "sprite": dict(stats),
           "with_face": sum(1 for r in out if r["face"]), "distinct_codes": len(codes),
           "top_codes": codes.most_common(25),
           "brow": dict(Counter(r["brow"] for r in out if r["face"])),
           "blush": sum(1 for r in out if r["blush"]), "tears": sum(1 for r in out if r["tears"]),
           "motion": dict(Counter(m for r in out for m in r["motion"])),
           "per_speaker_face_share": {s: round(sum(1 for r in out if r["speaker"] == s and r["face"]) /
                                           max(1, sum(1 for r in out if r["speaker"] == s)), 3)
                                      for s in NAMES}}
    (C.WORK / "b03_faces_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
