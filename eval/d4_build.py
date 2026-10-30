"""D4 - human emotion labels on a stratified sample, to calibrate emotion2vec on this cast.

Every emotion number this project reports (label quality, controllability of a trained
model) is only as good as emotion2vec's accuracy on these voices, which nobody has measured
for acted Mandarin. This builds a blind labelling page: ~300 real clips, stratified by
emotion2vec's top-1 class and spread over the 14 characters, with no model output shown and
the script line hidden until the rater asks for it (label the delivery, not the words).

    python eval/d4_build.py
Output: <WORK>/d4/index.html, <WORK>/d4/audio/c###.mp3 (published beside the page),
        <WORK>/d4/key.json (local only: page index -> clip id, emotion2vec label, face code)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

QUOTA = {"neutral": 50, "happy": 50, "angry": 50, "sad": 50, "surprised": 50, "fearful": 30, "other": 30}
NAMES = {"moxiaoju": "墨小菊", "wenzhi": "文芷", "luoyan": "骆衍", "chiguo": "迟菓", "chiyao": "迟耀",
         "yexin": "叶心", "zhute": "朱特", "huanglili": "黄丽丽", "moshu": "墨叔", "dinglaoshi": "丁老师",
         "linjie": "琳姐", "wenzhou": "文绉", "xuyayi": "徐阿姨", "ciertou": "刺儿头"}


def group(label: str) -> str:
    return label if label in QUOTA else "other"


def select() -> list[dict]:
    emo = {}
    for l in (C.WORK / "b05_emotion.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("e2v"):
            emo[r["id"]] = r["e2v"]
    faces = {}
    p = C.WORK / "b03_faces.jsonl"
    if p.exists():
        for l in p.open(encoding="utf-8"):
            r = json.loads(l)
            faces[r["id"]] = r.get("face")
    cells = defaultdict(lambda: defaultdict(list))
    for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        if not d.get("keep") or d["id"] not in emo or not 1.2 <= d["core_dur"] <= 8.0:
            continue
        sc = emo[d["id"]]
        top = max(sc, key=sc.get)
        d["e2v_top"], d["e2v_p"] = top, sc[top]
        cells[group(top)][d["speaker"]].append(d)
    picked = []
    for g, quota in QUOTA.items():
        per = {s: sorted(v, key=lambda d: C.stable_unit("d4:" + d["id"])) for s, v in cells[g].items()}
        order = sorted(per, key=lambda s: C.ROSTER.index(s))
        i = 0
        got = []
        while len(got) < quota and any(i < len(v) for v in per.values()):
            for s in order:
                if i < len(per[s]) and len(got) < quota:
                    got.append(per[s][i])
            i += 1
        picked += got
    picked.sort(key=lambda d: C.stable_unit("d4order:" + d["id"]))
    for d in picked:
        d["face"] = faces.get(d["id"])
    return picked


PAGE = Path(__file__).with_name("d4_page.html")


def main() -> None:
    out = C.WORK / "d4"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    picked = select()
    clips, key = [], {}
    for n, d in enumerate(picked, start=1):
        cid = f"c{n:03d}"
        wav, sr = sf.read(str(C.WORK / d["audio"]), dtype="float32")
        sf.write(str(out / "audio" / f"{cid}.mp3"), wav, sr, format="MP3")
        clips.append({"n": cid, "who": NAMES[d["speaker"]], "text": d["text"]})
        key[cid] = {"id": d["id"], "speaker": d["speaker"], "e2v_top": d["e2v_top"],
                    "e2v_p": round(d["e2v_p"], 3), "face": d["face"], "series": d["series"]}
    html = PAGE.read_text(encoding="utf-8").replace("/*__CLIPS__*/[]", json.dumps(clips, ensure_ascii=False))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(clips), "clips;", dict(Counter(group(v["e2v_top"]) for v in key.values())),
          dict(Counter(v["speaker"] for v in key.values())))


if __name__ == "__main__":
    main()
