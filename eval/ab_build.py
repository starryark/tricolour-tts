"""Blind emotion A/B for the selected checkpoint (T1 prereg selection step 5).

16 pairs per emotion: the S3 clip (instructed) vs its S2 twin (same character, same text,
no instruction), sides randomised. Forced choice: "which sounds more <emotion>?". An
emotion counts as followed if the instructed side wins with an exact 95% binomial lower
bound above 50% (>= 12 of 16).

    python eval/ab_build.py <checkpoint eval dir>   (e.g. tts_work/t1_eval/C1/checkpoint-step-834)
Output: <WORK>/ab/index.html, audio/a###_{A,B}.mp3, key.json (local only)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

EMO = {"happy": "开心", "angry": "生气", "sad": "伤心"}
NAMES = {"moxiaoju": "墨小菊", "wenzhi": "文芷", "luoyan": "骆衍", "chiguo": "迟菓", "chiyao": "迟耀",
         "yexin": "叶心", "zhute": "朱特", "huanglili": "黄丽丽", "moshu": "墨叔", "dinglaoshi": "丁老师",
         "linjie": "琳姐", "wenzhou": "文绉", "xuyayi": "徐阿姨", "ciertou": "刺儿头"}
PER = 16


def main() -> None:
    src = Path(sys.argv[1])
    rows = [json.loads(l) for l in (src / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    s2 = {(r["speaker"], r["text_id"]): r for r in rows if r["suite"] == "S2"}
    s3 = {(r["emotion"], r["speaker"], r["text_id"]): r for r in rows if r["suite"] == "S3"}
    rng = random.Random(20261030)
    out = C.WORK / "ab"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    pairs = []
    for emo in EMO:
        roster = list(C.ROSTER)
        rng.shuffle(roster)
        texts = [f"p{i:02d}" for i in range(30)]
        rng.shuffle(texts)
        for k in range(PER):
            spk, tid = roster[k % len(roster)], texts[k]
            pairs.append((emo, spk, tid))
    rng.shuffle(pairs)
    clips, key = [], {}
    for n, (emo, spk, tid) in enumerate(pairs, start=1):
        pid = f"a{n:03d}"
        inst_side = rng.choice("AB")
        files = {inst_side: s3[(emo, spk, tid)], ("B" if inst_side == "A" else "A"): s2[(spk, tid)]}
        for side, r in files.items():
            wav, sr = sf.read(str(src / r["audio"]), dtype="float32")
            sf.write(str(out / "audio" / f"{pid}_{side}.mp3"), wav, sr, format="MP3")
        clips.append({"n": pid, "who": NAMES[spk], "emo": EMO[emo], "text": s2[(spk, tid)]["text"]})
        key[pid] = {"emotion": emo, "speaker": spk, "text_id": tid, "instructed": inst_side}
    page = (Path(__file__).with_name("ab_page.html")).read_text(encoding="utf-8")
    (out / "index.html").write_text(page.replace("/*__PAIRS__*/[]", json.dumps(clips, ensure_ascii=False)), encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"source": str(src), "pairs": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(clips)} pairs -> {out}")


if __name__ == "__main__":
    main()
