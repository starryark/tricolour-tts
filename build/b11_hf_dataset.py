"""B11 - the public dataset: every clip of the 14-character manifest (what the model was built
from), as 24 kHz FLAC in the Hugging Face audiofolder layout, with all labels.

    python build/b11_hf_dataset.py <out dir>
Output: <out>/data/{train,validation,test}/<shard>/<clip id>.flac + metadata.jsonl, and
<out>/stats.json for the card. The README is written separately.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NAMES = {"moxiaoju": "墨小菊", "wenzhi": "文芷", "luoyan": "骆衍", "chiguo": "迟菓", "chiyao": "迟耀",
         "yexin": "叶心", "zhute": "朱特", "huanglili": "黄丽丽", "moshu": "墨叔", "dinglaoshi": "丁老师",
         "linjie": "琳姐", "wenzhou": "文绉", "xuyayi": "徐阿姨", "ciertou": "刺儿头"}
GAMES = {"TRI": "Tricolour Lovestory (三色绘恋)", "SLS": "Sunnyrain Lovestory (三色绘恋S)",
         "HM": "Heartful Memories (叶心)"}
SPLIT = {"train": "train", "val": "validation", "test": "test"}
SHARD = 1000


def human_labels() -> tuple[dict, dict]:
    emo = {}
    key = json.loads((C.WORK / "d4" / "key.json").read_text(encoding="utf-8"))
    for f in glob.glob(str(C.WORK / "d4" / "labels_final" / "labels" / "*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        d = d.get("data", d)
        emo[key[Path(f).stem]["id"]] = {"label": d["label"], "whisper": bool(d.get("soft")), "strong": bool(d.get("strong"))}
    fx = {}
    fmap = json.loads((C.WORK / "fx" / "map.json").read_text(encoding="utf-8"))
    for f in glob.glob(str(C.WORK / "fx" / "labels_final" / "labels" / "*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        d = d.get("data", d)
        fx[fmap[Path(f).stem]["clip_id"]] = [k for k in ("echo", "reverb", "phone", "noise", "other") if d.get(k)] or ["clean"]
    return emo, fx


def main() -> None:
    out = Path(sys.argv[1])
    man = [json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")]
    dropped = json.loads((C.WORK / "qwen" / "dropped_ids.json").read_text(encoding="utf-8"))
    trained = {json.loads(l)["id"] for l in (C.WORK / "qwen" / "train_rows.jsonl").open(encoding="utf-8")}
    emo, fx = human_labels()
    rows = defaultdict(list)
    jobs = []
    for r in sorted(man, key=lambda r: r["id"]):
        split = SPLIT[r["split"]]
        n = len(rows[split])
        rel = f"{n // SHARD:03d}/{r['id']}.flac"
        e2v = r.get("e2v") or {}
        rows[split].append({
            "file_name": rel, "id": r["id"], "speaker_id": r["speaker"], "speaker_name": NAMES[r["speaker"]],
            "text": r["text"], "text_source": r["text_src"], "game": r["series"], "game_title": GAMES[r["series"]],
            "scene": r["scene"], "duration_seconds": r["dur"], "sample_rate": C.SR, "source_tier": r["tier"],
            "asr_qwen3": r["asr_qwen"], "asr_paraformer": r["asr_paraformer"],
            "pinyin_error_best": r["pinyin_err_best"],
            "emotion2vec_top": r.get("e2v_top"), "emotion2vec_scores": {k: v for k, v in e2v.items() if k != "<unk>"},
            "sprite_face": r.get("face"),
            "effect_flags": dropped.get(r["id"], []),
            "used_in_training": r["id"] in trained,
            "human_emotion": emo.get(r["id"]), "human_effects": fx.get(r["id"]),
        })
        jobs.append((C.WORK / r["audio"], out / "data" / split / rel))

    def convert(job):
        src, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            x, sr = sf.read(str(src), dtype="int16")
            sf.write(str(dst), x, sr, format="FLAC", subtype="PCM_16")
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(convert, jobs))
    for split, rs in rows.items():
        with (out / "data" / split / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"clips": len(man), "hours": round(sum(r["dur"] for r in man) / 3600, 2),
             "splits": {s: len(v) for s, v in rows.items()},
             "per_speaker": {k: {"clips": c, "minutes": round(sum(r["dur"] for r in man if r["speaker"] == k) / 60, 1)}
                             for k, c in Counter(r["speaker"] for r in man).most_common()},
             "used_in_training": sum(1 for r in man if r["id"] in trained),
             "effect_flagged": len(dropped), "human_emotion_labels": len(emo),
             "human_effect_labels_in_set": sum(1 for r in man if r["id"] in fx),
             "bytes": sum(f.stat().st_size for f in (out / "data").rglob("*.flac"))}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
