"""B04 - does each clip say its line?  Two ASR families, compared in pinyin space.

Raw CER punishes homophones (ASR picks its own characters for names like 迟菓 -> 迟果), so
the comparison is toneless pinyin syllables. Interjection characters (嗯啊哦...) are removed
from both sides first: ASR adds and drops them freely, and a missed 嗯 is not a wrong line.

Threshold: a clip is dropped when BOTH systems are further from its own line than T, where T
is the 1st percentile of the control distribution (the same hypothesis scored against a
different line of the same speaker) - i.e. a rejected clip matches its line no better than
1 in 100 random lines would.

    python build/b04_verify.py
Output: <WORK>/b04_verify.jsonl, <WORK>/b04_report.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

_LATIN = re.compile(r"[A-Za-z]+")


def syllables(text: str, tone: bool = False) -> list[str]:
    from pypinyin import Style, lazy_pinyin
    text = C.digits_to_hanzi(text) if re.search(r"\d", text) else text
    text = "".join(ch for ch in text if ch not in C.INTERJ)
    out = []
    for piece in re.findall(r"[一-鿿]+|[A-Za-z]+", text):
        if _LATIN.fullmatch(piece):
            out.append(piece.lower())
        else:
            out += lazy_pinyin(piece, style=Style.TONE3 if tone else Style.NORMAL, neutral_tone_with_five=True)
    return out


def err(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    import jiwer
    return jiwer.wer(" ".join(ref), " ".join(hyp) if hyp else "<empty>")


def chars(text: str) -> str:
    return "".join(re.findall(r"[一-鿿A-Za-z0-9]", text))


def main() -> None:
    import jiwer
    dec = [json.loads(l) for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8")]
    dec = {d["id"]: d for d in dec if d.get("keep")}
    asr = {}
    for l in (C.WORK / "b02_asr.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        asr[r["id"]] = r
    ids = [i for i in dec if asr.get(i, {}).get("qwen") is not None and asr[i].get("para") is not None]
    print(f"[b04] {len(ids)}/{len(dec)} clips have both hypotheses")

    by_spk = defaultdict(list)
    for i in ids:
        by_spk[dec[i]["speaker"]].append(i)
    rng = random.Random(0)
    rows, ctrl = [], []
    for i in ids:
        d, a = dec[i], asr[i]
        ref = syllables(d["text"])
        r = {"id": i, "speaker": d["speaker"], "n_syl": len(ref)}
        for k in ("qwen", "para"):
            hyp = syllables(a[k])
            r[f"per_{k}"] = round(err(ref, hyp), 4)
            r[f"tper_{k}"] = round(err(syllables(d["text"], True), syllables(a[k], True)), 4)
            rc, hc = chars(d["text"]), chars(a[k])
            r[f"cer_{k}"] = round(jiwer.cer(rc, hc) if rc else 1.0, 4)
        other = rng.choice(by_spk[d["speaker"]])
        if other != i and len(ref) >= 3:
            oref = syllables(dec[other]["text"])
            ctrl.append(min(err(oref, syllables(a["qwen"])), err(oref, syllables(a["para"]))))
        rows.append(r)

    T = float(np.percentile(ctrl, 1))
    for r in rows:
        best = min(r["per_qwen"], r["per_para"])
        r["per_best"] = round(best, 4)
        # very short lines: one syllable of disagreement is not evidence of a wrong take
        r["mismatch"] = bool(best > T and best * max(1, r["n_syl"]) >= 2)
    with (C.WORK / "b04_verify.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def q(v):
        return {p: round(float(np.percentile(v, p)), 3) for p in (50, 90, 95, 99)}
    rep = {
        "clips": len(rows), "control_pairs": len(ctrl), "threshold_T": round(T, 3),
        "control_per": q(ctrl),
        "per_qwen": q([r["per_qwen"] for r in rows]), "per_para": q([r["per_para"] for r in rows]),
        "per_best": q([r["per_best"] for r in rows]),
        "cer_qwen": q([r["cer_qwen"] for r in rows]), "cer_para": q([r["cer_para"] for r in rows]),
        "exact_syllables_either": sum(1 for r in rows if r["per_best"] == 0),
        "mismatch": sum(r["mismatch"] for r in rows),
        "mismatch_by_speaker": dict(Counter(r["speaker"] for r in rows if r["mismatch"])),
    }
    (C.WORK / "b04_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
