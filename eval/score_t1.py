"""Score T1 exactly as pre-registered (research/T1_preregistration.md).

Per checkpoint: runaway rate, CER (Paraformer toneless pinyin error vs text, mean over
clips), ID / SIM (CAM++ centred nearest-centroid among the 14, cast mean), ID_instr (S3,
mean over emotions), LEAK (speaker-relative LTAS phone test), emotion2vec (report only).
Reference points: the real S1 recordings, and the untrained B0 / C0 checkpoints.
Then gates G-run / G-cer / G-id / G-leak and the selection steps 1-4.

    python eval/score_t1.py <eval dir>          (e.g. F:\\Tricolour\\tts_work\\t1_eval)
Output: <eval dir>/scores.json, <eval dir>/scores.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import judge  # noqa: E402
from b04_verify import err, syllables  # noqa: E402
from b07b_ltas_deviation import CENTERS, ltas  # noqa: E402

EMO = ["happy", "angry", "sad"]


def asr(paths: list[str]) -> list[str]:
    import hashlib
    cache = judge.CACHE / f"para_{hashlib.sha1(chr(0).join(paths).encode()).hexdigest()[:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from funasr import AutoModel
    m = AutoModel(model="paraformer-zh", disable_update=True, device="cuda:0")
    out = []
    for i in range(0, len(paths), 64):
        out += [h["text"].replace(" ", "").strip() for h in m.generate(input=paths[i:i + 64], batch_size=64, disable_pbar=True)]
    judge.CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def leak_flags(paths: list[str], speakers: list[str]) -> list[bool]:
    med = json.loads((C.WORK / "b07b_ltas_speaker.json").read_text(encoding="utf-8"))["median_ltas"]
    hi, lo = CENTERS >= 4000, CENTERS <= 250
    out = []
    for p, s in zip(paths, speakers):
        _, v = ltas(("x", p))
        if v is None:
            out.append(False)
            continue
        d = np.asarray(v) - np.asarray(med[s])
        out.append(bool(d[hi].mean() < -10 and d[lo].mean() < -10))
    return out


def runaway(r: dict) -> bool:
    return r["frames"] >= 399 or r["dur"] > 5 * max(1, C.cjk_count(r["text"])) / 4.5


def score_ckpt(cast, d: Path) -> dict:
    rows = [json.loads(l) for l in (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    paths = [str(d / r["audio"]) for r in rows]
    hyp = asr(paths)
    for r, h in zip(rows, hyp):
        r["per"] = err(syllables(r["text"]), syllables(h))
        r["runaway"] = runaway(r)
    out = {"n": len(rows)}
    base = [i for i, r in enumerate(rows) if r["suite"] in ("S1", "S2")]
    out["runaway_S12"] = round(float(np.mean([rows[i]["runaway"] for i in base])), 4)
    out["CER"] = round(float(np.mean([rows[i]["per"] for i in base])), 4)
    idr = cast.accuracy([paths[i] for i in base], [rows[i]["speaker"] for i in base])
    out["ID"], out["ID_per_speaker"], out["confusions"] = idr["cast_mean"], idr["per_speaker"], idr["top_confusions"]
    emb = judge.campp([paths[i] for i in base])
    _, sims = cast.classify(emb)
    out["SIM"] = round(float(np.mean([sims[k, cast.names.index(rows[i]["speaker"])] for k, i in enumerate(base)])), 4)
    lk = leak_flags([paths[i] for i in base], [rows[i]["speaker"] for i in base])
    out["LEAK"] = round(float(np.mean(lk)), 4)
    s3 = [i for i, r in enumerate(rows) if r["suite"] == "S3"]
    if s3:
        out["runaway_S3"] = round(float(np.mean([rows[i]["runaway"] for i in s3])), 4)
        per_emo = {}
        e2v = judge.e2v([paths[i] for i in s3])
        for emo in EMO:
            ii = [k for k, i in enumerate(s3) if rows[i]["emotion"] == emo]
            acc = cast.accuracy([paths[s3[k]] for k in ii], [rows[s3[k]]["speaker"] for k in ii])
            per_emo[emo] = {"ID": acc["cast_mean"], "confusions": acc["top_confusions"][:4],
                            "e2v_target": round(float(np.mean([judge.top1(e2v[k]) == emo for k in ii])), 3),
                            "CER": round(float(np.mean([rows[s3[k]]["per"] for k in ii])), 4)}
        out["S3"] = per_emo
        out["ID_instr"] = round(float(np.mean([per_emo[e]["ID"] for e in EMO])), 4)
        s2 = [i for i in base if rows[i]["suite"] == "S2"]
        e2v2 = judge.e2v([paths[i] for i in s2])
        out["S2_e2v"] = {e: round(float(np.mean([judge.top1(x) == e for x in e2v2])), 3) for e in EMO + ["neutral"]}
        # instruction effect: how much more often emotion2vec hears the requested emotion
        # with the instruction than on the same character and text without it
        out["instr_effect"] = round(float(np.mean([per_emo[e]["e2v_target"] - out["S2_e2v"][e] for e in EMO])), 4)
    return out


def real_refs(cast) -> dict:
    suites = json.loads((C.WORK / "qwen" / "suites.json").read_text(encoding="utf-8"))
    man = {json.loads(l)["id"]: json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")}
    ltas_rows = {json.loads(l)["id"]: json.loads(l) for l in (C.WORK / "b07b_ltas.jsonl").open(encoding="utf-8")}
    ids = [(spk, x["id"], x["text"]) for spk, xs in suites["S1"].items() for x in xs]
    paths = [str(C.WORK / man[i]["audio"]) for _, i, _ in ids]
    hyp = asr(paths)
    acc = cast.accuracy(paths, [s for s, _, _ in ids])
    return {"n": len(ids), "CER": round(float(np.mean([err(syllables(t), syllables(h)) for (_, _, t), h in zip(ids, hyp)])), 4),
            "ID": acc["cast_mean"],
            "LEAK": round(float(np.mean([ltas_rows[i]["hi_dev"] < -10 and ltas_rows[i]["lo_dev"] < -10 for _, i, _ in ids])), 4)}


def main() -> None:
    root = Path(sys.argv[1])
    cast = judge.Cast()
    res = {"real_S1": real_refs(cast), "ckpts": {}}
    for idx in sorted(root.glob("*/*/index.jsonl")):
        d = idx.parent
        if not (d / "DONE").exists():
            print(f"[score] {d} incomplete - skipped")
            continue
        key = f"{d.parent.name}/{d.name}"
        res["ckpts"][key] = score_ckpt(cast, d)
        print(f"[score] {key}: " + json.dumps({k: v for k, v in res["ckpts"][key].items()
                                                 if k in ("CER", "ID", "SIM", "LEAK", "ID_instr", "runaway_S12", "runaway_S3")}))
    ck = res["ckpts"]
    b0 = ck.get("B0/checkpoint-step-0")
    c0 = ck.get("C0/checkpoint-step-0")
    real = res["real_S1"]
    gates = {}
    for k, v in ck.items():
        if k.startswith(("B0/", "C0/")):
            continue
        g = {"G-run": v["runaway_S12"] <= 0.005 and v.get("runaway_S3", 0) <= 0.005,
             "G-cer": b0 is not None and v["CER"] <= b0["CER"] + 0.02,
             "G-id": v["ID"] >= 0.90,
             "G-leak": v["LEAK"] <= real["LEAK"] + 0.01}
        if "instr_effect" in v:     # Amendment 1: a CV checkpoint must keep >= half the untrained effect
            g["G-emo"] = c0 is not None and v["instr_effect"] >= 0.5 * c0["instr_effect"]
        gates[k] = {**g, "eligible": all(g.values())}
    res["gates"] = gates
    elig = [k for k, g in gates.items() if g["eligible"]]
    sel = {}
    b1 = [k for k in elig if k.startswith("B1/")]
    if b1:
        best = min(ck[k]["CER"] for k in b1)
        ties = [k for k in b1 if ck[k]["CER"] <= best + 0.005]
        sel["B1_winner"] = max(ties, key=lambda k: ck[k]["SIM"])
    cv = [k for k in elig if k.startswith(("C1/", "C2/"))]
    if cv:
        best = max(ck[k]["ID_instr"] for k in cv)
        ties = [k for k in cv if ck[k]["ID_instr"] >= best - 0.02]
        sel["CV_winner"] = min(ties, key=lambda k: ck[k]["CER"])
    if "CV_winner" in sel and "B1_winner" in sel:
        c, b = ck[sel["CV_winner"]], ck[sel["B1_winner"]]
        sel["CV_noninferior"] = c["CER"] <= b["CER"] + 0.01 and c["ID"] >= b["ID"] - 0.02
        sel["decision"] = sel["CV_winner"] if sel["CV_noninferior"] else "trade-off: report both to the user"
    elif "B1_winner" in sel:
        sel["decision"] = sel["B1_winner"] + " (no eligible CV checkpoint: no emotion control)"
    else:
        sel["decision"] = "no eligible checkpoint"
    res["selection"] = sel
    (root / "scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| checkpoint | CER | ID | SIM | LEAK | ID_instr | runaway | eligible |", "|---|---|---|---|---|---|---|---|"]
    for k, v in ck.items():
        g = gates.get(k, {})
        lines.append(f"| {k} | {v['CER']:.4f} | {v['ID']:.3f} | {v['SIM']:.3f} | {v['LEAK']:.3f} | "
                     f"{v.get('ID_instr', float('nan')):.3f} | {v['runaway_S12']:.3f} | {g.get('eligible', 'ref')} |")
    lines.append(f"\nreal S1: {real}\n\nselection: {json.dumps(sel, ensure_ascii=False)}")
    (root / "scores.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
