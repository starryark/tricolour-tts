"""Independent judges for generated speech: identity (CAM++ zh-cn 200k, centred
nearest-centroid among the 14 cast centroids) and emotion (emotion2vec+ large top-1).

Never judge identity with Qwen's own speaker encoder: it is the conditioning signal, so it
would agree with any output that copies the row (research/voice_model_approach s10).

Centroids: <= 200 real train-split clips per character (no val/test, no effect-dropped clip).
Real ceiling: real test-split clips with core 1.5-6 s (generated probes are that long).
Embeddings are cached by path in <WORK>/judge_cache/.

    python eval/judge.py d2            # score research/D2_preregistration.md
Needs funasr (e.g. the `.venv-curation` env).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

CACHE = C.WORK / "judge_cache"
_models = {}


def _model(kind: str):
    if kind not in _models:
        from funasr import AutoModel
        name = {"campp": "iic/speech_campplus_sv_zh-cn_16k-common", "e2v": "iic/emotion2vec_plus_large"}[kind]
        _models[kind] = AutoModel(model=name, disable_update=True, device="cuda:0")
    return _models[kind]


def _key(kind: str, paths: list[str]) -> Path:
    return CACHE / f"{kind}_{hashlib.sha1(chr(0).join(paths).encode()).hexdigest()[:16]}.npz"


def campp(paths: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    k = _key("campp", paths)
    if k.exists():
        return np.load(k)["e"]
    m = _model("campp")
    out = []
    for i in range(0, len(paths), 64):
        res = m.generate(input=paths[i:i + 64], disable_pbar=True)
        out += [r["spk_embedding"].reshape(-1).float().cpu().numpy() for r in res]
    e = np.stack(out)
    np.savez(k, e=e)
    return e


def e2v(paths: list[str]) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    k = _key("e2v", paths).with_suffix(".json")
    if k.exists():
        return json.loads(k.read_text(encoding="utf-8"))
    m = _model("e2v")
    out = []
    for i in range(0, len(paths), 32):
        res = m.generate(paths[i:i + 32], granularity="utterance", extract_embedding=False,
                         batch_size=32, disable_pbar=True)
        for r in res:
            out.append({l.split("/")[-1]: round(float(s), 4) for l, s in zip(r["labels"], r["scores"])})
    k.write_text(json.dumps(out), encoding="utf-8")
    return out


def top1(scores: dict) -> str:
    return max(scores, key=scores.get)


class Cast:
    """Centred CAM++ centroids for the 14 characters, from real training clips."""

    def __init__(self, n_per: int = 200):
        # Centroids from train-split clips only (never val/test, never effect-dropped), so
        # no evaluated real clip or held-out line sits inside a centroid (T1 review S-4).
        dropped_path = C.WORK / "qwen" / "dropped_ids.json"
        dropped = set(json.loads(dropped_path.read_text(encoding="utf-8"))) if dropped_path.exists() else set()
        train, val = defaultdict(list), defaultdict(list)
        for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8"):
            d = json.loads(l)
            if d["id"] in dropped:
                continue
            p = str(C.WORK / d["audio"])
            if d["split"] == "train":
                train[d["speaker"]].append((C.stable_unit("cent:" + d["id"]), p))
            elif d["split"] == "test" and 1.5 <= d["core_dur"] <= 6.0:
                val[d["speaker"]].append(p)
        self.names = list(C.ROSTER)
        cents = []
        for s in self.names:
            paths = [p for _, p in sorted(train[s])[:n_per]]
            cents.append(campp(paths).mean(0))
        self.raw = np.stack(cents)
        self.mu = self.raw.mean(0)
        self.cent = self._unit(self.raw - self.mu)
        self.val = {s: sorted(val[s])[:100] for s in self.names}

    @staticmethod
    def _unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def classify(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sims = self._unit(emb - self.mu) @ self.cent.T
        return sims.argmax(1), sims

    def accuracy(self, paths: list[str], speakers: list[str]) -> dict:
        if not paths:
            return {}
        top, sims = self.classify(campp(paths))
        per = defaultdict(list)
        conf = Counter()
        for t, s, row in zip(top, speakers, sims):
            ok = self.names[t] == s
            per[s].append(ok)
            if not ok:
                conf[f"{s}->{self.names[t]}"] += 1
        accs = {s: float(np.mean(v)) for s, v in per.items()}
        return {"cast_mean": round(float(np.mean(list(accs.values()))), 3),
                "per_speaker": {s: round(a, 3) for s, a in accs.items()},
                "top_confusions": conf.most_common(6)}

    def ceiling(self, speakers: list[str]) -> dict:
        paths, spk = [], []
        for s in speakers:
            paths += self.val[s]
            spk += [s] * len(self.val[s])
        return self.accuracy(paths, spk)


def score_d2() -> dict:
    idx = [json.loads(l) for l in (C.WORK / "d2" / "index.jsonl").read_text(encoding="utf-8").splitlines() if l]
    cast = Cast()
    by = defaultdict(list)
    for r in idx:
        by[r["cond"]].append(r)
    speakers = sorted({r["speaker"] for r in idx}, key=C.ROSTER.index)
    out = {"ceiling_real_val": cast.ceiling(speakers), "conds": {}}
    target = {"cv_happy": "happy", "cv_angry": "angry", "cv_sad": "sad"}
    for cond, rs in by.items():
        paths = [str(C.WORK / r["audio"]) for r in rs]
        emo = e2v(paths)
        labels = [top1(e) for e in emo]
        c = {"n": len(rs), "identity": cast.accuracy(paths, [r["speaker"] for r in rs]),
             "e2v_top1": dict(Counter(labels).most_common()),
             "runaway": sum(r["runaway"] for r in rs),
             "dur_median": round(float(np.median([r["dur"] for r in rs])), 2)}
        if cond in target:
            c["p_target"] = round(sum(l == target[cond] for l in labels) / len(labels), 3)
            c["p_target_per_speaker"] = {s: round(np.mean([l == target[cond] for l, r in zip(labels, rs) if r["speaker"] == s]), 3)
                                         for s in speakers}
        if cond == "cv_none":
            c["p_neutral"] = round(sum(l == "neutral" for l in labels) / len(labels), 3)
        out["conds"][cond] = c
    k = out["conds"]
    if all(x in k for x in ["base", "cv_none", "cv_happy", "cv_angry", "cv_sad"]):
        b = k["base"]["identity"]["cast_mean"]
        rule = {
            "1_identity_none": k["cv_none"]["identity"]["cast_mean"] >= b - 0.10,
            "2_identity_instruct": all(k[c]["identity"]["cast_mean"] >= b - 0.15 for c in target),
            "4_default_neutral": k["cv_none"]["p_neutral"] >= 0.70,
            "5_runaway": all(k[c]["runaway"] <= 0.01 * k[c]["n"] for c in k),
        }
        out["rule"] = rule
        # rule 3 is report-only (Amendment 1): emotion2vec precision on this cast is 0.08-0.40
        # for non-neutral classes (D4), so emotion following is judged by a blind human A/B
        out["report_only"] = {"3_emotion_mean_e2v": round(float(np.mean([k[c]["p_target"] for c in target])), 3),
                              "3_would_pass_at_0.40": float(np.mean([k[c]["p_target"] for c in target])) >= 0.40}
        out["pass_automatic"] = all(rule.values())
    return out


def score_d3() -> dict:
    """Identity and emotion per (emotion, alpha) for Base + row + alpha*tau."""
    idx = [json.loads(l) for l in (C.WORK / "d3" / "index.jsonl").read_text(encoding="utf-8").splitlines() if l]
    cast = Cast()
    by = defaultdict(list)
    for r in idx:
        by[r["cond"]].append(r)
    out = {}
    for cond, rs in sorted(by.items()):
        paths = [str(C.WORK / r["audio"]) for r in rs]
        labels = [top1(e) for e in e2v(paths)]
        out[cond] = {"identity": cast.accuracy(paths, [r["speaker"] for r in rs]),
                     "p_target": round(sum(l == rs[0]["emotion"] for l in labels) / len(labels), 3),
                     "e2v_top1": dict(Counter(labels).most_common()),
                     "runaway": sum(r["runaway"] for r in rs)}
    return out


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "d2"
    res = {"d2": score_d2, "d3": score_d3}[what]()
    (C.WORK / what / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
