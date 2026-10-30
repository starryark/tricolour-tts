"""B01 - select the 14-character pool, normalise text, re-cut and level the audio.

Input:  voice_dataset_game/metadata.jsonl and its 48 kHz wavs.
Output: <WORK>/wavs/<speaker>/<clip_id>.wav (24 kHz mono PCM16, exactly silent 0.12 s lead /
        0.25 s tail, 5 ms fades), <WORK>/b01_decisions.jsonl (every candidate row, kept or
        not, with the reason) and <WORK>/b01_report.json.

Loudness: one gain per (series, speaker), bringing that group's median clip loudness to
-23 LUFS, so whispers stay quieter than shouts within a character. A clip whose peak would
pass -1 dBFS gets its own lower gain (flag `peak_limited`).

    python build/b01_build.py [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EXCLUDE_FLAGS = {"name_conflict", "joint_line"}


def measure(job: tuple[str, str]) -> dict:
    """Pass 1: decode, resample, find speech, measure the faded core's loudness."""
    from scipy.signal import resample_poly
    import pyloudnorm as pyln
    clip_id, path = job
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if sr != C.SR:
        from math import gcd
        g = gcd(sr, C.SR)
        x = resample_poly(x, C.SR // g, sr // g).astype(np.float32)
    b = C.speech_bounds(x)
    if b is None:
        return {"id": clip_id, "ok": False}
    core = x[b[0]:b[1]]
    meter = pyln.Meter(C.SR)
    seg = core if len(core) >= int(0.4 * C.SR) else np.tile(core, int(0.4 * C.SR) // max(1, len(core)) + 1)
    try:
        loud = float(meter.integrated_loudness(seg))
    except Exception:
        loud = float("-inf")
    if not np.isfinite(loud):
        loud = 20 * np.log10(np.sqrt(np.mean(core ** 2)) + 1e-9) - 3.0
    return {"id": clip_id, "ok": True, "b0": int(b[0]), "b1": int(b[1]),
            "core_dur": (b[1] - b[0]) / C.SR, "loud": loud,
            "peak": float(np.max(np.abs(core)))}


def write(job: tuple[str, str, int, int, float, str]) -> dict:
    """Pass 2: apply the group gain (peak-limited), fades and pads, write PCM16."""
    from scipy.signal import resample_poly
    clip_id, path, b0, b1, gain_db, out = job
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if sr != C.SR:
        from math import gcd
        g = gcd(sr, C.SR)
        x = resample_poly(x, C.SR // g, sr // g).astype(np.float32)
    core = x[b0:b1].astype(np.float32).copy()
    fade = int(C.FADE * C.SR)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    core[:fade] *= ramp
    core[-fade:] *= ramp[::-1]
    gain = 10 ** (gain_db / 20)
    limited = False
    peak = float(np.max(np.abs(core))) * gain
    if peak > C.PEAK_MAX:
        gain *= C.PEAK_MAX / peak
        limited = True
    y = np.concatenate([np.zeros(int(C.LEAD_PAD * C.SR), np.float32), core * gain,
                        np.zeros(int(C.TAIL_PAD * C.SR), np.float32)])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, C.SR, subtype="PCM_16")
    return {"id": clip_id, "gain_db": round(20 * np.log10(gain), 2), "peak_limited": limited,
            "dur": round(len(y) / C.SR, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    C.WORK.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in (C.DATASET / "metadata.jsonl").open(encoding="utf-8")]
    cand, decisions = [], {}
    for r in rows:
        if r["speaker_id"] not in C.ROSTER:
            continue
        d = {"id": r["clip_id"], "speaker": r["speaker_id"], "series": r["series"],
             "scene": r["scene"], "tier": r["tier"], "src": str(C.DATASET / r["rel_path"]),
             "text_src": r["text"], "src_flags": r["flags"]}
        if r["tier"] not in ("A", "B"):
            d.update(keep=False, reason=f"tier_{r['tier']}")
        elif EXCLUDE_FLAGS & set(r["flags"]):
            d.update(keep=False, reason=sorted(EXCLUDE_FLAGS & set(r["flags"]))[0])
        else:
            norm = C.normalise_text(r["text"])
            d["text"] = norm
            d["digits_converted"] = any(ch.isdigit() for ch in r["text"])
            why = C.text_drop_reason(r["text"], norm)
            if why:
                d.update(keep=False, reason=why)
            else:
                cand.append(d)
        decisions[d["id"]] = d

    print(f"[b01] {len(cand)} candidates of {len(decisions)} roster rows", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        meas = {m["id"]: m for m in ex.map(measure, [(d["id"], d["src"]) for d in cand], chunksize=64)}

    # duration gates on the re-cut clip (core + pads)
    pads = C.LEAD_PAD + C.TAIL_PAD
    short_pool = []
    for d in cand:
        m = meas[d["id"]]
        if not m["ok"]:
            d.update(keep=False, reason="no_speech")
            continue
        d.update(loud_core=round(m["loud"], 2), core_dur=round(m["core_dur"], 3))
        dur = m["core_dur"] + pads
        if dur > C.MAX_DUR:
            d.update(keep=False, reason="too_long")
        elif dur < C.MIN_DUR:
            if C.lexical_count(d["text"]) >= 2:
                short_pool.append(d)
                d.update(keep=None, reason="short_pool")
            else:
                d.update(keep=False, reason="short_nonlexical")
        else:
            d.update(keep=True, reason="")
    n_main = sum(1 for d in cand if d.get("keep") is True)
    cap = int(C.SHORT_SHARE / (1 - C.SHORT_SHARE) * n_main)
    short_pool.sort(key=lambda d: C.stable_unit("short:" + d["id"]))
    for i, d in enumerate(short_pool):
        d.update(keep=i < cap, reason="short_kept" if i < cap else "short_over_cap")

    kept = [d for d in cand if d.get("keep")]
    groups = defaultdict(list)
    for d in kept:
        groups[(d["series"], d["speaker"])].append(d["loud_core"])
    ggain = {k: float(np.clip(C.TARGET_LUFS - statistics.median(v), -C.GROUP_GAIN_LIMIT, C.GROUP_GAIN_LIMIT))
             for k, v in groups.items()}

    jobs = []
    for d in kept:
        out = C.WORK / "wavs" / d["speaker"] / f"{d['id']}.wav"
        d["audio"] = str(out.relative_to(C.WORK)).replace("\\", "/")
        m = meas[d["id"]]
        jobs.append((d["id"], d["src"], m["b0"], m["b1"], ggain[(d["series"], d["speaker"])], str(out)))
    with ProcessPoolExecutor(args.workers) as ex:
        for w in ex.map(write, jobs, chunksize=64):
            decisions[w["id"]].update(gain_db=w["gain_db"], peak_limited=w["peak_limited"], dur=w["dur"])

    with (C.WORK / "b01_decisions.jsonl").open("w", encoding="utf-8") as f:
        for d in decisions.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    kept = [d for d in decisions.values() if d.get("keep")]
    by_spk = defaultdict(lambda: [0, 0.0])
    for d in kept:
        by_spk[d["speaker"]][0] += 1
        by_spk[d["speaker"]][1] += d["dur"]
    report = {
        "rows": len(decisions), "kept": len(kept),
        "kept_hours": round(sum(d["dur"] for d in kept) / 3600, 3),
        "short_kept": sum(1 for d in kept if d["reason"] == "short_kept"),
        "reasons": dict(Counter(d["reason"] for d in decisions.values() if not d.get("keep"))),
        "peak_limited": sum(1 for d in kept if d.get("peak_limited")),
        "digits_converted": sum(1 for d in kept if d.get("digits_converted")),
        "group_gain_db": {f"{s}:{k}": round(v, 2) for (s, k), v in sorted(ggain.items())},
        "per_speaker": {k: {"clips": v[0], "minutes": round(v[1] / 60, 1)}
                        for k, v in sorted(by_spk.items(), key=lambda x: -x[1][1])},
    }
    (C.WORK / "b01_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
