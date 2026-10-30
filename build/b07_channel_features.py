"""B07 - cheap per-clip channel features: band limits (telephone/radio) and noise floor.

Computed on the 24 kHz training clip's speech core (the exact-zero pads excluded), over
"active" STFT frames within 30 dB of the clip's 95th-percentile frame energy:
  hf4, hf6   share of energy above 4 / 6 kHz (telephone speech carries ~none above 3.4 kHz)
  lf250      share of energy below 250 Hz (telephone band starts ~300 Hz)
  bw99       frequency below which 99% of the energy lies
  floor_db   10th-percentile frame energy minus the 95th, over all core frames (a noise-floor
             / SNR proxy; strongly negative = clean)
These say nothing about reverb, which needs a model (see research notes E/F).

    python build/b07_channel_features.py [--workers 12]
Output: <WORK>/b07_channel.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NFFT, HOP = 1024, 256


def features(job: tuple[str, str]) -> dict:
    cid, path = job
    x, sr = sf.read(path, dtype="float32")
    a, b = int(C.LEAD_PAD * sr), len(x) - int(C.TAIL_PAD * sr)
    x = x[a:b]
    if len(x) < NFFT * 2:
        return {"id": cid}
    n = 1 + (len(x) - NFFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP][:n] * np.hanning(NFFT)
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    e = P.sum(1) + 1e-12
    edb = 10 * np.log10(e)
    p95 = np.percentile(edb, 95)
    act = edb > p95 - 30
    S = P[act].sum(0)
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    tot = S.sum() + 1e-12
    cum = np.cumsum(S) / tot
    return {"id": cid,
            "hf4": round(float(S[f >= 4000].sum() / tot), 5),
            "hf6": round(float(S[f >= 6000].sum() / tot), 5),
            "lf250": round(float(S[f < 250].sum() / tot), 5),
            "bw99": int(f[np.searchsorted(cum, 0.99)]),
            "floor_db": round(float(np.percentile(edb, 10) - p95), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")]
    jobs = [(r["id"], str(C.WORK / r["audio"])) for r in rows]
    with ProcessPoolExecutor(args.workers) as ex, (C.WORK / "b07_channel.jsonl").open("w", encoding="utf-8") as f:
        for r in ex.map(features, jobs, chunksize=64):
            f.write(json.dumps(r) + "\n")
    print(f"[b07] {len(jobs)} clips -> {C.WORK / 'b07_channel.jsonl'}")


if __name__ == "__main__":
    main()
