"""B07b - speaker-relative spectral deviation: telephone/radio EQ detector.

Raw band-energy features confound voice type with channel (male voices carry little above
4 kHz anyway, so they all look "band-limited"). Here each clip's long-term average spectrum
(1/3-octave bands, 100 Hz - 10 kHz, active frames, normalised to 0 dB mean) is compared with
its own character's median LTAS:
  hi_dev   mean deviation (dB) in bands >= 4 kHz   (telephone: strongly negative)
  lo_dev   mean deviation (dB) in bands <= 250 Hz  (telephone: strongly negative)
  tilt_dev slope of the deviation across bands, dB per octave
A telephone-band clip shows hi_dev and lo_dev both far below the character's own norm.

    python build/b07b_ltas_deviation.py
Output: <WORK>/b07b_ltas.jsonl, <WORK>/b07b_ltas_speaker.json (per-character median LTAS)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NFFT, HOP = 2048, 512
CENTERS = 1000 * 2 ** (np.arange(-10, 11) / 3)          # 99 Hz .. 10.1 kHz
CENTERS = CENTERS[(CENTERS >= 100) & (CENTERS <= 10100)]


def ltas(job):
    cid, path = job
    x, sr = sf.read(path, dtype="float32")
    x = x[int(C.LEAD_PAD * sr): len(x) - int(C.TAIL_PAD * sr)]
    if len(x) < NFFT * 2:
        return cid, None
    n = 1 + (len(x) - NFFT) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP][:n] * np.hanning(NFFT)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    edb = 10 * np.log10(P.sum(1) + 1e-12)
    S = P[edb > np.percentile(edb, 95) - 30].mean(0)
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    bands = []
    for c in CENTERS:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        bands.append(S[(f >= lo) & (f < hi)].mean())
    b = 10 * np.log10(np.asarray(bands) + 1e-14)
    return cid, (b - b.mean()).round(2).tolist()


def main() -> None:
    rows = [json.loads(l) for l in (C.WORK / "manifest.jsonl").open(encoding="utf-8")]
    spk = {r["id"]: r["speaker"] for r in rows}
    with ProcessPoolExecutor(12) as ex:
        res = dict(ex.map(ltas, [(r["id"], str(C.WORK / r["audio"])) for r in rows], chunksize=64))
    by = defaultdict(list)
    for cid, v in res.items():
        if v is not None:
            by[spk[cid]].append(v)
    med = {s: np.median(np.asarray(v), 0) for s, v in by.items()}
    hi = CENTERS >= 4000
    lo = CENTERS <= 250
    oct_ = np.log2(CENTERS / 1000)
    with (C.WORK / "b07b_ltas.jsonl").open("w", encoding="utf-8") as f:
        for cid, v in res.items():
            if v is None:
                continue
            d = np.asarray(v) - med[spk[cid]]
            f.write(json.dumps({"id": cid, "hi_dev": round(float(d[hi].mean()), 2),
                                "lo_dev": round(float(d[lo].mean()), 2),
                                "tilt_dev": round(float(np.polyfit(oct_, d, 1)[0]), 3),
                                "ltas": v}) + "\n")
    (C.WORK / "b07b_ltas_speaker.json").write_text(json.dumps(
        {"centers_hz": CENTERS.round(1).tolist(), "median_ltas": {s: m.round(2).tolist() for s, m in med.items()}}),
        encoding="utf-8")
    print(f"[b07b] {len(res)} clips")


if __name__ == "__main__":
    main()
