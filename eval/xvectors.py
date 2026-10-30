"""Locked speaker vectors for the cast: the mean Base speaker-encoder vector per character.

Same computation as kurisu_v2/pod/train_sft.py `speaker_embedding()` (24 kHz, 128-mel,
mean over clips, rescaled to the clips' typical norm), so a vector used in probes is the
vector a fine-tune would lock into its row.

Clips: tier A, core 2.5-8 s, not in the dataset's val split, <= 60 per character, chosen by
a seeded hash and round-robin over the games so no single game dominates the vector.

    python eval/xvectors.py
Output: <WORK>/xvec_rows.pt  {"names": [...], "table": [n, 2048], "refs": {name: [paths]}}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

N_REF = 60


def val_ids() -> set[str]:
    with (C.DATASET / "splits" / "val.csv").open(encoding="utf-8") as f:
        return {Path(r["path"]).stem for r in csv.DictReader(f, delimiter="|")}


def pick_refs() -> dict[str, list[Path]]:
    held = val_ids()
    pool = defaultdict(lambda: defaultdict(list))
    for l in (C.WORK / "b01_decisions.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        if d.get("keep") and d["tier"] == "A" and d["id"] not in held and 2.5 <= d["core_dur"] <= 8.0:
            pool[d["speaker"]][d["series"]].append(d)
    refs = {}
    for spk in C.ROSTER:
        by_series = [sorted(v, key=lambda d: C.stable_unit("ref:" + d["id"])) for v in pool[spk].values()]
        out, i = [], 0
        while len(out) < N_REF and any(i < len(s) for s in by_series):
            for s in by_series:
                if i < len(s) and len(out) < N_REF:
                    out.append(C.WORK / s[i]["audio"])
            i += 1
        refs[spk] = out
    return refs


def speaker_embedding(model, paths, device, dtype):
    import soundfile as sf
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    embs = []
    for p in paths:
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        assert sr == C.SR, (p, sr)
        with torch.inference_mode():
            mels = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128,
                                   sampling_rate=24000, hop_size=256, win_size=1024, fmin=0,
                                   fmax=12000).transpose(1, 2)
            embs.append(model.speaker_encoder(mels.to(device).to(dtype))[:1].float())
    stack = torch.cat(embs, 0)
    emb = stack.mean(0, keepdim=True)
    return (emb / emb.norm() * stack.norm(dim=-1).mean()).cpu()


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    refs = pick_refs()
    w = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=True),
                                      dtype=torch.bfloat16, device_map="cuda:0")
    enc = w.model.speaker_encoder.eval()
    table = torch.cat([speaker_embedding(w.model, refs[s], "cuda:0", torch.bfloat16) for s in C.ROSTER], 0)
    torch.save({"names": C.ROSTER, "table": table, "refs": {k: [str(p) for p in v] for k, v in refs.items()}},
               C.WORK / "xvec_rows.pt")
    u = torch.nn.functional.normalize(table - table.mean(0, keepdim=True), dim=-1)
    sims = u @ u.T
    for i, s in enumerate(C.ROSTER):
        j = max((k for k in range(len(C.ROSTER)) if k != i), key=lambda k: float(sims[i, k]))
        print(f"{s:12s} refs={len(refs[s]):3d} |e|={table[i].norm():.2f}  nearest(centred)={C.ROSTER[j]} {float(sims[i, j]):.3f}")


if __name__ == "__main__":
    main()
