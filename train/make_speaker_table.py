"""Locked speaker table for the 14 characters, from the screened references (b09).

Same computation as train_sft.speaker_embedding (Base speaker encoder, 24 kHz 128-mel,
mean over clips rescaled to the clips' typical norm). Written once and passed to every arm
with --speaker_table, so the Base and CustomVoice arms condition on identical vectors.

    python train/make_speaker_table.py
Output: <WORK>/qwen/speaker_table.pt {"names": [...], "table": [14, 2048] fp32, "refs": {...}}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import common as C  # noqa: E402


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from xvectors import speaker_embedding

    refs = {s: [Path(p) for p in (C.WORK / "refs" / f"{s}.txt").read_text(encoding="utf-8").split("\n") if p]
            for s in C.ROSTER}
    w = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=True),
                                      dtype=torch.bfloat16, device_map="cuda:0")
    w.model.speaker_encoder.eval()
    table = torch.cat([speaker_embedding(w.model, refs[s], "cuda:0", torch.bfloat16) for s in C.ROSTER], 0).float()
    out = C.WORK / "qwen" / "speaker_table.pt"
    torch.save({"names": C.ROSTER, "table": table, "refs": {k: [str(p) for p in v] for k, v in refs.items()}}, out)

    old = torch.load(C.WORK / "xvec_rows.pt")
    assert old["names"] == C.ROSTER
    mu = table.mean(0)
    u = torch.nn.functional.normalize(table - mu, dim=-1)
    nn_gap = []
    for i, s in enumerate(C.ROSTER):
        d_old = float((table[i] - old["table"][i].float()).norm())
        d_nn = min(float((table[i] - table[j]).norm()) for j in range(len(C.ROSTER)) if j != i)
        sims = u @ u[i]
        j = max((k for k in range(len(C.ROSTER)) if k != i), key=lambda k: float(sims[k]))
        nn_gap.append(d_old / d_nn)
        print(f"{s:12s} |e|={table[i].norm():.2f}  moved {d_old:.3f} = {d_old / d_nn:.2f} of nearest-row gap  "
              f"nearest(centred) {C.ROSTER[j]} {float(sims[j]):.3f}")
    print(f"[table] {out}")


if __name__ == "__main__":
    main()
