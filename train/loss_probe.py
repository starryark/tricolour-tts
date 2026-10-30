"""Step-0 sanity probe through the trainer's own data path (no training, bf16, local GPU).

Talker / sub-talker cross-entropy on held-out val rows for:
  correct   each row with its own character's locked vector
  shuffled  each row with another character's vector (a fixed derangement)
  zero      the speaker slot zeroed
If the row layout and the speaker table are right, `correct` must be clearly lowest, and
its talker CE should sit near the Base model's held-out value (~1.1, upstream PR #278).

    python train/loss_probe.py --init <model dir> [--n 96]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sft as T  # noqa: E402


def main() -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--table", default=r"F:\Tricolour\tts_work\qwen\speaker_table.pt")
    ap.add_argument("--val", default=r"F:\Tricolour\tts_work\qwen\val_with_codes.jsonl")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--language", default="chinese")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]
    for r in rows:
        r["language"] = a.language
    w = Qwen3TTSModel.from_pretrained(a.init, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    model = w.model.eval()
    tab = torch.load(a.table)
    names = tab["names"]
    spk = tab["table"].to(torch.bfloat16).cuda()
    idx = {n: i for i, n in enumerate(names)}
    ds = T.CodedDataset(rows, w.processor, model.config, 400, spk_index=idx)
    capture = {}
    model.talker.model.register_forward_hook(
        lambda _m, _i, o: capture.__setitem__("h", o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]))
    args = argparse.Namespace(text_projection=True, label_alignment="fixed", sub_talker_weight=0.3)
    perm = list(range(len(names)))
    perm = perm[1:] + perm[:1]                      # derangement: i -> i+1
    tables = {"correct": spk, "shuffled": spk[perm], "zero": torch.zeros_like(spk)}
    import torch.nn.functional as F
    K = 12                                           # ~1 s: before the audio context reveals the voice
    for name, table in tables.items():
        tot = torch.zeros(2)
        frames = 0
        early, n_early = 0.0, 0
        with torch.no_grad():
            for b in range(0, len(ds), 8):
                batch = ds.collate([ds[i] for i in range(b, min(b + 8, len(ds)))], train=False)
                _, main, sub, n = T.compute_loss(model, batch, table, "cuda", args, capture)
                tot += torch.tensor([float(main), float(sub)]) * n
                frames += n
                # per-position talker CE on the first K frames of each clip (labels shifted by
                # one, as ForCausalLMLoss does: logits at t predict the label at t+1)
                embeds, _, codec_mask = T.build_inputs(model, batch, table, "cuda", True)
                logits = model.talker(inputs_embeds=embeds, attention_mask=batch["attention_mask"].cuda(),
                                      use_cache=False).logits.float()
                labels = batch["codec_0_labels"].cuda()
                ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                     ignore_index=-100, reduction="none").view(labels.shape[0], -1)
                cm = codec_mask[:, 1:].cuda()
                rank = cm.long().cumsum(1)               # 1-based frame index along each row
                sel = cm & (rank <= K)
                early += float(ce[sel].sum())
                n_early += int(sel.sum())
        print(f"{name:9s} talker CE {tot[0] / frames:.4f}   sub-talker CE {tot[1] / frames:.4f}   "
              f"first-{K}-frames talker CE {early / n_early:.4f}   ({frames} frames)")


if __name__ == "__main__":
    main()
