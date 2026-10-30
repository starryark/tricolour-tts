"""Filter the R/P generations into training rows.

  --asr     Paraformer-zh transcripts (FunASR env)                      -> rp/asr.jsonl
  --ident   Qwen Base speaker-encoder identity for P clips (qwen_tts env) -> rp/ident.jsonl
  --select  keep / drop, write trainer rows (any env)                  -> qwen/{R,P}_rows.jsonl
Keep a clip when: no runaway; toneless pinyin error vs its text <= 0.10 (b04 scoring);
and for P, the clip's Qwen-encoder vector is nearest (centred on the 14 locked vectors) to
its own character's locked vector, by a margin >= 0.05. Identity is filtered with Qwen's
encoder on purpose: the evaluation judges identity with CAM++, a different model, so the
filter cannot simply teach to the test.
Rows are split 97/3 train/val by a seeded hash of the clip id.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RP = C.WORK / "rp"
MAX_PER = 0.10
MIN_MARGIN = 0.05


def index() -> list[dict]:
    rows = [json.loads(l) for l in (RP / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:      # pod-generated rows carry paths relative to the rp folder
        if not Path(r["audio"]).is_absolute():
            r["audio"] = str(RP / r["audio"])
    return rows


def asr() -> None:
    from funasr import AutoModel
    rows = index()
    model = AutoModel(model="paraformer-zh", disable_update=True, device="cuda:0")
    out = {}
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        res = model.generate(input=[r["audio"] for r in chunk], batch_size=64, disable_pbar=True)
        for r, h in zip(chunk, res):
            out[r["id"]] = h["text"].replace(" ", "").strip()
    (RP / "asr.jsonl").write_text("".join(json.dumps({"id": k, "para": v}, ensure_ascii=False) + "\n"
                                          for k, v in out.items()), encoding="utf-8")
    print(f"[asr] {len(out)}")


def ident() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from xvectors import speaker_embedding
    rows = [r for r in index() if r["set"] == "P"]
    table = torch.load(C.WORK / "qwen" / "speaker_table.pt")
    names, T = table["names"], table["table"].float()
    mu = T.mean(0)
    U = torch.nn.functional.normalize(T - mu, dim=-1)
    w = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=True),
                                      dtype=torch.bfloat16, device_map="cuda:0")
    out = []
    for r in rows:
        # one clip's raw encoder vector (the helper averages a list and rescales; a single clip is its own mean)
        v = speaker_embedding(w.model, [Path(r["audio"])], "cuda:0", torch.bfloat16)[0].float()
        sims = torch.nn.functional.normalize(v - mu, dim=-1) @ U.T
        own = names.index(r["speaker"])
        other = max(float(sims[j]) for j in range(len(names)) if j != own)
        out.append({"id": r["id"], "own": round(float(sims[own]), 4), "best_other": round(other, 4),
                    "top": names[int(sims.argmax())]})
    (RP / "ident.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out), encoding="utf-8")
    print(f"[ident] {len(out)}")


def select() -> None:
    from b04_verify import err, syllables
    rows = index()
    hyp = {json.loads(l)["id"]: json.loads(l)["para"] for l in (RP / "asr.jsonl").read_text(encoding="utf-8").splitlines()}
    idn = {json.loads(l)["id"]: json.loads(l) for l in (RP / "ident.jsonl").read_text(encoding="utf-8").splitlines()}
    kept, why = {"R": [], "P": []}, Counter()
    for r in rows:
        if r["runaway"]:
            why[f"{r['set']}:runaway"] += 1
            continue
        if err(syllables(r["text"]), syllables(hyp[r["id"]])) > MAX_PER:
            why[f"{r['set']}:asr"] += 1
            continue
        if r["set"] == "P":
            d = idn[r["id"]]
            if d["top"] != r["speaker"] or d["own"] - d["best_other"] < MIN_MARGIN:
                why["P:identity"] += 1
                continue
        split = "val" if C.stable_unit("rpsplit:" + r["id"]) < 0.03 else "train"
        kept[r["set"]].append({"id": r["id"], "audio": r["audio"], "text": r["text"], "language": "chinese",
                               "speaker": r["speaker"], "instruct": r["instruct"], "split": split,
                               "dur": r["dur"], "source": r["set"]})
    for st, rs in kept.items():
        for split in ("train", "val"):
            sel = [x for x in rs if x["split"] == split]
            (C.WORK / "qwen" / f"{st}_{split}_rows.jsonl").write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sel), encoding="utf-8")
    total = Counter(r["set"] for r in rows)
    rep = {"generated": dict(total), "dropped": dict(why), "kept": {k: len(v) for k, v in kept.items()},
           "kept_by_instruct": {st: dict(Counter(x["instruct"] or "-" for x in rs)) for st, rs in kept.items()},
           "P_kept_by_speaker": dict(Counter(x["speaker"] for x in kept["P"]))}
    (RP / "select_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"--asr": asr, "--ident": ident, "--select": select}[sys.argv[1]]()
