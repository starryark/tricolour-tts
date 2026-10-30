"""Local end-to-end smoke test of the T1 training path (LoRA: the 16 GB card cannot hold a
full fine-tune; collate / speaker table / presets / save are mode-independent).

CustomVoice init + --speaker_table + --preset_speakers + instruction rows -> 24 steps ->
checkpoint -> reload with the stock loader -> generate_custom_voice for one of our
characters and one preset, with and without an instruction. Rows marked as presets reuse
real audio (plumbing only, not training data).

    python train/smoke_local.py --cv <CustomVoice dir> --out <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = Path(r"F:\Tricolour\tts_work\qwen")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (Q / "val_with_codes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(1).shuffle(rows)
    train = []
    for i, r in enumerate(rows[:160]):
        r = dict(r)
        if i % 4 == 1:
            r["instruct"] = "用开心的语气说"
        if i % 8 == 3:
            r["speaker"], r["instruct"] = "vivian", "用生气的语气说"
        train.append(r)
    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train[:128]), encoding="utf-8")
    (out / "val.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train[128:]), encoding="utf-8")
    cmd = [sys.executable, "-X", "utf8", str(HERE / "train_sft.py"), "--init_model_path", a.cv,
           "--spk_row0", "3020", "--preset_speakers", "vivian,serena,uncle_fu",
           "--speaker_table", str(Q / "speaker_table.pt"),
           "--train_jsonl", str(out / "train.jsonl"), "--val_jsonl", str(out / "val.jsonl"),
           "--allow_small_dataset", "--mode", "lora", "--epochs", "1", "--batch_size", "4",
           "--effective_batch", "8", "--lr", "1e-4", "--eval_every", "8", "--log_every", "4",
           "--save_at", "1.0", "--stop_after", "1.0", "--p_auto_lang", "0.25", "--attn", "sdpa",
           "--vram_fraction", "0.9", "--out", str(out / "run"), "--name", "smoke"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    ck = sorted((out / "run").glob("checkpoint-step-*"))[-1]
    cfg = json.loads((ck / "config.json").read_text(encoding="utf-8"))["talker_config"]
    print("[smoke] speakers in checkpoint:", cfg["spk_id"])
    w = Qwen3TTSModel.from_pretrained(str(ck), dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    for spk in ("chiguo", "vivian"):
        for ins in (None, "用生气的语气说"):
            torch.manual_seed(0)
            wavs, sr = w.generate_custom_voice(text="明天上午九点，我们在图书馆门口集合。", speaker=spk,
                                               language="Chinese", instruct=ins, max_new_tokens=400,
                                               non_streaming_mode=True)
            p = out / f"{spk}_{'angry' if ins else 'none'}.wav"
            sf.write(p, wavs[0], sr)
            print(f"[smoke] {spk} {ins or '-'}: {len(wavs[0]) / sr:.2f}s -> {p.name}")


if __name__ == "__main__":
    main()
