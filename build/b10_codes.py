"""B10 - Qwen3-TTS 12 Hz codec codes for every training row (local GPU), so the pod only
receives JSONL and never the audio.

Same encoder call as kurisu_v2/pod/p04_codes_v3.py. Every clip must yield dur * 12.5
frames (+-3) or the run stops: a mismatch means a wrong sample rate or a truncated file.

    python build/b10_codes.py <rows.jsonl> [<rows.jsonl> ...]
Output: <rows>_with_codes.jsonl next to each input (resumable: rows already coded are kept)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BATCH = 32


def main() -> None:
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", local_files_only=True),
                                            device_map="cuda:0")
    for src in map(Path, sys.argv[1:]):
        out = src.with_name(src.name.replace("_rows.jsonl", "_with_codes.jsonl"))
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        done = {}
        if out.exists():
            for l in out.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    done[r["id"]] = r
        todo = [r for r in rows if r["id"] not in done]
        for r in todo:
            info = sf.info(r["audio"])
            if info.samplerate != 24000 or info.channels != 1:
                raise SystemExit(f"{r['id']}: {info.samplerate} Hz x {info.channels} ch, expected 24 kHz mono")
            r["dur"] = info.frames / info.samplerate
        todo.sort(key=lambda r: r["dur"])
        with out.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                enc = tok.encode([r["audio"] for r in chunk])
                for r, codes in zip(chunk, enc.audio_codes):
                    codes = codes.cpu().tolist()
                    if abs(len(codes) - r["dur"] * 12.5) > 3 or len(codes[0]) != 16:
                        raise SystemExit(f"{r['id']}: {len(codes)} frames x {len(codes[0])} for {r['dur']:.2f}s")
                    f.write(json.dumps({**r, "audio_codes": codes}, ensure_ascii=False) + "\n")
                if i // BATCH % 50 == 0:
                    f.flush()
                    print(f"[codes] {src.name}: {i + len(chunk)}/{len(todo)}", flush=True)
        # final order and completeness check
        coded = {json.loads(l)["id"]: l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        missing = [r["id"] for r in rows if r["id"] not in coded]
        if missing:
            raise SystemExit(f"{out.name}: {len(missing)} rows missing, e.g. {missing[:3]}")
        out.write_text("".join(coded[r["id"]] + "\n" for r in rows), encoding="utf-8")
        print(f"[codes] {out.name}: {len(rows)} rows, {out.stat().st_size / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
