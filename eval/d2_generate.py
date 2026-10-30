"""D2 generation - CustomVoice + our locked rows, with and without emotion instruct; Base +
the same vectors as the identity baseline.  Rule: research/D2_preregistration.md.

    python eval/d2_generate.py [--chars moxiaoju,wenzhi,...]
Output: <WORK>/d2/<cond>/<speaker>_<idx>.wav and <WORK>/d2/index.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import common as C  # noqa: E402

CHARS = ["moxiaoju", "wenzhi", "luoyan", "chiguo", "chiyao", "yexin", "wenzhou", "zhute"]
INSTRUCT = {"cv_none": None, "cv_happy": "用开心的语气说", "cv_angry": "用生气的语气说", "cv_sad": "用伤心的语气说"}
ROW0 = 3020
KW = dict(max_new_tokens=400, non_streaming_mode=True)


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--conds", default="base," + ",".join(INSTRUCT))
    args = ap.parse_args()
    chars = args.chars.split(",")
    conds = args.conds.split(",")
    texts = [t.strip() for t in (Path(__file__).parent / "probes_neutral_zh.txt").read_text(encoding="utf-8").splitlines() if t.strip()]
    rows = torch.load(C.WORK / "xvec_rows.pt")
    vec = {n: rows["table"][i] for i, n in enumerate(rows["names"])}
    out = C.WORK / "d2"
    index_path = out / "index.jsonl"
    done = set()
    if index_path.exists():
        done = {(r["cond"], r["speaker"]) for r in map(json.loads, index_path.read_text(encoding="utf-8").splitlines())}

    def emit(cond, spk, wavs, sr):
        (out / cond).mkdir(parents=True, exist_ok=True)
        with index_path.open("a", encoding="utf-8") as f:
            for i, (t, w) in enumerate(zip(texts, wavs)):
                p = out / cond / f"{spk}_{i:02d}.wav"
                sf.write(p, np.asarray(w, dtype=np.float32), sr, subtype="PCM_16")
                dur = len(w) / sr
                expect = C.cjk_count(t) / 4.5
                f.write(json.dumps({"cond": cond, "speaker": spk, "idx": i, "text": t,
                                    "audio": str(p.relative_to(C.WORK)).replace("\\", "/"),
                                    "dur": round(dur, 3), "frames": int(round(dur * 12.5)),
                                    "runaway": bool(dur * 12.5 >= 399 or dur > 5 * expect)},
                                   ensure_ascii=False) + "\n")

    if "base" in conds:
        base = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=True),
                                             dtype=torch.bfloat16, device_map="cuda:0")
        for ci, spk in enumerate(chars):
            if ("base", spk) in done:
                continue
            item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec[spk].to("cuda:0").to(torch.bfloat16),
                                        x_vector_only_mode=True, icl_mode=False)
            torch.manual_seed(1000 + C.ROSTER.index(spk))
            wavs, sr = base.generate_voice_clone(text=texts, language=["Chinese"] * len(texts),
                                                 voice_clone_prompt=[item] * len(texts), **KW)
            emit("base", spk, wavs, sr)
            print(f"[d2] base {spk}", flush=True)
        del base
        torch.cuda.empty_cache()

    cvc = [c for c in conds if c in INSTRUCT]
    if cvc:
        cv = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", local_files_only=True),
                                           dtype=torch.bfloat16, device_map="cuda:0")
        tc = cv.model.config.talker_config
        emb = cv.model.talker.model.codec_embedding.weight
        taken = set(tc.spk_id.values())
        with torch.no_grad():
            for i, name in enumerate(C.ROSTER):
                row = ROW0 + i
                assert row not in taken, row
                emb[row] = vec[name].to(emb.device, emb.dtype)
                tc.spk_id[name] = row
                tc.spk_is_dialect[name] = False
        for cond in cvc:
            for spk in chars:
                if (cond, spk) in done:
                    continue
                torch.manual_seed(1000 + C.ROSTER.index(spk))
                ins = INSTRUCT[cond]
                wavs, sr = cv.generate_custom_voice(text=texts, speaker=[spk] * len(texts),
                                                    language=["Chinese"] * len(texts),
                                                    instruct=None if ins is None else [ins] * len(texts), **KW)
                emit(cond, spk, wavs, sr)
                print(f"[d2] {cond} {spk}", flush=True)


if __name__ == "__main__":
    main()
