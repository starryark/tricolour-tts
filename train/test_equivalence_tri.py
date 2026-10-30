"""The trainer's rows must be exactly what generate_custom_voice() feeds the talker.

For a Base init (rows from 3000) and a CustomVoice init (rows from 3020, presets kept):
  1. save_checkpoint writes our speakers without touching existing presets, and the
     config keeps every preset (spk_id and spk_is_dialect) next to ours;
  2. for every text x speaker (ours + presets) x language (Auto, Chinese) x instruction
     (none, short, long), the embeddings CodedDataset.collate + build_inputs build equal
     the ones generate_custom_voice hands to talker.generate, position by position, up
     to and including tts_pad + codec_bos;
  3. a speaker-row clash with a preset and a dialect speaker are refused.
fp32 on CPU, ~10 GB RAM, a few minutes per model.

    python test_equivalence_tri.py --base <Base dir or id> --cv <CustomVoice dir or id> --tmp <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sft as T  # noqa: E402

OURS = ["moxiaoju", "chiguo", "zhute"]
PRESETS = ["vivian", "serena", "uncle_fu"]
TEXTS = ["明天上午九点，我们在图书馆门口集合。",
         "……哥哥。笨蛋！没常识！",
         "四点……四十分……",
         "欢迎大家来到maid cafe lapuro！"]
LANGS = ["Auto", "Chinese"]
INSTRUCTS = ["", "用开心的语气说", "用非常生气、语速很快的语气说，带一点哭腔"]


class Captured(Exception):
    pass


def captured_embeds(wrapper, text, speaker, language, instruct):
    talker = wrapper.model.talker
    original = talker.generate

    def fake(*_a, **kw):
        raise Captured(kw["inputs_embeds"].detach().clone())

    talker.generate = fake
    try:
        wrapper.generate_custom_voice(text=text, speaker=speaker, language=language,
                                      instruct=instruct or None, non_streaming_mode=True)
    except Captured as c:
        return c.args[0]
    finally:
        talker.generate = original
    raise RuntimeError("talker.generate was never reached")


def check_model(init: str, row0: int, presets: list[str], tmp: Path) -> list[str]:
    from qwen_tts import Qwen3TTSModel

    failures: list[str] = []
    base_dir = T.resolve_base_dir(init)
    torch.manual_seed(0)
    wrapper = Qwen3TTSModel.from_pretrained(str(base_dir), dtype=torch.float32, attn_implementation="sdpa")
    model = wrapper.model
    base_cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    base_spk = base_cfg["talker_config"].get("spk_id") or {}
    emb0 = model.talker.model.codec_embedding.weight.detach().clone()
    table = torch.randn(len(OURS), model.talker.config.hidden_size) * 0.5

    out = tmp / f"ckpt_{row0}"
    if out.exists():
        shutil.rmtree(out)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    T.save_checkpoint(out, base_dir, sd, table, OURS, None, None, row0)
    del wrapper, model, sd

    # 1. config and rows
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    tc = cfg["talker_config"]
    want = {**base_spk, **{n: row0 + i for i, n in enumerate(OURS)}}
    if tc["spk_id"] != want:
        failures.append(f"[{row0}] spk_id {tc['spk_id']} != {want}")
    if set(tc["spk_is_dialect"]) != set(tc["spk_id"]):
        failures.append(f"[{row0}] spk_is_dialect keys differ from spk_id")
    for n, r in base_spk.items():
        if tc["spk_is_dialect"].get(n) != base_cfg["talker_config"]["spk_is_dialect"].get(n):
            failures.append(f"[{row0}] dialect flag of preset {n} changed")
    w2 = Qwen3TTSModel.from_pretrained(str(out), dtype=torch.float32, attn_implementation="sdpa")
    m2 = w2.model.eval()
    e2 = m2.talker.model.codec_embedding.weight.detach()
    if not torch.allclose(e2[row0:row0 + len(OURS)], table.to(torch.bfloat16).float(), atol=1e-6):
        failures.append(f"[{row0}] saved speaker rows differ from the table")
    for n, r in base_spk.items():   # presets untouched (bf16 round trip of the original row)
        if not torch.allclose(e2[r], emb0[r].to(torch.bfloat16).float(), atol=1e-6):
            failures.append(f"[{row0}] preset row {n}@{r} changed")
    print(f"[{row0}] supported speakers: {m2.get_supported_speakers()}")

    # 2. layout equivalence; the trainer's table = our rows + presets' own rows
    spk_table = torch.cat([e2[row0:row0 + len(OURS)]] + [e2[base_spk[p]].unsqueeze(0) for p in presets], 0)
    names = OURS + presets
    spk_index = {n: i for i, n in enumerate(names)}
    checked = 0
    for text, speaker, lang, ins in itertools.product(TEXTS, names, LANGS, INSTRUCTS):
        ref = captured_embeds(w2, text, speaker, lang, ins)
        row = {"audio": "x", "text": text, "audio_codes": [[1] * 16] * 3,
               "language": lang.lower(), "speaker": speaker, "instruct": ins}
        ds = T.CodedDataset([row], w2.processor, m2.config, None, spk_index=spk_index)
        batch = ds.collate([ds[0]], train=False)
        with torch.no_grad():
            mine, _, _ = T.build_inputs(m2, batch, spk_table, "cpu", True, None)
        n = ref.shape[1]
        p = int(batch["spk_pos"][0]) + 2
        nb = ds.items[0]["text_ids"].shape[1] - 3
        if n != p + nb + 2:
            failures.append(f"[{row0}] length {text[:6]!r}/{speaker}/{lang}/{ins[:4]!r}: generate {n}, trainer {p + nb + 2}")
            continue
        diff = (mine[:, :n] - ref).abs()
        if float(diff.max()) > 1e-5:
            bad = sorted(set(diff.max(-1).values[0].nonzero().flatten().tolist()))[:10]
            failures.append(f"[{row0}] embeds {text[:6]!r}/{speaker}/{lang}/{ins[:4]!r}: max |d| "
                            f"{float(diff.max()):.2e} at {bad} (P={p}, n={n})")
        # masks: instruction positions carry text only
        o = 0 if ds.items[0]["instruct_ids"] is None else ds.items[0]["instruct_ids"].shape[1]
        cem = batch["codec_embedding_mask"][0, :, 0]
        if o and (bool(cem[:o + 3].any()) or not bool(batch["text_embedding_mask"][0, :o, 0].all())):
            failures.append(f"[{row0}] instruction/role masks wrong for {speaker}/{ins[:4]!r}")
        checked += 1
    print(f"[{row0}] layout: {checked} combinations compared")

    # 3. guards
    if base_spk:
        try:
            T.speaker_rows(base_cfg, OURS, min(base_spk.values()) - 1)
            failures.append(f"[{row0}] a row clash with a preset was not refused")
        except SystemExit:
            pass
        dial = [n for n, v in base_cfg["talker_config"]["spk_is_dialect"].items() if v]
        if dial:
            try:
                T.CodedDataset([{"audio": "x", "text": TEXTS[0], "audio_codes": [[1] * 16] * 3,
                                 "language": "chinese", "speaker": dial[0]}], w2.processor, m2.config, None,
                               spk_index={dial[0]: 0})
                failures.append(f"[{row0}] dialect speaker {dial[0]} was not refused")
            except ValueError:
                pass
    del w2, m2
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--cv", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--tmp", required=True)
    args = ap.parse_args()
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    failures = check_model(args.base, 3000, [], tmp)
    failures += check_model(args.cv, 3020, PRESETS, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print(f"PASS: trainer rows == generate_custom_voice inputs (Base@3000 and CustomVoice@3020; "
          f"{len(TEXTS)} texts x speakers x {len(LANGS)} languages x {len(INSTRUCTS)} instructions)")


if __name__ == "__main__":
    main()
