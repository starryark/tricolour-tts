# Fine-tuning Qwen3-TTS: four silent bugs in the official `sft_12hz.py`, and multi-speaker and emotion-instruction training

You fine-tuned [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with the official `finetuning/sft_12hz.py`, and the loss fell, but the voice sounds off: slurred, rushed or not quite the speaker. This page covers four likely causes. They are silent: on the 1.7B model nothing crashes and the loss still goes down.

**Status:**
- **Checked against** upstream commit [`022e286`](https://github.com/QwenLM/Qwen3-TTS/commit/022e286) (2026-03-17). That is the latest change to `finetuning/` as of 2026-10-30.
- **Library version:** `qwen-tts` 0.1.1, the latest on PyPI.
- **Upstream issues:** each bug has one, all still open.
- **If upstream has changed since,** check the code again before relying on this page.

**This repository's trainer, [`train/train_sft.py`](../train/train_sft.py), fixes all four.** [`train/test_equivalence_tri.py`](../train/test_equivalence_tri.py) checks that the trainer's inputs are exactly what `generate_custom_voice` builds at inference. The trainer also adds two things upstream lacks:
- several speakers in one model;
- fine-tuning the CustomVoice model while keeping its presets and its emotion and style instructions.

It trained the [Tricolour Lovestory (三色绘恋) character voice model](https://huggingface.co/starrydark/Tricolour_Lovestory_Qwen3_TTS).

中文：官方 Qwen3-TTS 微调脚本 `sft_12hz.py` 有四个"不报错、loss 照样下降"的隐性问题：标签被移位两次、子模型（code predictor）读到了要预测的那一帧、漏了 `text_projection`、说话人编码器被权重衰减悄悄改变。本仓库的 `train/train_sft.py` 全部修复，并支持多角色（多说话人）微调、在 CustomVoice 上微调时保留预置音色和情感指令（instruct）。

## The four bugs

| # | Bug | Effect | Upstream issue |
|---|---|---|---|
| 1 | Labels are shifted twice | the talker learns to predict two codec frames ahead instead of one | [#278](https://github.com/QwenLM/Qwen3-TTS/issues/278) |
| 2 | The sub-talker reads the hidden state of the frame it is predicting | codebooks 1–15 can be copied from the input in training, but not at inference | [#337](https://github.com/QwenLM/Qwen3-TTS/issues/337) |
| 3 | `text_projection` is skipped | the model trains on text inputs it never sees at inference; the 0.6B crashes | [#278](https://github.com/QwenLM/Qwen3-TTS/issues/278), [#336](https://github.com/QwenLM/Qwen3-TTS/issues/336) |
| 4 | The speaker encoder decays under AdamW | the speaker vector drifts away from the one saved in the checkpoint | [#351](https://github.com/QwenLM/Qwen3-TTS/issues/351) |

### 1. Labels are shifted twice

```python
# upstream finetuning/sft_12hz.py
outputs = model.talker(inputs_embeds=input_embeddings[:, :-1, :],
                       attention_mask=attention_mask[:, :-1],
                       labels=codec_0_labels[:, 1:], ...)
```

- **Cause:** the talker's loss (the `transformers` causal-LM loss) already shifts labels by one position. Slicing inputs and labels by hand shifts them a second time.
- **Effect:** every hidden state is trained against the codebook-0 code two frames ahead. The sub-talker has the mirror problem: its 15 logits are compared with 14 labels plus one ignored position.
- **Fix:** pass the unshifted tensors and let the loss shift once. This is `--label_alignment fixed`, the default; `--label_alignment official` reproduces upstream for an A/B comparison.

### 2. The sub-talker sees the frame it predicts

```python
# upstream, as of 022e286
talker_hidden_states = hidden_states[codec_mask[:, :-1]]
```

- **How inference works:** the sub-talker (code predictor) predicts codebooks 1–15 of frame *t* from two inputs: the talker hidden state and codebook 0 of frame *t*. That hidden state is the one that predicted frame *t*, at position *t*−1.
- **What upstream does:** the mask above selects position *t*. The input at position *t* already holds the embeddings of all 16 codebooks of frame *t*. In training the sub-talker can read its answer from its input; at inference it cannot.
- **Fix:** take the hidden state at *t*−1.

### 3. The text projection is skipped

- **Cause:** upstream adds `talker.model.text_embedding(ids)` straight to the codec embedding. Inference passes text embeddings through `talker.text_projection` first.
- **On the 1.7B:** the widths happen to match, so nothing fails. The model simply trains on inputs it will never see again.
- **On the 0.6B:** the widths differ and training crashes ([#336](https://github.com/QwenLM/Qwen3-TTS/issues/336)).
- **Fix:** apply `text_projection`.

### 4. The speaker encoder drifts

- **Cause:** upstream optimises `AdamW(model.parameters(), weight_decay=0.01)`. The speaker-encoder output is detached, so the encoder gets no gradient, but weight decay still shrinks its weights at every step.
- **Effect:** the conditioning vector the model trains with drifts away from the one written into the saved checkpoint.
- **Fix:** freeze the speaker encoder and compute the speaker vector once.

## How the fixes were checked

- **Equivalence test** (`train/test_equivalence_tri.py`):
  - For 216 combinations of speaker, language id and instruction, the trainer's input embeddings match `generate_custom_voice` position by position.
  - The test also catches a deliberately injected bug, so it can fail.
  - It runs on the GPU machine before every training job (`pod/bootstrap.sh`), with the model revisions pinned.
- **Outcome:** with the fixes, the released model scores as follows on held-out lines, under a selection rule written before training ([results](../research/T1_results.md)):
  - pinyin CER 0.0096;
  - identity 0.936 among 14 characters, against 0.967 for real recordings.

## What the trainer adds beyond single-speaker fine-tuning

- **Many speakers in one model:**
  - Upstream saves one speaker, at codec-embedding row 3000, and replaces the config's speaker list with it.
  - Here, `--speaker_table` gives each character a fixed speaker vector: the mean Base-model speaker-encoder embedding of 60 screened clean clips (`train/make_speaker_table.py`). A CustomVoice init requires this table.
  - The vectors are written to consecutive rows from `--spk_row0` and are never trained.
- **CustomVoice presets survive:**
  - The 1.7B CustomVoice presets sit at rows between 2861 and 3066 (`uncle_fu` is 3010), so this model's characters use 3020–3033.
  - `speaker_rows()` refuses any row the base model already uses.
  - The saved config keeps the presets next to the new speakers.
- **Instructions (emotion and style):**
  - A row with an `instruct` field gets the prefix `<|im_start|>user\n{instruct}<|im_end|>\n` on the text channel, before the role tokens. This is the layout `generate_custom_voice(..., instruct=...)` uses.
- **Instruction replay keeps instruction-following alive:**
  - Fine-tuning CustomVoice on uninstructed character lines alone erodes its ability to follow instructions.
  - To prevent that, mix in about 1,500 clips that the untrained CustomVoice model generates with its own presets, under no instruction and five instructions. Filter them by ASR first (`eval/rp_generate.py`).
  - Name those presets with `--preset_speakers`, so their training rows use the presets' own codec rows.
  - Bonus: identity under instruction rose from 0.74 to 0.94, because the characters stopped drifting to a generic voice when instructed.

## The command used for the released model

```bash
python train/train_sft.py \
  --init_model_path <Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice snapshot> \
  --spk_row0 3020 --preset_speakers vivian,serena,uncle_fu --speaker_table speaker_table.pt \
  --train_jsonl train.jsonl --val_jsonl val.jsonl \
  --mode full --full_dtype fp32 --optimizer adamw --lr 5e-6 --weight_decay 0.01 \
  --epochs 2 --warmup_ratio 0.03 --min_lr_ratio 0.1 --batch_size 16 --effective_batch 32 \
  --grad_clip 1.0 --sub_talker_weight 0.3 --label_alignment fixed --p_auto_lang 0.25 \
  --max_frames 400 --save_at 0.5,0.75,1.0,1.25 --stop_after 1.25 --attn sdpa
```

- **Data:** 19,829 character lines plus 1,491 replay clips.
- **Training time:** 13 minutes on one H100 80GB.
- **Total cost:** about $12 for the whole study, including the arms that were not selected.
- **Full queue:** `pod/run_t1.sh`.
- **Smaller GPUs:** `--mode lora` is the trainer's low-memory default, written for a 16 GB card. The released model is a full fine-tune.

## Related

- [Build your own character voice model](../README.md#build-your-own-character-voice-model): the full recipe, from game files to a selected checkpoint.
- [FAQ](../README.md#faq)
- [Research report](../research/voice_model_approach_2026-10.md): model choice, emotion control, evaluation and production effects.
