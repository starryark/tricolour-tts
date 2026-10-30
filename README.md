# Tricolour Lovestory (三色绘恋) character voice TTS: multi-speaker Qwen3-TTS fine-tuning with emotion control

**Reference-free Mandarin text-to-speech in the voices of 14 *Tricolour Lovestory* (三色绘恋) characters, made by fine-tuning Qwen3-TTS on the games' own voice lines, with emotion instructions.**

This repository is the complete, reproducible recipe, from extracted game voice tracks to a selected checkpoint:
- data building and verification;
- effect screening;
- a fixed multi-speaker Qwen3-TTS trainer with instruction support;
- evaluation under a pre-registered selection rule;
- the research notes behind each choice.

It is an unofficial fan project, with no affiliation to the games' developers, publishers or voice actors.

| | |
|---|---|
| **Try it** | [Demo on Hugging Face Spaces](https://huggingface.co/spaces/starrydark/Tricolour_Lovestory_TTS): pick a character, type text, choose an emotion |
| **Model** | [starrydark/Tricolour_Lovestory_Qwen3_TTS](https://huggingface.co/starrydark/Tricolour_Lovestory_Qwen3_TTS) |
| **Dataset** | [starrydark/Tricolour_Lovestory_Voice](https://huggingface.co/datasets/starrydark/Tricolour_Lovestory_Voice): 21,665 voice lines, 22.8 h |
| **Fine-tuning guide** | [Four silent bugs in the official Qwen3-TTS fine-tuning script, and multi-speaker / instruction training](docs/qwen3-tts-finetuning-fixes.md) |
| **Results** | [`research/T1_results.md`](research/T1_results.md) |

**Also known as:**
- Tricolor Lovestory;
- Sanse Huilian;
- 三色绘恋S *Sunnyrain Lovestory*;
- 叶心 *Heartful Memories*.

**Characters:**
- **Main cast:** 墨小菊 Mo Xiaoju, 文芷 Wen Zhi, 骆衍 Luo Yan, 迟菓 Chi Guo, 迟耀 Chi Yao, 叶心 Ye Xin.
- **Side characters:** 朱特 Zhu Te, 墨叔 Mo Shu, 黄丽丽 Huang Lili, 丁老师 Ding Laoshi, 琳姐 Lin Jie, 文绉 Wen Zhou, 徐阿姨 Xu Ayi, 刺儿头 Ci'ertou.

## 中文简介

《三色绘恋》角色语音合成（TTS）：用游戏原声台词微调 Qwen3-TTS，得到 14 位角色的中文 AI 语音模型：

- **调用方式：** 按角色名直接合成，无需参考音频，不是声音克隆。
- **情感指令：** 支持「生气 / 伤心 / 开心」（盲听测试中生气、伤心效果明确，开心较弱）。
- **本仓库内容：** 完整流程，包括数据整理与校验、录音特效（电话音、回声）筛查、多角色（多说话人）微调代码，以及预先登记的评测。
- **官方微调脚本问题：** 修复了官方 `sft_12hz.py` 的四个隐性问题，见 [微调说明](docs/qwen3-tts-finetuning-fixes.md)。
- **在线试听：** [Hugging Face Space](https://huggingface.co/spaces/starrydark/Tricolour_Lovestory_TTS)。
- **模型与数据集：** [模型](https://huggingface.co/starrydark/Tricolour_Lovestory_Qwen3_TTS)；[数据集](https://huggingface.co/datasets/starrydark/Tricolour_Lovestory_Voice)（21,665 条台词，22.8 小时）。

**角色：** 墨小菊、文芷、骆衍、迟菓、迟耀、叶心、朱特、墨叔、黄丽丽、丁老师、琳姐、文绉、徐阿姨、刺儿头。

**适合谁：** 想为游戏、动画或视觉小说（galgame）角色训练专属语音模型的同好，可以直接参考本仓库流程。

非官方粉丝项目，与游戏官方及配音演员无关；角色与声音权利归原权利人所有。

## Use the model

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts

tts = Qwen3TTSModel.from_pretrained("starrydark/Tricolour_Lovestory_Qwen3_TTS",
                                    device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(text="明天上午九点，我们在图书馆门口集合。",
                                     speaker="chiguo", language="Chinese",
                                     instruct="用生气的语气说")   # optional; or 用伤心的语气说 / 用开心的语气说
sf.write("chiguo.wav", wavs[0], sr)
```

The speaker ids are the pinyin names: `moxiaoju`, `wenzhi`, `luoyan`, `chiguo`, `chiyao`, `yexin`, `zhute`, `moshu`, `huanglili`, `dinglaoshi`, `linjie`, `wenzhou`, `xuyayi` and `ciertou`. Per-character quality is on the [model card](https://huggingface.co/starrydark/Tricolour_Lovestory_Qwen3_TTS#characters).

## Build your own character voice model

The pipeline generalises to any game, anime or audio drama where each character has isolated voice lines with known text. Two steps read these games' own formats: `b01` (the Kirikiri and Unity/Utage engines) and `b03` (sprite faces). The other steps are generic.

1. **Pair audio with text from the source, not from alignment.** Game engines reference each voice file from a script line, which gives clip, text and speaker for free (`build/b01_build.py`).
2. **Pick characters with enough speech.**
   - Here the threshold was 12 usable minutes.
   - Side characters with 10–27 minutes of training data reached 0.87–1.00 identity.
   - The exception, 黄丽丽 (22 minutes, one game only), reached 0.47. Amount is not the only factor.
3. **Normalise the text, re-cut the edges and level the audio** per (game, character), so whispers stay quieter than shouts (`b01`).
4. **Transcribe every clip twice, with unrelated ASR systems.**
   - Here: Qwen3-ASR and Paraformer, compared in toneless pinyin.
   - Drop clips whose audio does not say their line (`b02`, `b04`). That removed 186 clips here.
5. **Screen production effects.**
   - Phone, radio and echo takes are about 4% of game lines (`b07`–`b09`).
   - Keep them out of training and, above all, out of the clips that define a speaker's vector.
   - Add those effects back with DSP after generation.
6. **Split by script block,** so no test line shares a scene with training (`b06`).
7. **Encode the audio to Qwen3-TTS codec codes** (`b10`) and **build one fixed speaker vector per character** (`train/make_speaker_table.py`).
8. **To keep emotion control, start from CustomVoice and add instruction replay:** clips the untrained model generates with its own presets under emotion instructions (`eval/rp_generate.py`).
9. **Train with the fixed trainer** (`train/train_sft.py`; see the [fine-tuning guide](docs/qwen3-tts-finetuning-fixes.md)). The released model took 13 minutes on one H100. The whole study cost about $12.
10. **Judge with models that are not part of the system.**
    - **Identity:** CAM++ nearest-centroid accuracy.
    - **Words:** Paraformer pinyin CER.
    - **Selection:** pick the checkpoint by a rule written down before training (`eval/judge.py`, `eval/score_t1.py`).
    - **Emotion:** test it with blind human A/B pairs (`eval/ab_*`), not with a speech-emotion model.

## Layout

```
build/   data: select, re-cut, transcribe twice, verify, label, screen effects, splits, codec codes, HF export
  b01_build.py            14-character pool, text normalisation, re-cut and levelling (24 kHz)
  b02_asr.py, b04_verify  Qwen3-ASR + Paraformer transcripts, compared in toneless pinyin
  b03_faces.py            TRI sprite face code per line (from the KAG scripts)
  b05_emotion_audio.py    emotion2vec+ scores (shown to be unreliable on this cast)
  b06_manifest.py         manifest and block-level test split
  b07*/b08*.py            channel features: speaker-relative LTAS phone test, AudioSet tags
  b09_train_sets.py       effect exclusions, screened speaker references, trainer rows
  b10_codes.py            Qwen3-TTS 12 Hz codec codes
  b11_hf_dataset.py       the public dataset export
train/   trainer and its checks
  train_sft.py            full / LoRA fine-tune of Qwen3-TTS (fixed label alignment, sub-talker
                          frame, text_projection; instruction prefix; preset and locked speaker rows)
  test_equivalence_tri.py trainer inputs == generate_custom_voice inputs, position by position
  loss_probe.py, smoke_local.py, make_speaker_table.py, make_suites.py, make_step0.py, gen_eval.py
eval/    judges and studies
  judge.py, score_t1.py   CAM++ identity, Paraformer pinyin CER, emotion2vec, LTAS leak; T1 gates
  rp_generate.py/score    instruction replay (R) and character-voiced pseudo-data (P)
  d2_*, d4_*, fx_*, ab_*  experiments and blind listening pages (emotion labels, effects, A/B)
pod/     RunPod: bootstrap with an on-pod equivalence gate, the T1 queue, a launcher with a watchdog
docs/    the Qwen3-TTS fine-tuning guide
research/  the approach report, pre-registrations, results and literature notes
```

The scripts default to the author's local folders; `TRI_DATASET`, `TRI_WORK` and `TRI_POD_WORK` override them. The source audio is not in this repository. The dataset on Hugging Face holds the processed clips, but not the raw game extracts that `b01` starts from.

## Findings worth knowing if you train your own

- **The official Qwen3-TTS fine-tuning script has silent bugs.**
  - Labels are shifted twice.
  - The sub-talker sees the frame it predicts.
  - `text_projection` is skipped.
  - The speaker encoder decays under weight decay.

  `train/train_sft.py` fixes them, and `train/test_equivalence_tri.py` proves its inputs match inference. Details, with the upstream code and issue links: [docs/qwen3-tts-finetuning-fixes.md](docs/qwen3-tts-finetuning-fixes.md).
- **For a CustomVoice init, write new speaker rows clear of the presets.** `uncle_fu` sits at row 3010; this trainer uses 3020+ and keeps the presets in the saved config.
- **Fine-tuning CustomVoice on uninstructed data with instruction replay keeps emotion instructions working.**
  - It also stops characters drifting to a generic voice under instructions: identity under instruction went from 0.74 to 0.94.
  - Character-voiced emotional pseudo-data brings back stronger emotion, but brings the drift back too.
- **emotion2vec is not a usable judge for acted Mandarin game speech.**
  - Against human labels, its non-neutral precision was 0.08–0.40.
  - It reported "sad" as lost when a blind listener heard it in 16/16 pairs.
- **Production effects (phone, echo) are about 4% of the lines.**
  - Left in the references, they poison a speaker's mean vector.
  - They are better applied with DSP after generation.

## FAQ

**Can I hear the voices without installing anything?**
Yes. The [Hugging Face Space](https://huggingface.co/spaces/starrydark/Tricolour_Lovestory_TTS) runs the model on ZeroGPU in the browser.

**Is this voice cloning? Does it need a reference clip?**
No. Each character is a fixed speaker vector stored in the model and called by name (`speaker="wenzhi"`). No reference audio is used at inference.

**Which emotions work?**
`用生气的语气说` (angry) and `用伤心的语气说` (sad) were clearly followed in a blind A/B test (14/16 and 16/16 pairs). `用开心的语气说` (happy) is weaker (12/16). Leaving out `instruct` gives the character's natural delivery.

**Can I fine-tune Qwen3-TTS on several speakers at once?**
Yes, with `train/train_sft.py`. The official script trains one speaker and saves it to a single row. This trainer saves any number of speakers to consecutive codec rows and keeps the CustomVoice presets. See the [fine-tuning guide](docs/qwen3-tts-finetuning-fixes.md).

**How do I keep emotion and style instructions working after fine-tuning?**
1. Start from `Qwen3-TTS-12Hz-1.7B-CustomVoice`, not Base.
2. Mix in about 1,500 instruction-replay clips: the untrained model's own presets speaking under instructions.

Without replay, instruction-following erodes. Character-voiced emotional pseudo-data restored emotion but pulled voices off-character (identity under instruction 0.48), so it was rejected.

**How much audio does each character need?**
Twelve usable minutes was the threshold here, and most side characters with 10–27 minutes worked well (identity 0.87–1.00). Speaker similarity, recording consistency and coverage matter as well: 黄丽丽 has 22 minutes from one game and reached only 0.47.

**Can I label emotions in game voice data with emotion2vec or another speech-emotion model?**
Not reliably on acted Mandarin game speech. Against 310 human labels, emotion2vec+ large was right about neutral (precision 0.94), but its angry, happy, sad and surprised calls were right only 8–40% of the time. Use blind human A/B tests to judge emotion.

**What about lines with phone, echo or radio effects?**
They are about 4% of the lines. Drop them from training and from speaker references, then apply the effect with DSP after generation. This project applies effects with DSP rather than learning them as extra speaker rows; the reasoning is in `research/voice_model_approach_2026-10.md` §15.

**What hardware does it need, and what did it cost?**
- **Inference:** a CUDA GPU. The model was developed and tested on a 16 GB consumer card, in bf16.
- **Training:** one H100 80GB, about 3 hours for the whole study, evaluation included: about $12. The released model's own training took 13 minutes.

**Why does the dataset have 22.8 h but the model card says 20.9 h of training data?**
- The dataset holds all 21,665 clips, including the validation, test and effect-flagged ones.
- Training used the 19,829 clean train-split lines plus 1,491 synthetic instruction-replay clips: 20.9 h together.

**Can I use the voices commercially?**
No rights are granted to the voices, characters or game audio: they belong to their owners. The code in this repository is Apache-2.0 and can be reused for your own data.

## License

The code is Apache-2.0 (see `LICENSE`). Parts of `train/train_sft.py` derive from the Apache-2.0 Qwen3-TTS fine-tuning scripts (© Alibaba Qwen team).

The model weights and the dataset are separate releases with no rights grant: the voices and game content belong to their owners. See their cards.

## Citation

```bibtex
@misc{tricolour_qwen3_tts_2026,
  title  = {Tricolour Lovestory character voices: a Qwen3-TTS fine-tune},
  author = {starrydark},
  year   = {2026},
  howpublished = {\url{https://github.com/starryark/tricolour-tts}}
}
```
