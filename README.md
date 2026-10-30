# tricolour-tts

The code and research behind the [Tricolour Lovestory Qwen3-TTS model](https://huggingface.co/starrydark/Tricolour_Lovestory_Qwen3_TTS): a reference-free Mandarin text-to-speech model with 14 character voices from the *Tricolour Lovestory* (三色绘恋) visual novels, and emotion instructions. The pipeline runs end to end, from extracted game voice tracks to a selected checkpoint:
- data building and verification;
- effect screening;
- multi-speaker fine-tuning of Qwen3-TTS with instruction support;
- evaluation under a pre-registered selection rule.

| | |
|---|---|
| Model | [starrydark/Tricolour_Lovestory_Qwen3_TTS](https://huggingface.co/starrydark/Tricolour_Lovestory_Qwen3_TTS) |
| Demo | [starrydark/Tricolour_Lovestory_TTS](https://huggingface.co/spaces/starrydark/Tricolour_Lovestory_TTS) |
| Dataset | [starrydark/Tricolour_Lovestory_Voice](https://huggingface.co/datasets/starrydark/Tricolour_Lovestory_Voice) |
| Results | [`research/T1_results.md`](research/T1_results.md) |

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
research/  the approach report, pre-registrations, results and literature notes
```

The scripts default to the author's local folders (`TRI_DATASET`, `TRI_WORK` and `TRI_POD_WORK` override them). The source audio is not in this repository. The dataset on Hugging Face holds the processed clips, but not the raw game extracts that `b01` starts from.

## Findings worth knowing if you train your own

- **The upstream Qwen3-TTS fine-tuning script has silent bugs:** labels are shifted twice, the sub-talker sees the frame it predicts, and `text_projection` is skipped. `train/train_sft.py` fixes them, and `train/test_equivalence_tri.py` proves its inputs match inference.
- **For a CustomVoice init, write new speaker rows clear of the presets.** `uncle_fu` sits at row 3010; this trainer uses 3020+ and keeps the presets in the saved config.
- **Fine-tuning CustomVoice on uninstructed data with instruction replay keeps emotion instructions working.** It also stops characters drifting to a generic voice under instructions: identity under instruction went from 0.74 to 0.94. Character-voiced emotional pseudo-data brings back stronger emotion but brings the drift back too.
- **emotion2vec is not a usable judge for acted Mandarin game speech.** Against human labels its non-neutral precision was 0.08–0.40, and it reported "sad" as lost when a blind listener heard it in 16/16 pairs.
- **Production effects** (phone, echo) are about 4% of the lines. They poison a speaker's mean vector if left in the references, and are better applied with DSP after generation.

## License

The code is Apache-2.0 (see `LICENSE`). Parts of `train/train_sft.py` derive from the Apache-2.0 Qwen3-TTS fine-tuning scripts (© Alibaba Qwen team).

The model weights and the dataset are separate releases with no rights grant: the voices and game content belong to their owners. See their cards.
