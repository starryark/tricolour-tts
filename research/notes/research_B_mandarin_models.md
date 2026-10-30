# Mandarin re-rank of open TTS bases for the 三色绘恋 cast (state as of 2026-10-29)

Scope: re-rank the candidates from `E:\STEINS;GATEVoice\kurisu_v2\research\sota_models_2026-10.md` (the "local survey") for a **native-Mandarin, multi-character, expressive galgame** target. The data is 23.5 h of clean studio lines with scripts: 6 mains at 2–6 h each, about 8 sides at 15–30 min, and the rest a few minutes. Japanese and cross-lingual requirements are dropped. Facts the local survey already verified are cited as [L] and were not re-fetched.

**Tags**
- **[P]** I read it in a primary source: a repo README, code, issue, HF card or API, or an arXiv paper.
- **[S]** Secondary: a blog, Zhihu or CSDN page, or a search-result snippet.
- **[I]** My inference.
- **[U]** Unverified.
- **[L]** From the local survey (2026-10-28).

All sources were accessed on 2026-10-29.

---

## 0. Answer in brief

1. **The Japanese-driven reasons for Qwen3-TTS disappear.** Its JA CER, ja→zh CER, and the only published evidence that a single-speaker fine-tune stays multilingual (all [L]) do not matter for a Mandarin-only cast. What is left is good Mandarin CER (test-zh 0.77 in its own report [P]) and *weaker* Mandarin speaker similarity than its main rivals:
   - **Strongest independent evidence:** in the IndexTTS-2.5 harness (third-party authors, same test, same SIM model), CV3-zh SS is 73.0 for Qwen against **80.0 for the open CosyVoice3-0.5B** [P].
   - Seed test-zh SIM is 77.0 against 78–81 for CosyVoice3, dots.tts, VoxCPM2, FireRedTTS3 and LongCat. This number comes from third-party tables with no stated origin [P/U].
   - In the Qwen-Audio-3.0 paper's CV3 hard-zh table, Qwen (69.1) and VoxCPM2 (69.9) trail dots.tts (77.5) and LongCat (72.8) on SIM. The open CosyVoice3-0.5B was not measured there; the 78.5 belongs to the unreleased 1.5B [P].
   - **Qwen3-TTS-CustomVoice has no Seed-TTS-eval numbers anywhere.** Its only published zh numbers are target-speaker WER 0.903, long-zh 2.356 and InstructTTSEval-zh [P].
2. **Re-ranked (on paper):**
   1. **Fun-CosyVoice3-0.5B-2512, multi-speaker SFT.** It is Mandarin-first. It has the best open test-hard CER (RL 5.44, own harness) [P], built-in instruct (emotion, speed, volume, dialect) [P], speaker-prompt multi-speaker SFT by design [P, CosyVoice2 §2.7], and the only published recipe that keeps instruction following after speaker SFT [P, CV3 §2.6]. An `inference_sft(spk_id)` code path can drive inference from a stored speaker embedding with no prompt audio [P code].
      - **The two operational selling points are described methods and inherited code paths, not shipped, tested recipes.** One is reference-free speaker IDs. The other is instruct surviving SFT. Nobody has published a CosyVoice3 mSFT model driven this way [U].
      - The streaming-text variant of `inference_sft` crashed on CosyVoice3 in #1731 (Dec 2025). A community PR, #1700, reportedly fixes it; its merge status is unverified [P].
      - Fallback for reference-free use: `add_zero_shot_spk`, which stores a prompt clip once [P].
      - Other risks: stability complaints about the 0.5B (#1703) [P]. The public recipe is LibriTTS-style, so you must write the speaker and instruct prefixes, build `spk2info`, and source instruction-replay data yourself [P/I].
   2. **Qwen3-TTS-12Hz-1.7B-Base.** You already have a working trainer [L]. Named-speaker inference needs no reference audio [P].
      - It has no emotion control after fine-tuning [L, P #14], and official fine-tuning is single-speaker only [P].
      - It is the right choice if the bake-off shows its fine-tuned similarity and emotion match CosyVoice3's, because the engineering cost is the lowest.
   3. **VoxCPM2, one LoRA per character.** It is the easiest documented path: LoRA, hot-swap, inference from text alone [P]. Weaknesses: hard-zh SIM 69.9–75.3 [P] and a low English EmergentTTS emotion score (42.3%) [P].
   4. **dots.tts (challenger).** It has the highest open Mandarin SIM (Seed zh 81.0, hard 79.5) [P] and the best open EmergentTTS emotion score [P], under Apache-2.0 with official training code. But its manifest has no speaker field, so every character means a separate 2B checkpoint, or reference-based cloning [P/I].
   - **Zero-shot bar, not bases:** IndexTTS-2.5, FireRedTTS3, LongCat-AudioDiT-3.5B and Fish S2 Pro. All have strong Mandarin numbers, and all still lack an official fine-tuning path (S2: old tag only, non-commercial licence) [P/L].
   - **Community baseline:** GPT-SoVITS v2ProPlus / V5. It always needs a reference clip at inference [P].
3. **Community practice (China, 2026)** [S]:
   - GPT-SoVITS (v2ProPlus) is still the default way to *train a fixed character voice*.
   - IndexTTS2 is the favourite for *zero-shot emotional* dubbing.
   - Qwen3-TTS character fine-tunes exist but are few. CosyVoice3 fine-tunes are rarer still.
   - I found no published measured comparison between these routes for Chinese character voices [U].
4. **Emotion control after a speaker fine-tune:**
   - Only CosyVoice3 has a documented recipe (instruction replay plus speaker/style prompt masking) [P].
   - For Qwen there is no evidence that a new speaker can be fine-tuned into CustomVoice or VoiceDesign with instruct intact. The maintainer's answer (2026-01-23) was a future "VoiceEditing" model, "without the need for fine-tuning", which was never released [P, L].
   - With IndexTTS2 or GPT-SoVITS, emotion comes from reference or emotion prompts at inference [P/S].
5. **Zero-shot vs fine-tune:** there is no published comparison for expressive character voices [U]. Proxies:
   - CosyVoice2 single-speaker SFT: SS 0.721 → 0.795, NMOS 3.91 → 3.96, but WER 5.34 → 7.15 [P].
   - Baseten's Qwen3-TTS run (1.5 h, English) found fine-tuning "didn't materially beat zero-shot on similarity or MOS-style quality", but expressiveness improved where cloning "sounds flat" [P blog].
   - With 2–6 h per main character, fine-tuning should win on range and consistency, but this has to be measured [I].

---

## 1. Q1: Mandarin zero-shot quality

Seed-TTS-eval test-zh and test-hard use a character-level error rate. Some papers label it "WER" and some "CER"; it is the same metric. Speaker-similarity models also differ between papers (WavLM-large vs ERes2Net), so **compare only within a column**. Many "baseline" rows in vendor tables are copied from the original papers, not re-run [I].

### 1.1 Seed-TTS-eval test-zh / test-hard (CER % / SIM)

| Model (open checkpoint unless noted) | Own report | dots.tts README (H2) | FireRedTTS3 README (H3) | LongCat card (H4) | Qwen-Audio-3.0 paper Tbl 3 (H5) |
|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base** | zh **0.77** (no SIM, no hard) [P README/report] | 1.22 / 77.0; hard 6.76 / 74.8 | 1.22 / 77.0; hard 6.76 / 74.8 | 1.22 / 0.770; hard 6.76 / 0.748 | 0.77 / – |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | **No Seed-TTS-eval numbers anywhere.** Own: target-speaker zh WER 0.903; long-zh 2.356; InstructTTSEval-zh 83.0 / 77.8 / 61.2 [P] | – | – | – | – |
| **Fun-CosyVoice3-0.5B-2512** | 1.21 / 78.0; hard **6.71 / 75.8** [P CosyVoice README] | – | – | – | – |
| **Fun-CosyVoice3-0.5B-2512_RL** | 0.81 / 77.4; hard **5.44 / 75.0** [P] | – | – | – | – |
| CosyVoice3-1.5B (**not released**) | 0.71 (Qwen table) | 1.12 / 78.1; hard 5.83 / 75.8 | same | 1.12 / 0.781; hard 5.83 / 0.758 | 1.12 / 0.781 (0.837 ERes2Net) |
| CosyVoice 3.5 (API only [L]) | – | – | – | 0.87 / 0.797; hard 5.71 / 0.786 | – |
| **VoxCPM2** (2B) | 0.97 / 79.5; hard 8.13 / 75.3 [P VoxCPM README] | same | same | (VoxCPM1: 0.93 / 0.772; hard 8.87 / 0.730) | 0.97 / 0.795 (0.756) |
| **dots.tts SOAR** (2B) | 0.94 / **81.0**; hard 6.60 / **79.5** [P] | — | Pretrain 0.97 / 80.4; hard 6.65 / 78.8 | – | 0.94 / 0.810 (0.818) |
| **FireRedTTS3-Base** | 1.01 / **80.9**; hard 6.50 / 78.4 [P] | – | — | – | – |
| **LongCat-AudioDiT-3.5B** (zh/en) | 1.09 / **0.818**; hard 6.04 / **0.797** [P card] | – | – | — | – |
| LongCat-AudioDiT-1B | 1.18 / 0.812; hard 6.33 / 0.787 [P] | – | – | — | – |
| **IndexTTS-2** (1.5B) | 1.008 / 0.865 (own SS model) [P paper] | 1.03 / 76.5; hard 7.12 / 75.5 | same | 1.03 / 0.765; hard 7.12 / 0.755 | – |
| IndexTTS-2.5 | Not on Seed-TTS-eval; CV3 only (§1.2) [P] | – | – | – | – |
| **Confucius4-TTS** | zh 0.94 / 0.765 (no hard) [P README] | – | – | – | – |
| **Fish Audio S2 (Pro)** | zh **0.54**; hard 5.99 (no SIM) [P arXiv 2603.08823] | – | – | – | – |
| Fish S2 (Confucius4 harness) | – | – | – | – | 0.98 / 0.737 [P Confucius4 README] |
| **Ming-3.0-tts-0.5B / tiny** | 0.80 / 0.758; tiny 0.82 / 0.781 (no hard) [P card] | – | – | – | – |
| **Spark-TTS** (0.5B) | 1.20 / 66.0 (CosyVoice table) [P]; 2.636 / 0.683 in the IndexTTS2 harness [P] | – | – | – | – |
| **MegaTTS3** | – | – | 1.52 / 79.0 | – | – |
| **MOSS-TTS** (TTSD sibling) | 8B Delay 1.37 / 76.98; 1.7B Local 1.44 / 79.62 [P MOSS-TTS README] | – | – | – | – |
| **MOSS-TTSD v1.0** | No Seed numbers. TTSD-eval zh: SIM 0.795, speaker-attribution acc 0.959, WER 4.85% [P] | – | – | – | – |
| **GPT-SoVITS** (version unstated) | No Seed numbers. CV3-Eval zh CER **7.34** against CosyVoice3-0.5B 3.89 [P CV3 paper Tbl 5] | – | – | – | – |
| Seed-TTS (closed reference) | 1.12 / 79.6; hard 7.59 / 77.6 | same | same | 1.12 / 0.796; hard 7.59 / 0.776 | – |

**Reading the table** [I]:
- On test-zh, everything strong sits at 0.8–1.2 CER, which is saturated.
- **Test-hard and SIM separate the models.**
  - The best open CER on hard is CosyVoice3-0.5B-RL (5.44, own) and S2 (5.99, own).
  - The best open SIM is LongCat-3.5B (0.818 / 0.797), dots.tts (81.0 / 79.5) and FireRedTTS3 (80.9 / 78.4).
  - Qwen3-TTS is mid-pack on hard CER (6.76) and near the bottom on SIM (77.0 / 74.8).
- The third-party Qwen number (1.22 / 77.0, hard 6.76 / 74.8) appears identically in three tables. Its original measurement source is not stated [U]. Qwen's own 0.77 is the only first-party figure.

### 1.2 CV3-Eval (in-the-wild) Chinese

| Model | CV3 zh CER/WER (QA3 Tbl 4) [P] | CV3 zh WER / SS (IndexTTS-2.5 README) [P] | CV3 zh WER (dots README) [P] | CV3 hard-zh CER / SIM (QA3 Tbl 5) [P] | Own paper |
|---|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base | **3.09** | 3.27 / 73.02 | – | 11.24 / **69.1** | – |
| Fun-CosyVoice3-0.5B | 3.89 | 3.84 / **80.01** | – | – | 3.89 (CV3 paper) |
| CosyVoice3-1.5B (unreleased) | 3.91 | 3.91† | 3.91; hard 9.77 | 9.77 / 78.5 | – |
| VoxCPM2 | 3.55 | 3.88 / 74.99 | 3.65; hard 8.55 | **8.10** / 69.9 | 3.65; hard 8.55 |
| dots.tts SOAR | 3.58 | – | 3.71; hard 9.22 | 11.75 / 77.5 | – |
| LongCat-AudioDiT-3.5B | – | – | – | 9.24 / 72.8 | – |
| IndexTTS-2.5 / -RL | – | 4.36 / 77.10; RL 3.93 / 77.92 | – | – | – |
| Fish S2 Pro | – | 3.62 / 67.79 | **2.65**; hard 9.10 | – | 2.65 |
| OmniVoice | – | 3.41 / 72.99 | – | – | – |
| MOSS-TTS 1.5 (8B) | – | 4.02 / 72.68 | – | – | – |
| GPT-SoVITS | – | – | – | – | 7.34 (CV3 paper) |
| MiniMax-2.8-HD (closed) | 3.42 | – | – | 7.42 / 74.6 | – |
| Qwen-Audio-3.0-TTS (closed) | 3.35 | – | – | 7.44 / 78.7 | – |

**What this table shows** [I]:
- **IndexTTS-2.5's harness** (third-party) gives the open CosyVoice3-0.5B the highest zh SS (80.0) and Qwen3-TTS 73.0. This is the cleanest like-for-like SIM comparison found.
- **QA3 is Qwen's own paper.** In its hard-zh table, Qwen3-TTS (69.1) and VoxCPM2 (69.9) trail dots.tts (77.5) and LongCat (72.8) on SIM. The open CosyVoice3-0.5B is not in that table.

Both point the same way: on Mandarin, Qwen3-TTS clones *intelligibly* but with *less timbre fidelity* than CosyVoice3 and dots.tts.

### 1.3 Emotion, expressiveness and instruction benchmarks

| Benchmark (language, harness) | Results | Tag |
|---|---|---|
| **IndexTTS2 emotional test set** (own; WER / SS / emotion-sim ES / EMOS) | IndexTTS2 1.883 / 0.836 / **0.887 / 4.22**. MaskGCT 4.059 / 0.810 / 0.841 / 3.37. CosyVoice2 1.831 / 0.803 / 0.802 / 3.09. Spark 2.299 / 0.673 / 0.832 / 3.16. F5 3.053 / 0.773 / 0.757 / 3.16 | [P arXiv 2506.21619 Tbl 2] |
| IndexTTS2 text-emotion vs CosyVoice2 instruct (EMOS) | 3.786 vs 3.339 | [P Tbl 3] |
| **CV3-Eval emotion cloning** (zh+en pooled, emo2vec accuracy; text-unrelated happy / sad / angry) | CosyVoice3-0.5B 0.64 / 0.42 / 0.58. CosyVoice3-1.5B 0.64 / 0.44 / 0.48. +DiffRO-EMO (unreleased) 0.98 / 0.50 / 0.68. CosyVoice2 0.56 / 0.44 / 0.38. Spark 0.50 / 0.60 / 0.36. **GPT-SoVITS 0.48 / 0.40 / 0.30**. F5 0.80 / 0.28 / 0.64 | [P CV3 paper Tbl 9] |
| CV3 emotion control (Ming card harness; language not stated) | Ming-3.0-0.5B 72.33%, tiny 74.00%, CosyVoice3-1.5B+DiffRO-EMO 77.67%. Hard set (CV3N): CosyVoice3-0.5B **40.00%** (leads), Ming 37.33% | [P Ming card] |
| **InstructTTSEval-ZH** (APS / DSD / RP; Gemini judge) | FireRedTTS3-Instruct **85.8 / 82.0 / 69.7**. Qwen3-VD 85.2 / 81.1 / 65.1. Qwen3-CustomVoice-12Hz 83.0 / 77.8 / 61.2. VoxCPM2 85.2 / 71.5 / 60.8. Ming-3.0 voice-design avg 72.3 (0.5B) / 75.4 (tiny) against Qwen3 77.1 | [P READMEs / cards] |
| QA3 zh instruction following (emotion, rate, style; Tbl 10) | CosyVoice3-1.5B **75.91**, IndexTTS2 54.39 (closed Qwen-Audio-3.0: 78.94) | [P arXiv 2607.23938] |
| **EmergentTTS-Eval, Emotions** (English, win-rate vs gpt-4o-mini-tts, dots harness) | dots Pretrain 72.7, SOAR 63.9, Qwen3 (Ryan preset) 60.5, **Qwen3 clone 39.8**, VoxCPM2 42.3 | [P dots README] |
| EmergentTTS-Eval, Emotions (English, own) | Fish S2 **86.61** | [P arXiv 2603.08823] |
| Fish Instruction Benchmark (own, Gemini 3 Pro judge) | S2 TAR 0.984, Naturalness 4.40, Expressiveness 4.94; language mix unclear | [P / U] |

**Caveats** [I]:
- InstructTTSEval-zh measures voice-design and instruction following, not emotional range *within a cloned voice*.
- EmergentTTS is English and judge-based.
- The only Mandarin-relevant emotion-*cloning* evidence comes from IndexTTS2's own set (strongly in its favour) and CV3-Eval's pooled zh+en emotion subset. In the latter (text-unrelated), CosyVoice3-0.5B beats GPT-SoVITS on all three emotions and beats CosyVoice2 and Spark on happy and angry. It trails F5 on happy and angry, and trails Spark (and, marginally, CosyVoice2) on sad.

---

## 2. Q2: Fine-tuning path for multi-speaker Mandarin (as of 2026-10-29)

| Candidate | Official SFT / LoRA code | Several speakers in one model? | Inference without reference audio after fine-tuning? | Licence |
|---|---|---|---|---|
| **Qwen3-TTS-1.7B-Base** | `finetuning/sft_12hz.py`, JSONL `audio`/`text`/`ref_audio`, plus `--speaker_name` [P]. Known bugs: #278/#337 [L], #179 (speech gets faster each epoch, open), #343 (short utterances unstable after fine-tuning) [P titles] | **No.** README: "currently supports single-speaker fine-tuning… Multi-speaker… will be supported in future releases" [P]. The speaker embedding uses one fixed codec-embedding slot (index 3000) [S vaghawan card]. TaoLive's 6-speaker zh CustomVoice shows in-house code can do it [P card] | **Yes:** `generate_custom_voice(text, speaker=…)` with no reference [P README] | Apache-2.0 [L] |
| Qwen3-TTS-CustomVoice / VoiceDesign | No official fine-tuning from these checkpoints. SpragAI applied a LoRA (r32) to 1.7B-CustomVoice for 9 English emotion tags on the **preset** speakers. Speaker rows were frozen, no new speaker, no evaluation [P card] | Presets only | Presets only | Apache-2.0 |
| **Fun-CosyVoice3-0.5B-2512** | `examples/libritts/cosyvoice3/run.sh`: embeddings, `speech_tokenizer_v3` tokens, parquet, then `torchrun` train of `llm`/`flow`/`hifigan` with DeepSpeed. Every utterance gets the fixed instruct prefix "You are a helpful assistant.<\|endofprompt\|>" [P] | **Yes, by design, but not packaged.** mSFT prepends "Speaker A<\|endofprompt\|>", lr 1e-5 [P CosyVoice2 §2.7]. The CV3 paper (§2.6) uses data "partially labeled with speaker IDs" with speaker and style prompts randomly masked [P]. The recipe emits `spk2embedding.pt`, not `spk2info.pt` [P]. Community 0.5B multi-speaker SFT `gglabs/cosyvoice3-multilingual-multispeaker-v3_ep49` (1–2 h per speaker, lr 1e-5, 49 epochs; ships a `spk2info.pt`) [S; HF page 401] | **Code path exists, untested for CosyVoice3** [P/U]. `inference_sft(tts_text, spk_id)` is inherited with no class guard. `frontend.py` has one shared frontend with no CosyVoice3 override, and `frontend_sft` returns only `text` plus `llm_embedding = flow_embedding = spk2info[spk_id]['embedding']`. `spk2info.pt` is loaded at init. You write that file yourself: one embedding per character, and the speaker/instruct prefix goes inside `tts_text` [P code / I]. The bistream variant raised `'CosyVoice3LM' has no attribute 'llm_embedding'` in #1731 (2025-12-24); community PR #1700 reportedly fixes it; the issue was bot-closed [P]. **Fallback:** `add_zero_shot_spk` + `save_spkinfo` stores a prompt once, so calls pass only `zero_shot_spk_id` (a stored clip, but no reference supplied at call time) [P]. For instruct on CosyVoice3, use `inference_instruct2`; `inference_instruct` asserts CosyVoice-v1 [P] | Apache-2.0 [P] |
| **VoxCPM2** | Official LoRA (r32/α32, lr 1e-4) and full (lr 1e-5) fine-tuning; ~20 GB / 40 GB VRAM [P docs] | No `speaker` field; identity comes only through `ref_audio` [P]. **One LoRA per character**, with `load_lora`/`unload_lora` hot-swap [P] | **Yes:** `model.generate(text=…)` [P] | Apache-2.0 [P] |
| **dots.tts** | `scripts/train_dots_tts.py` + `configs/dots_tts.yaml` ("smoke configuration"); manifest `{fid, audio, text}` [P] | **No speaker field.** "Random voice… only meaningful with a fine-tuned single-speaker checkpoint" [P]. So one 2B checkpoint per character, or a multi-speaker fine-tune used with x-vector/continuation cloning [I] | Only with per-character checkpoints [P/I] | Apache-2.0 [P] |
| **IndexTTS-2 / 2.5** | **None.** Issue #518 (2025-10-24) is still open with no visible replies [P, re-checked]. Community code, unverified: `instavar/indextts2-finetuning` ("first public training code", full SFT, 2★, 2026-08-14) and `SrQingChen/IndexTTS-2.5-Pro` (claims LoRA + DPO, 7★, 2026-10-26) [P listing]. HF language fine-tunes (vi 21 likes, kk, th, ko, ja-prosody) show third-party training works [P HF API] | Speaker comes from the prompt audio [P] | **No:** a speaker prompt is always required [P README] | bilibili Model Use License [P] |
| **FireRedTTS3** | **None.** The roadmap lists only model and report releases; none of the 6 open issues asks for training code [P, re-checked] | – | – | Apache-2.0 [P] |
| Confucius4-TTS | T2S and S2A training CLIs; TSV with language, audio, text, tokens, reference paths [P] | Not stated; speaker comes from reference via a learnable encoder [P/I] | No (reference-based) [I] | Apache-2.0 (code badge) [P] |
| LongCat-AudioDiT | Inference, weight conversion and model code only [P] | – | – | MIT [P] |
| Ming-3.0-tts | None on card [L/P] | – | – | MIT |
| Fish S2 Pro | The card says the release includes fine-tuning code [P]. The code exists only at tag v2.0.0-beta (removed in the Drama 3 beta), and the docs say LoRA learns "patterns, not timbre" [L] | – | – | Fish Audio Research License, non-commercial [P] |
| **GPT-SoVITS v2ProPlus / V5** | Full GPT + SoVITS training per character [P/L]. V5 weights `gsv-v5-pretrained/s2Gv5dev.pth`, `s2Gv5turbo.pth` and a new `vocoder.pth` (HF lastModified 2026-10-06); code on branch `cuda_graph_accel_v5` [P] | Per-character models (community practice) [S] | **No:** `ref_audio_path` is "(required)" in `api_v2.py` [P] | MIT [L] |
| MOSS-TTSD v1.0 (2026-02-10) | Full / LoRA / multi-node fine-tuning code for v0.5 (2025-07-16); not stated for v1.0 [P] | Dialogue model (speaker roles from prompts) | Reference-based [I] | Apache-2.0 [P] |
| Spark-TTS, MegaTTS3 | No 2026 update [L]; not pursued | – | – | – |

**Takeaways for a cast of about 14+ voices** [I]:
- **Of the shortlisted candidates, CosyVoice3 is the only one whose architecture expects several speakers in one SFT, selected by a text speaker prompt.** (MOSS-TTSD assigns speaker roles too, but it is a reference-based dialogue model.) That suits joint training, where side characters with 15–30 min borrow from the shared model.
- Qwen needs an extension to your trainer: N speaker slots, with a speaker index at inference.
- VoxCPM2 gives N LoRAs. That is cheap and isolated, and the side characters' small data fits its "5–50 clips" guidance [P docs].
- dots.tts gives N full 2B checkpoints.
- For characters with only a few minutes, a registered zero-shot speaker (CosyVoice3 `add_zero_shot_spk`) may beat any fine-tune, and it still needs no reference at call time [I].

---

## 3. Q3: Chinese community practice in 2026 (galgame, 原神/星铁, anime character voices)

Zhihu and 53ai pages returned 403/404 to the fetcher. Where marked, claims come from search snippets [S].

1. **GPT-SoVITS is still the default way to *train a fixed voice*.**
   - A 2026-01 selection report (53ai) says: "首选:GPT-SoVITS(需微调). 理由:微调后音色克隆质量目前无替代方案" [S snippet; page 404].
   - A Zhihu comparison: "GPT-SoVITS：零样本克隆基本不可用，必须长时间干声训练… 只训练固定情绪，难以自由切换情感" [S snippet; 403].
   - **Version:**
     - v2Pro/v2ProPlus is the current recommendation. The README says v2Pro has "Slightly higher VRAM usage than v2, surpassing v4's performance, with v2's hardware cost and speed", and that v3/v4 do badly on average-quality data and their timbre "lean[s] more toward the reference audio" [P README].
     - V5 is 3 weeks old (changelog entry 2026-10-05 "V5新版本"; weights 2026-10-06). It claims "Significantly improved voice similarity without SoVITS fine-tuning" and a new vocoder [P].
     - No V5 benchmarks or community evaluations found [U].
   - **Character packs** (AI-Hobbyist GPT-SoVITS models from 红血球AE3803's datasets; `baicai1145/GPT-SoVITS-STAR`, 52 Star Rail characters, "not tested yet") are GPT-SoVITS-based and mostly 2024–2025. AI-Hobbyist's HF repos were last modified 2024-11 [P HF API].
2. **IndexTTS2 is the favourite for zero-shot emotional dubbing.**
   - Gitee 一键整合包 exist [S].
   - An Arknights character cloning test (linux.do, via 80aj, 2025-12-17) judged IndexTTS2 "在语音自然度上表现更佳，接近原声效果" over CosyVoice3, while CosyVoice3 was much faster (~10 s vs ~1.5 min) [S].
   - Chinese guides: "情感控制选 IndexTTS2" (cnblogs, 2026-07-11, AI-generated aggregate) [S].
   - IndexTTS2 is slow on consumer GPUs: over 2 min per clip on a 4090 in liudon.com's AutoDL test (2026-05/09) [S].
3. **Qwen3-TTS character fine-tunes exist but are few.**
   - `mon3stera/Qwen3-tts-Mon3tr` (Arknights, zh, 20 epochs) [P HF].
   - `mnski/xiadie-qwen3tts` (zh) and `starrydark/Makise_Kurisu_Qwen3_TTS` [P HF API].
   - A CSDN walkthrough, galgame → Qwen3-TTS-1.7B-Base (2026-06-01): ~70 utterances, lr 1e-5, bs 2, 10 epochs, warns that "Epoch 不是越多越好… 背答案", inference via `generate_custom_voice` [S].
   - TaoLive's zh multi-speaker CustomVoice (6 speakers, pause and speed tokens, Apache-2.0, 2026-07-30) is the only multi-speaker Qwen fine-tune found [P].
   - Only ~50 public derivatives of 1.7B-Base on HF, mostly language adaptations [P HF API].
4. **CosyVoice3 fine-tunes are rare.**
   - HF derivatives of 0.5B-2512 are mostly MLX ports, plus one multi-speaker SFT (gglabs) [P HF API / S].
   - Huawei MindSpeed-MM added CosyVoice3 training on 2026-02-14 (Ascend) [S].
   - Alibaba Bailian sells cloud SFT for `cosyvoice-v3-flash`. It is single-speaker only: "混入多个发音人会降低音色还原度" [S help.aliyun.com].
   - Users report 0.5B zero-shot instability: repetition, prompt leakage. Fixes: the "You are a helpful assistant.<|endofprompt|>" header, fp16 off, 24 kHz prompts [P #1703].
5. **Other routes:**
   - Bert-VITS2 has no successor (Style-Bert-VITS2 v2.7.0 is the last) [L].
   - MoeTTS and VITS galgame packs are unmaintained [S].
   - Fish S2 Pro and VoxCPM appear in 2026 Chinese deploy-and-listen posts as zero-shot options. liudon.com recommends VoxCPM for ease of setup [S].
   - No AutoDL or codewithgpu fine-tuning images for Qwen3-TTS, CosyVoice3 or IndexTTS2 were found [U].
6. **Measured comparisons:**
   - None found that compare *fine-tuned* routes for Chinese character voices [U].
   - Blog "comparisons" are zero-shot listening tests or copied paper numbers [S].

---

## 4. Q4: Emotion control after a speaker fine-tune

| Candidate | Emotion/instruct after adding *your* speaker | Evidence |
|---|---|---|
| Qwen3-TTS-Base fine-tune | **None.** Emotion comes only implicitly from text and training data | [L #25] |
| Qwen3-TTS-CustomVoice / VoiceDesign + new speaker | **No evidence it works.** Issue #14 (2026-01-23) asked exactly this for 3–4 h of data. Collaborator wangxiongts said the capability would come via the VoiceEditing model "without the need for fine-tuning on additional data", then closed the issue. The 25 Hz / VoiceEditing weights were never released [P #14; L]. SpragAI shows a LoRA on CustomVoice can add emotion tags *for the presets* (English, no evaluation) [P]. Writing a new speaker into a CustomVoice codec-embedding row while replaying instruct data is untested [I/U] | |
| **CosyVoice3 SFT** | **Yes, by published recipe.** "Mix of target-speaker data and the pre-training instruction dataset… randomly mask the speaker prompt or the style prompt… helps prevent potential catastrophic forgetting in instructed generation" [P CV3 §2.6]. The pre-training instruction set is not public, so you would replay instruct-style samples you generate yourself, e.g. CosyVoice3 instruct outputs, or your lines auto-labelled with an emotion classifier [I] | |
| IndexTTS-2 / 2.5 | Emotion vectors (8 dimensions), an emotion reference clip (`emo_alpha`) and `use_emo_text`/`emo_text` are all inference-time and zero-shot. 2.5 needs `use_qwen_emo=True` for text emotion [P README]. Whether community fine-tunes keep the emotion adapter: [U] | |
| VoxCPM2 LoRA | `(style)` prefix / controllable cloning exists in the base [P]. Whether it survives LoRA is not documented [U]. The LoRA touches LM+DiT, so this is plausible but untested [I] | |
| dots.tts | No TTS instruct. `dots.tts.edit` accepts `<emo>`, `<pitch>`, `<rate>` edit tags on *existing* audio [P], so it could post-edit a fine-tuned voice's output [I/U] | |
| FireRedTTS3-Instruct | Best open InstructTTSEval-zh, but no fine-tuning [P] | |
| GPT-SoVITS | Emotion follows the chosen reference clip; common practice keeps per-emotion reference clips [S/I] | |
| Fish S2 Pro | Free-form `[tag]`s at inference [P]. Whether they survive fine-tuning: [U]. Fish's docs advise against fine-tuning the RL model [L] | |

**For acted galgame lines** [I]:
- Most emotional range comes from the data through the fine-tune: the script text carries the cue, and 2–6 h of acted lines span the range.
- Explicit control is a bonus. It matters when a line's text is ambiguous.
- Only CosyVoice3 offers both a speaker fine-tune and retained instruct as a documented path.

---

## 5. Q5: Zero-shot as the bar

- **No published comparison of zero-shot vs fine-tune for expressive Mandarin character voices was found** [U].
- **Proxies:**
  - **CosyVoice 2, Table 11** (one speaker, "Spk E"): zero-shot vs SFT on the in-house set [P arXiv 2412.10117]. The authors: "the SFT model shows higher speaker similarity and speech quality, however, the WER could be worse".
    - SS **0.721 → 0.795**.
    - NMOS 3.91 → 3.96.
    - WER 5.34 → 7.15.
    - Seed hard 6.83 → 7.90.
  - **CosyVoice 3, §5.7 / Fig. 5:** SFT lowers WER, "particularly… test-en and test-hard"; no numbers [P].
  - **Qwen3-TTS:** Base zero-shot multilingual-set zh WER 0.928 / SIM 0.799 against CustomVoice target-speaker zh WER 0.903 [P]. These are different test sets, WER only, so not a real comparison [I].
  - **Baseten** (2026-07-31; Qwen3-TTS, 1.5 h LJSpeech, English) [P blog]:
    - "didn't materially beat zero-shot on similarity or MOS-style quality".
    - Benefits are "operational (no reference audio, stable latency)" plus "better expressiveness on prompts where cloning sounds flat".
    - Fine-tuning has "room to improve as your dataset grows beyond a few hours".
    - No numbers.
  - **Community:** GPT-SoVITS zero-shot is "基本不可用" without training [S]. Users treat IndexTTS2 zero-shot as near-original for game characters [S].
- **Where the bar sits** [I]:
  - The strongest open zero-shot Mandarin similarity is about 0.80–0.82 (LongCat-3.5B, dots.tts, FireRedTTS3; Seed zh), with IndexTTS-2.5 adding explicit emotion control.
  - With 2–6 h per main character, a fine-tune should beat one-clip zero-shot on *consistency across emotions*, because a zero-shot clone copies one clip's mood and timbre. It should also win on *in-character delivery* of untagged lines.
  - For 2–5-minute characters, zero-shot (or a registered zero-shot speaker) is likely as good or better.
  - This must be measured; see §7.

---

## 6. Ranked recommendation and what changed vs the Japanese survey

| Rank | Model | Why (Mandarin) | Main risk |
|---|---|---|---|
| 1 | **Fun-CosyVoice3-0.5B-2512 (+RL), multi-speaker SFT with speaker prompts, flow trained on the cast, `spk2info` per character** | Best open test-hard (5.44 RL) and top-tier zh SS (78–80) [P]. Instruct plus non-verbal tags [P/L]. Speaker-prompt mSFT [P]. Published instruct-retention method [P]. `inference_sft` embedding-only code path [P]. Apache | Reference-free use and instruct retention are untested on CosyVoice3 [U]. #1731 bistream bug [P]. 0.5B stability (#1703) [P]. Recipe assembly is DIY [P/I]. SFT may raise WER (CosyVoice2 Tbl 11) [P] |
| 2 | **Qwen3-TTS-12Hz-1.7B-Base**, with your trainer extended to N speaker slots (or one checkpoint per main character) | Existing pipeline [L]. Best zh CER (0.77 own; CV3-zh 3.09) [P]. Reference-free named speakers [P]. Community zh character fine-tunes exist [P/S] | Lower Mandarin SIM: CV3-zh SS 73.0 vs CosyVoice3-0.5B 80.0; hard-zh 69.1, tied with VoxCPM2 [P]. No emotion control after fine-tuning [P/L]. Single-speaker official code. Fine-tuning bugs (#179, #343, #337) [P/L] |
| 3 | **VoxCPM2, one LoRA per character** | Simplest documented path. Hot-swap, reference-free, 48 kHz output, Apache [P]. zh 0.97 / 79.5 [P] | Hard-zh: 8.13 CER / 75.3 SIM in its own table [P], but QA3 hard-zh SIM is only 69.9 [P]. English emotion win-rate 42.3% [P]. Style prefix after LoRA [U] |
| 4 | **dots.tts SOAR** (watch / challenger) | Highest open zh SIM (81.0 / 79.5) [P]. Best open EmergentTTS emotion score [P]. Apache, official training | No speaker IDs, so N × 2B checkpoints [P/I]. Smoke-only config. Young (VAE artifacts #58, garbled short texts #17) [P titles] |
| Bar | IndexTTS-2.5 (emotion), FireRedTTS3-Base/Instruct (SIM + instruct), LongCat-AudioDiT-3.5B (SIM), Fish S2 Pro (expressive; non-commercial) | Define the zero-shot quality any fine-tune must beat | No official fine-tuning [P/L] |
| Baseline | GPT-SoVITS v2ProPlus (V5 once tested) | The community default for fixed character voices [S] | Reference clip always required [P]. CV3-zh CER 7.34, and the lowest text-unrelated happy and angry accuracy in CV3 Tbl 9 [P] |
| Out | Spark-TTS (SIM 66), MegaTTS3 (no training, no update), Ming-3.0 (no fine-tuning), MOSS-TTSD (dialogue/podcast), Confucius4 (reference-based, no emotion control), Irodori (JA only) | | |

**What changes because the target is Mandarin, not Japanese** [I]:
- **Qwen3-TTS loses its decisive advantages.** JA CER, ja→zh, and multilingual retention after fine-tuning were the reasons [L]. On Mandarin it is a good-CER, lower-SIM model with no emotion control after fine-tuning. It falls from "#1 by elimination" to "#2, keep if the bake-off ties".
- **CosyVoice3 rises from "borrow the recipe" to #1 on paper.** It was set aside only for weak Japanese [L]. Mandarin is its home language, and its instruct, tags and multi-speaker SFT design fit the cast.
- **dots.tts rises.** Its weak JA CER [L] is irrelevant, and its Mandarin SIM and emotion scores are top.
- **VoxCPM2's edge disappears.** Its edge was human-rated *cross-lingual* pronunciation [L], which is irrelevant here. It stays as the low-effort fallback.
- **IndexTTS-2.5, FireRedTTS3 and LongCat now look excellent on Mandarin**, but their status does not change: still no official training code. They define the bar.
- **New discriminators:** multi-speaker support, reference-free inference, and emotion control retained after fine-tuning. Those were not requirements for Kurisu.

---

## 7. Minimal experiment to settle it

1. **Zero-shot bar (no training).**
   - Pick 3 main and 2 side characters. Hold out 40 script lines per character, covering calm, happy, angry, sad and shouting.
   - Use one good 8–12 s in-character reference per character. Render with IndexTTS-2.5 (with and without `use_emo_text`), FireRedTTS3-Base, LongCat-3.5B, CosyVoice3-0.5B-RL, dots.tts SOAR and Qwen3-Base.
   - Metrics:
     - CER with Paraformer-zh.
     - SIM against a per-character centroid (ERes2Net and WavLM).
     - Blind native-listener ratings (1–5) for "sounds like <character>" and "emotion fits the line".
2. **Fine-tunes on identical splits.**
   - CosyVoice3 mSFT: prefix "You are <角色>.<|endofprompt|>". Optionally add an emotion clause on a subset, with masking.
   - Qwen3 trainer, multi-speaker extension.
   - VoxCPM2 LoRA per character.
   - Optionally, one dots.tts single-character run on the largest main character.
3. **Decision rule.** Adopt a fine-tune only if it beats the best zero-shot by ≥ 0.3 on the character and emotion MOS, or by ≥ 0.03 SIM, without a CER regression above 0.5 points. Choose between CosyVoice3 and Qwen on the same scores. Break ties in favour of CosyVoice3 if you want instruct control, and in favour of Qwen to save engineering time.

---

## 8. Could not verify

1. Any published zero-shot vs fine-tune comparison on expressive Mandarin character voices.
2. Whether CosyVoice3 keeps instruct after an mSFT built from the public recipe without the in-house instruction data.
   - Whether non-streaming `inference_sft` actually runs end-to-end on a fine-tuned CosyVoice3. `frontend_sft` was read: it feeds only an embedding. The flow normally also gets prompt tokens and mel, and whether it behaves without them is untested.
   - Whether PR #1700 (the #1731 bistream fix) is merged.
3. Contents and quality of `gglabs/cosyvoice3-multilingual-multispeaker-v3_ep49` (HF returned 401; facts come from a search snippet).
4. The original source of the third-party Qwen3-TTS Seed numbers (1.22 / 77.0; hard 6.76 / 74.8).
5. GPT-SoVITS V5 quality, languages and training changes (no notes beyond the README lines).
6. Whether community IndexTTS2 trainers (instavar, SrQingChen) work, and whether they keep emotion control.
7. Whether the VoxCPM2 `(style)` prefix, or Fish S2 tags, survive a LoRA.
8. Which GPT-SoVITS version the CosyVoice3 paper evaluated.
9. Zhihu and 53ai article bodies (403/404); only snippets were used.
10. Licences of Spark-TTS and MegaTTS3 (not pursued, since both are out).

---

## 9. Sources

**Model repos, cards and papers [P]**
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS ; fine-tuning README https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning ; report https://arxiv.org/abs/2601.15621 ; issues https://github.com/QwenLM/Qwen3-TTS/issues/14 , /179, /343 ; TaoLive https://huggingface.co/TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice ; SpragAI https://huggingface.co/SpragAI/qwen3-tts-emotion-tags ; derivatives https://huggingface.co/api/models?filter=base_model:finetune:Qwen/Qwen3-TTS-12Hz-1.7B-Base
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice ; recipe https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/libritts/cosyvoice3/run.sh ; code https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/cli/cosyvoice.py ; CV3 paper https://arxiv.org/abs/2505.17589 ; CV2 paper https://arxiv.org/abs/2412.10117 ; issue https://github.com/FunAudioLLM/CosyVoice/issues/1703 ; CV3-Eval https://github.com/FunAudioLLM/CV3-Eval ; derivatives https://huggingface.co/api/models?filter=base_model:finetune:FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- IndexTTS: https://github.com/index-tts/index-tts ; https://github.com/index-tts/index-tts/issues/518 ; IndexTTS2 paper https://arxiv.org/abs/2506.21619 ; topic https://github.com/topics/indextts2 ; derivatives https://huggingface.co/api/models?filter=base_model:finetune:IndexTeam/IndexTTS-2
- VoxCPM: https://github.com/OpenBMB/VoxCPM ; https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
- dots.tts: https://github.com/rednote-hilab/dots.tts ; issues https://github.com/rednote-hilab/dots.tts/issues
- FireRedTTS3: https://github.com/FireRedTeam/FireRedTTS3 ; issues https://github.com/FireRedTeam/FireRedTTS3/issues
- Confucius4-TTS: https://github.com/netease-youdao/Confucius4-TTS
- LongCat-AudioDiT: https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B
- Ming-3.0-tts: https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B
- Fish S2: https://huggingface.co/fishaudio/s2-pro ; https://arxiv.org/abs/2603.08823
- Qwen-Audio-3.0-TTS (CV3 and hard-zh tables): https://arxiv.org/abs/2607.23938
- GPT-SoVITS: https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/README.md ; https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/docs/cn/Changelog_CN.md ; https://huggingface.co/api/models/lj1995/GPT-SoVITS ; https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/api_v2.py
- MOSS: https://github.com/OpenMOSS/MOSS-TTSD ; https://github.com/OpenMOSS/MOSS-TTS
- Community HF: https://huggingface.co/mon3stera/Qwen3-tts-Mon3tr ; https://huggingface.co/baicai1145/GPT-SoVITS-STAR ; https://huggingface.co/api/models?author=AI-Hobbyist

**Secondary [S]**
- Baseten, Qwen3-TTS fine-tune vs zero-shot (2026-07-31): https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/
- CSDN galgame → Qwen3-TTS (2026-06-01): https://adg.csdn.net/6a311cd310ee7a33f27ded9e.html
- 80aj / linux.do CosyVoice3 vs IndexTTS2, Arknights (2025-12-17): https://www.80aj.com/2025/12/17/liangdattsmoxingduijuecosyvoice3yuindextts2shicetiyan/
- liudon.com 2026 cloning comparison: https://liudon.com/posts/voice-cloning-solution-comparison/
- cnblogs 2026 selection guide: https://www.cnblogs.com/sensorsen/p/21367537
- Zhihu (snippets only): https://zhuanlan.zhihu.com/p/2050877590802638365 ; https://zhuanlan.zhihu.com/p/2093708094308087657 ; 53ai: https://www.53ai.com/news/OpenSourceLLM/2026010435620.html
- Aliyun Bailian CosyVoice SFT: https://help.aliyun.com/zh/model-studio/fine-tune-speech-synthesis-model-by-api
- MindSpeed-MM CosyVoice3: https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/cosyvoice3/README.md
- gglabs multi-speaker CosyVoice3 (snippet): https://huggingface.co/gglabs/cosyvoice3-multilingual-multispeaker-v3_ep49
- vaghawan Qwen3 speaker-slot note: https://huggingface.co/vaghawan/tts-best
