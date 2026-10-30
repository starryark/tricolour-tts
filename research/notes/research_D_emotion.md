# Explicit emotion control for the Tricolour cast (state as of 2026-10-29)

Scope: the new hard requirement is **explicit emotion control at inference** that overrides what the text implies, tested as *neutral text plus a non-neutral control*, on top of character identity. This note decides between (a) Qwen3-TTS-1.7B-Base with taught emotion control, (b) Fun-CosyVoice3-0.5B-2512 with built-in instruct, and (c) IndexTTS-2/2.5. It does not repeat `voice_model_approach_2026-10.md` §3 or `research_B_mandarin_models.md`.

**Tags**
- **[P]** primary source read by me or my research sub-agent: paper table, repo code, issue, HF card.
- **[S]** secondary: blog, forum, search snippet, unevaluated community claim.
- **[I]** my inference.
- **[U]** unverified.
- **[L]** local file or measurement (Kurisu notes under `E:\STEINS;GATEVoice\kurisu_v2\research\`).

All sources were accessed on 2026-10-29. **Text congruence** is flagged on every accuracy number: *congruent* means the test text already implies the emotion, which overstates control; *neutral/unrelated* matches the acceptance test.

---

## 0. Answer in brief

1. **Nobody has published a measured result for teaching whole-utterance emotion control to Qwen3-TTS-Base** [P, searched HF derivatives, GitHub, Qwen issues and discussions]. The nearest things:
   - **arXiv 2606.05367 (2026-06), on Qwen3-TTS-12Hz-1.7B-Base itself:** per-emotion LoRA/full fine-tunes found "no useful window: noise (high lr) or calm speech (low lr)". What did work was **training-free x-vector arithmetic**, `x = x(target, neutral) + α·τ_emo`: emotion2vec cosine rose by about +0.29 on held-out ESD speakers (English), and α trades emotion for identity. The authors "localize the dominant carrier of emotional prosody to the x-vector" [P]. One practitioner calls x-vector emotion injection a dead end ("too little state") [S].
   - **vspeech/Qwen3-TTS-Train** has an `--instruct_model` mode for Base (instruct column, e.g. `用特别愤怒的语气说` on a neutral sentence). It reports no results, and its only user report is "only noise" after instruct training (#5, open) [P].
   - **Discussion #121:** the fine-tuned Base "completely fails to follow instructions". One user is trying an instruct column (2026-06) and has posted no result [P].
2. **Teaching emotion by fine-tuning works on other codec-LM TTS, when labels are clean and the text is parallel/neutral** [P]:
   - **Spark-TTS, ESD SFT** (350 utterances × 10 speakers ≈ 3,500 per class, human labels, ESD texts shared across emotions):
     - emotion2vec+ large accuracy 0.81, human 0.76;
     - with GRPO: 0.88 / 0.89 (EMORL-TTS).
   - **CosyVoice-300M-Instruct, ESD instruction-tuning (SFT) only:** per-class emotion2vec+ 0.60–0.83, rising to 0.60–0.84 with DPO (Emo-DPO; the baseline is very probably the SFT-only stage, §1.1).
   - **CosyVoice2, SFT on synthetic emotion data:**
     - emotion2vec accuracy 25% → 43% after SFT and 69% after GRPO;
     - speaker similarity 0.792 → 0.794 → 0.780;
     - on a **neutral-text** set, the CosyVoice2 instruct baseline scores **23% (5 classes, about chance)** (arXiv 2609.11523).
   - **RL with an SER reward is where the big gains come from** in every paper that separates the stages.
3. **The open CosyVoice3-0.5B's documented emotion control is narrow and weak on neutral text** [P]:
   - Its shipped `instruct_list` has exactly three emotions: **请非常开心/伤心/生气地说一句话**. It also has loud/soft, fast/slow, 17 dialects, Peppa Pig and robot.
   - Community re-measurements on CV3-Eval's emotion set (emotion2vec+ large, **text-unrelated**):
     - 0.5B: happy 0.65–0.68, sad 0.36–0.60, angry 0.48–0.64;
     - RL model: 0.76 / 0.48 / 0.56;
     - all from issue #1687.
   - **Correction to the framing in earlier notes:** the CV3 paper's own Table 9 (text-unrelated 0.64 / 0.42 / 0.58) measures **emotion cloning from an emotional reference clip** (`CV3-Eval/data/emotion_zeroshot`), not instruct control. The #1687 figures are the relevant instruct numbers.
   - Users report that emotion is "strongly bound to the text" and that instruct lowers speaker similarity (#1057) [S]. The CosyVoice2 paper measured SS 0.817 → 0.804 with instruct [P].
   - **No one has measured instruct retention after a CosyVoice3 speaker SFT** [P/U].
4. **CosyVoice3 code facts that change the plan (b)** [P code]:
   - **`inference_instruct2(..., zero_shot_spk_id=…)` silently drops the instruction.** It reuses the stored entry's prompt text (#1400; fix in PR #1475, open since 2025-07).
   - The CV3 LLM **ignores the speaker embedding** (`embedding` is accepted but unused in `Qwen2LM.inference`). So in a reference-free mSFT, the LLM learns identity **only from a text prefix** ("Speaker A<|endofprompt|>"), and the instruction rides in the same prefix.
   - CosyVoice3 training accepts a per-row `instruct` field (CosyVoice2 training ignores it, #1909).
5. **IndexTTS-2/2.5 has the best emotion interface, but no trainer keeps it measurably** [P]:
   - Community trainers train the GPT (instavar, a re-badged JarodMica trainer) or GPT + CFM LoRA (SrQingChen), with the emotion modules frozen. None measures emotion after fine-tuning.
   - The one at-scale report (#369, 500–600 h) says emotion transfer "基本上就退化没了" (was essentially gone) after 1–2 epochs.
   - The s2mel stage always takes a prompt mel. Shipping that mel is about the same as shipping audio, so (c) fails the reference-free public-release constraint [I].
6. **A path the brief did not list: (a′) Qwen3-TTS-CustomVoice with our locked x-vectors written into spare speaker rows.**
   - CustomVoice already follows emotion instructions. #248 is a user judging happy/sad/embarrassed/angry/disgusted/fearful by ear [P issue, user-judged]. The only numbers are InstructTTSEval-zh 83.0 / 77.8 / 61.2 [P, research_B].
   - Its preset rows are x-vector-scale (norm 13.9–15.6, pairwise cosine 0.90–0.96) [L].
   - The community Rust engine already injects Base speaker-encoder x-vectors into CustomVoice's speaker slot together with an instruct [P code]. Nobody has evaluated it [U].
   - It costs one local afternoon to test, and it avoids #213 because rows are written, not trained [I].
7. **Labelling** (§5) [P]:
   - **emotion2vec+ large** has no published accuracy on its own card. Its training saw CASIA/M3ED/MER2023, so it has no clean Mandarin benchmark. It is unreliable on fear, surprise and disgust, and **untested on acted or anime Mandarin**.
   - **Qwen3-Omni** reports no SER numbers and does not fit 16 GB. Zero-shot Mandarin UA for open models is low (M3ED 19–37, CASIA 34–53).
   - **SenseVoice's emotion labels derive from emotion2vec+**, so it is not an independent vote.
   - **Script-context text labels** are the best independent second signal: dialogue context lifts text-based emotion F1 by 6–9 points on M3ED.
   - **Even audio+text agreement is not clean:** Emo-Emilia kept 1,400 of 4,200 agreeing clips after expert review.
   - Our own ~300-clip human calibration is required.
8. **Inventory** (§6): **happy, angry, sad, neutral** are the classes every strong system and every labeller supports. **Surprised** and **fearful** are supported by IndexTTS2 and emotion2vec, but labellers are weakest on them. **Shy/whisper is phonation, not an emotion**: control it as a separate soft/whisper style flag (CosyVoice3 "very soft voice"). **Intensity** is feasible only as a coarse 2–3 level control, or as a continuous α in x-vector/emotion-vector arithmetic; it is measured only in English (EMORL-TTS listener pairwise 0.65–0.72).

**What decides (a) vs (a′) vs (b) vs (c)** is in §7. Four cheap local probes (D1–D4) settle most of it before any H100 time.

---

## 1. Q1: teaching emotion control by fine-tuning a codec-LM TTS

### 1.1 Results with numbers

| Work (date) | Base | Control input | Data per class, label source | Emotion accuracy (judge; text congruence) | Speaker sim / WER effect | Tag |
|---|---|---|---|---|---|---|
| **EMORL-TTS** ([2510.05758](https://arxiv.org/abs/2510.05758), ICASSP 2026) | Spark-TTS (BiCodec frozen, LLM trained) | Emotion + intensity control tokens prepended to text | ESD-en 10 spk × 350 ≈ **3,500 per class** (5 classes), plus Expresso 4,717 utt; human labels. Intensity labels from a VAD estimator | emotion2vec+ large (500 samples/model): **SFT only 0.81**, +GRPO 0.88; CosyVoice2 0.63. Human: SFT 0.76, +GRPO 0.89, CosyVoice2 0.55. Text source not stated | MOS 4.96 (Spark) → 4.92 (SFT) → 4.94; no SIM, no WER reported | [P] |
| **Emo-DPO** ([2409.10157](https://arxiv.org/abs/2409.10157), 2024) | CosyVoice-300M-Instruct (LLM tuned; flow frozen) | Emotion word + `<endofprompt>` (`E.<endofprompt>x`) | ESD-en, 350 per speaker × emotion (≈3,500 per class); human labels | emotion2vec+ large per class (Neu/Ang/Hap/Sad/Sur): "cosyvoice" baseline **0.69/0.83/0.60/0.69/0.65**; Emo-DPO 0.76/0.84/0.60/0.71/0.72. The baseline row's other metrics (98.73 / 3.69 / 4.94) are identical to the ablation row "− L_DPO − L_SFT − L_KL" (Stage-1 instruction tuning on ESD only), so **the baseline is very probably the SFT-only model** [P numbers / I identity]. ESD texts are shared across emotions (neutral content) [I from ESD design] | Not reported; intelligibility column 4.94 → 4.54 (unit unstated) | [P] |
| **Emotion + duration post-training** ([2609.11523](https://arxiv.org/abs/2609.11523), 2026-09) | CosyVoice2 (LLM trained) | Free-form NL instruction | Synthetic IndexTTS2 speech (MED-TTS texts, ESD emotion refs), manually screened; hours not given | emotion2vec Acc: base 25.3 → **SFT 42.7** → SFT+GRPO 68.9; MERaLiON-SER Acc 26.7 → 55.3 → 89.8. Base "ASR↓" is printed as 77.62 vs 35–39 after training (unexplained; possibly instruction leakage on multi-segment prompts [U]). **Neutral-text counterfactual set:** CosyVoice2 instruct base **E2V 23.0 / MER 25.6 (5 classes ≈ chance)**; final Joint-RL 40.8 / 71.4 | Spk 0.7919 (base) → 0.7935 (SFT) → 0.7797 (RL) | [P] |
| **EmoVoice** ([2504.12867](https://arxiv.org/abs/2504.12867), 2025-04) | Qwen2.5-0.5B LLM on CosyVoice tokens | Free-text description ("Say this sentence with emotion of …") | EmoVoice-DB: 7 classes, ~3,000–3,500 per class (40 h), **all GPT-4o-audio synthetic, labels never checked** (WER filter only); zh fine-tuning on an internal set | **Chinese Secap test (real recordings):** EmoVoice recall 0.411, EmoVoice-PP 0.434, CosyVoice2 0.403, **real recordings 0.800**. English: 0.395–0.424 vs 0.525 for the synthetic GT. Recall excludes fear/disgust/surprise "because emotion2vec was unreliable". Texts are congruent by construction | No SIM. WER zh 7.6 vs GT 6.6 | [P] |
| **CosyVoice 3 DiffRO-EMO** ([2505.17589](https://arxiv.org/abs/2505.17589) §5.4) | CosyVoice3-1.5B (unreleased) | SER-reward post-training | in-house | CV3-Eval **emotion cloning** (emotion comes from the reference clip, *not* an instruction; the subset is `data/emotion_zeroshot`): text-unrelated happy/sad/angry 0.64/0.44/0.48 → **0.98/0.50/0.68**. Paper: "TTS systems primarily infer the emotional tone of output audio from text sentiment" | "can adversely affect pronunciation" | [P] |
| **Task-vector / x-vector arithmetic** ([2606.05367](https://arxiv.org/abs/2606.05367), 2026-06) | **Qwen3-TTS-12Hz-1.7B-Base** | No training: `x_new = x(target, neutral) + α·τ_emo` in the speaker slot | ESD-en; τ from 50 utt/speaker/emotion × 4 speakers | emotion2vec **cosine** (EECS, not accuracy), held-out speakers: angry 0.539 → 0.925, happy 0.425 → 0.687, sad 0.540 → 0.761 (mean +0.29). PT-BR +0.09 | WER 0.03–0.09. SECS ≥ 0.88 for the averaged τ (PT-BR); α trades emotion for identity. **Per-emotion fine-tunes failed** ("noise (high lr) or calm speech (low lr)") | [P] |
| **Marco-Voice** ([2508.02038](https://arxiv.org/abs/2508.02038), 2025-08) | CosyVoice1 + rotational emotion embedding (from emotional − neutral pairs) | Emotion embedding | CSEMOTIONS: 10 h Mandarin, 10 actors, 7 classes; plus ESD 29 h | emotion2vec_base_finetuned: **zh 0.78** vs CosyVoice1 0.72. **Congruent** (emotional prompts with fillers and interjections) | AISHELL SS-ERes2Net 73.5 → 67.6–73.8; WER 3.0 → 15.9–18.2 | [P] |
| **macminix/qwen3_voice_design_t8** (2026-05-05) | Qwen3-TTS-12Hz-1.7B-**VoiceDesign**, i.e. instruct already present | LoRA r16 on talker, KL-to-base β 0.02 on 20% neutral batches | VCTK + EARS regular reads + Expresso, ~46k clips, English, captions rewritten | "emotion accuracy" **0.559 → 0.712**. Classifier unnamed; eval is part of a 59-prompt set, so the emotion matrix has only a few dozen items. It refines an existing instruct ability, it does not teach one | WER 0.0215 → 0.0187; gender acc 0.847 → 0.746 | [P card] |
| macminix T5/T6 (2026-04) | same | same, stronger emotion data | — | No numbers. **"strongly emotional descriptions act partially as speaker-identity cues"** because emotion data came from few speakers, so sad/fear prompts flipped gender | — | [P card] |
| **SpragAI/qwen3-tts-emotion-tags** (2026-09-09) | 1.7B-**CustomVoice**, 9 presets only | Plain-text `[Angry] [Sad] [Happy] [Fast] [Gentle] [Tired] [Fearful] [Disgusted] [Surprised]` | 76,862 rows, 9 speakers, 9 tags (≈ 8.5k per tag); source not stated (likely CustomVoice-instruct self-generated [I]). LoRA r32 + code-predictor input embeddings; `codec_embedding` frozen | **No evaluation.** Samples use neutral text | Speaker rows "bit-identical" | [P card] |
| Kurisu v3 [L] | Qwen3-TTS Base, our trainer | `[sigh]`, `[nv]` point events, BPE-split | 81 / 176 examples | Realisation 13% / 14% (detector at 5% FPR) vs 3% / 0% for untagged twins | ×3 upsampling cost similarity | [L] |
| CosyVoice issue #1275 (2025-05) | CosyVoice2-0.5B | New special tokens for events | 1,000 samples | "does not learn"; tags rendered as silence | — | [S] |
| IndexTTS2 ([2506.21619](https://arxiv.org/abs/2506.21619)) | — | Emotion vectors / emotion prompt | 135 h emotional speech | (zero-shot system; already in research_B) | — | [L/P] |

### 1.2 What these say about our regime

- **Examples per class.** The positive results used **≥ ~1,000 examples per class with clean human labels**:
  - Marco-Voice's CSEMOTIONS is ~10 h / 7 classes, roughly 1,000–1,500 clips per class, with congruent text;
  - the strongest results (ESD-based) had ~3,500 per class with parallel neutral text.
  - Kurisu's 81–176 point events and #1275's 1,000 *new event-token* examples failed or were weak. The Tricolour data is ~20k clips. If the emotion distribution is typical of VN scripts (mostly neutral), the rarer classes (fear, surprise) may have a few hundred clips each across the cast [I, must be counted after labelling]. No paper gives a minimum per class.
- **Text congruence is the trap.** Our training data is *acted to the script*, so emotion and text are strongly correlated. A model trained on it can satisfy the loss by reading emotion off the text and ignoring the tag. That is exactly CV3's finding ("infer the emotional tone… from text sentiment") and the 23% neutral-text score of CosyVoice2's instruct. ESD-style successes had **the same sentences in every emotion**, which forces the model to use the control [I]. Mitigations [I]:
  - (i) select or upweight clips whose text is emotionally neutral but whose audio is not; the script-context LLM label from §5 finds them;
  - (ii) tag dropout (some rows with no tag), so the tag is not the only route to emotion;
  - (iii) RL/DPO with an SER reward on neutral texts, which is where every paper got its large gains (EMORL, 2609.11523, DiffRO-EMO, Emo-DPO).
- **Label noise.** Nothing measures noise tolerance directly [U]. Indirect evidence:
  - EmoVoice trained on never-verified synthetic labels and got about half the real-recording recall on Chinese (0.41 vs 0.80) [P].
  - EmoCtrl-TTS ([2407.12229](https://arxiv.org/abs/2407.12229)) pseudo-labelled 200k h with emotion2vec, keeping non-neutral classes at any confidence and neutral/happy only at confidence 1.0. It reports no label accuracy and finds "simply adding emotional data does not improve emotion transferability" [P].
- **Speaker identity.** SFT barely moved speaker similarity in 2609.11523 (0.792 → 0.794); RL cost about 0.01 [P].
  - **The bigger identity risk is confounding:** when an emotion is concentrated in a few speakers, the emotion control becomes a speaker cue (macminix T5) [P].
  - The VN cast will be unbalanced the same way, e.g. one character is mostly angry [I]. Balance emotion × character in sampling, and measure identity per emotion.
- **WER.** Emotion SFT/RL in CosyVoice-family models costs some intelligibility (DiffRO-EMO; Marco-Voice WER 3.0 → 16–18 on AISHELL; EmoVoice's hard-case WER 18) [P]. Gate it.
- **Judges disagree.** On the same outputs, emotion2vec accuracy 68.9% vs MERaLiON-SER 89.8% (2609.11523) [P]. Absolute numbers are judge-specific. Compare systems with the same judge, against the judge's accuracy on our real clips (§5).

### 1.3 Qwen3-TTS-Base specifically

| Project | What | Result | Tag |
|---|---|---|---|
| arXiv 2606.05367 | Base, emotion via x-vector offset | Works on EECS (cosine); weight fine-tuning per emotion failed | [P] |
| vspeech/Qwen3-TTS-Train (★119, last push 2026-03-18) | `--instruct_model`: instruct column trained into Base's ChatML slot | No results; #5 (2026-06-08) "only noise" after instruct training, unanswered. #6 notes speaker-embedding norm collapse from AdamW weight decay | [P] |
| Discussion #121 (2026-01-28 → 2026-07-31) | Fine-tuned Base ignores instructs; question whether to fine-tune CustomVoice instead | One user (2026-06-04) trying an instruct column; no follow-up | [P] |
| lunahr/Celune-1.7B-{Calm,Neutral,Energetic,Upbeat} (2026-04) | **One Base fine-tune per style** | Claims "prompt steering"; no evaluation | [S] |
| gabriele-mastrapasqua/qwen3-tts (dev.to, Jul 8) | CustomVoice weights + ICL clone + residual-stream steering vectors (layers 21–25) + an Italian LoRA (".expr") + English instruct | "Neither alone is enough on a clone; together they push emotion hard enough to be heard." Dead ends: τ-vectors "muddy and timbre-shifting", x-vector injection "too little state". No numbers | [S] |
| Kurisu internals [L] | The Base already has a latent instruct prior: issue #25 calls 12Hz Base control "relatively unstable", not absent. CustomVoice drifted 3–7% from Base, VoiceDesign 8–21%, Kurisu v2 0.3–0.6% | A zero-cost probe (feed an emotion instruct to the existing custom_voice checkpoint) was specified but never run | [L] |

**Caveat on the two negative Base results [I]:** 2606.05367's failed per-emotion fine-tunes ("noise at high lr, calm at low lr") and vspeech #5's "only noise" both match the symptoms of the official `sft_12hz.py` label-alignment and `text_projection` bugs: Discussion #288, issue #39, PR #278 [P]. Our trainer fixed those bugs [L]. Neither paper says which trainer it used, so treat both as weak evidence against (a1) until reproduced with a fixed trainer.

**Verdict on (a):** teaching emotion to Base is **unmeasured** and has one failed public attempt (vspeech #5) [P]. Two mechanisms are plausible [I]:
- **(a1) emotion tokens in the text** (new embedding rows), the regime of EMORL/Emo-DPO;
- **(a2) per-(character, emotion) locked x-vector rows**: either centroids of each character's clips in that emotion, or `char_neutral + α·τ_emo` with τ pooled across the cast. This uses the slot that 2606.05367 says carries emotion, needs no new tokens, gives α as an intensity knob, and the trainer already supports many rows.

The risk in (a2) is that |τ_emo| is comparable to inter-character distances. Raw cosines between cast centroids are already 0.978–0.994 [L], so emotion offsets could push a character toward a neighbour. That is measurable locally before training (D3).

---

## 2. Q2: alternatives inside the Qwen3-TTS family

**Is anyone fine-tuning a new speaker into CustomVoice/VoiceDesign while keeping instruct?**

| Who | How | Instruct after? | Evidence | Tag |
|---|---|---|---|---|
| scrappylabs/narrator-tts (2026-02-05) | Fine-tuned 1.7B-CustomVoice on 461 narrator clips, 5 epochs, lr 2e-5, bs 64. How the speaker row was set is not stated | Claims "Instruct/emotion support … Full support" | No evaluation; examples use congruent text | [P card / U] |
| syyxsxx, Discussion #231 (2026-03-11) | "I tried fine-tuning the Custom-Voice model. It works for a single speaker." | Not stated | One line, no numbers | [S] |
| Issue #217 thread (2026-02) | CustomVoice fine-tuning fails with "no speaker encoder"; workaround: load Base's `speaker_encoder` manually | — | Mechanics only | [P] |
| cgisky1980/Qwen3-TTS-Rust (★70, last push 2026-06-26) | **No training.** Loads CustomVoice GGUF weights, computes an x-vector with Base's speaker encoder (ONNX), and adds it in the speaker slot (`marker_emb + spk_emb`), with an optional instruct block | Claims "voice cloning + instruction control"; README examples use congruent text (`对不起，我不是故意的...` + `悲伤啜泣`) | No evaluation | [P code / U] |
| Mastrapasqua engine | CustomVoice weights + ICL clone + steering + LoRA | Yes, with steering | Listening only | [S] |
| macminix T1–T8, vocence miners (SN78) | VoiceDesign LoRA, no fixed speaker | Instruct kept (VoiceDesign) | macminix numbers in §1.1 | [P] |
| SpragAI | CustomVoice LoRA, presets only, rows frozen | Tags added | No eval | [P] |
| TaoLive TBLive-CustomVoice | Fine-tuned **from Base** (6 speakers, pause/speed tokens) | Not claimed | — | [P, research_B] |

**How it could work for us [I]:**
- **Don't train a CustomVoice row from scratch.** #213 (2026-02-14): training a speaker index token "推理的结果每一个文本的音色随机变化" (timbre changes randomly with each text) [P].
- **Write our locked fp32 mean x-vectors into spare rows (3000–3013) of the CustomVoice checkpoint.** This is plausible because CustomVoice's preset rows are x-vector-scale (norm 13.9–15.6, pairwise cosine 0.90–0.96; Base has the same rows at init level 0.022), and our Kurisu row (norm 17.0) sits at cosine 0.945–0.960 to vivian, ono_anna and serena [L]. The Rust engine shows the slot accepts encoder x-vectors at inference [P code].
- **Then either:**
  - (i) use it as is, with CustomVoice instruct (D2); or
  - (ii) fine-tune CustomVoice with rows locked on our data. Mix three kinds of rows:
    - our lines with no instruct;
    - our lines with an emotion instruct from the labels, on emotionally neutral text where possible;
    - an **instruct replay** set that CustomVoice generates on its own presets (self-distillation), so the instruct pathway is not overwritten.
  - That is the CosyVoice3 §2.6 recipe ported. Use a low-drift regime (KL-to-base on replay batches, as macminix did).
- **No result exists for any of this** [U]. The Qwen maintainer's answer to the same question (#14, 2026-01-23) was a future 25Hz VoiceEditing model; #25 repeats it, and it was never released [P].

---

## 3. Q3: CosyVoice3 specifics

### 3.1 Prompt format and the supported list [P code]

- **Format:** `instruct_text` must end with `<|endofprompt|>`. For CosyVoice3 the examples prefix `You are a helpful assistant. `. `Qwen2LM.inference` asserts that token id 151646 is present for `CosyVoice3LM`.
  - Example: `cosyvoice.inference_instruct2(tts_text, 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>', prompt_wav)` (`example.py`).
- **`cosyvoice/utils/common.py` `instruct_list`** ("for supported control, check cosyvoice/utils/common.py#L28"):
  - 17 dialects (广东, 东北, 甘肃, 贵州, 河南, 湖北, 湖南, 江西, 闽南, 宁夏, 山西, 陕西, 山东, 上海, 四川, 天津, 云南);
  - "Please say a sentence as loudly as possible." / "…in a very soft voice.";
  - 请用尽可能慢/快地语速说一句话;
  - **请非常开心地说一句话 / 请非常伤心地说一句话 / 请非常生气地说一句话**;
  - 小猪佩奇风格; 机器人.
- **That is three emotions.**
  - The CosyVoice3 paper's Table 1 lists "over 100" styles in the pre-training instruction data (5,000 h), including many adjectives, but these are not documented as supported for the open 0.5B [P].
  - CosyVoice2's paper lists happy, sad, surprised, angry, fearful, disgusted, calm and serious [P]. A user applying CosyVoice2's demo list (惊讶, 恐惧, 恶心…) got "结果不明显" (results not noticeable) (#1057) [S].
- **Fine-grained tags:** `[breath]`, `[laughter]`, `<strong>…</strong>`, `<laughter>…</laughter>` [P].
- **Known failure: instruction read aloud.** #1903 (2026-06, "frequent on CosyVoice3 when the instruction is long, CosyVoice2 fine"), #1802, #1263, #1084 [S].
- **Measured instruct emotion accuracy, open 0.5B, CV3-Eval emotion texts, emotion2vec+ large (#1687)** [S, community]. Instruct `You are a helpful assistant.请用高兴/生气/悲伤的情感说。<|endofprompt|>`:

  | Model / measurer | Text-related H/S/A | Text-unrelated H/S/A |
  |---|---|---|
  | 0.5B (non-RL), user 1 | 0.70 / 0.72 / 0.88 | **0.65 / 0.60 / 0.64** |
  | 0.5B (non-RL), user 2 | 0.88 / 0.84 / 0.84 | **0.68 / 0.36 / 0.48** |
  | 0.5B RL, user 1 | 0.73 / 0.59 / 0.82 | **0.76 / 0.48 / 0.56** |

  A third user said emotion2vec_plus_large "根本不准" (simply inaccurate) on their run [S]. The paper's own Table 9 numbers (0.64/0.42/0.58 text-unrelated) are **emotion cloning from a reference clip**, not instruct [P, CV3-Eval tree `data/emotion_zeroshot`].

### 3.2 Speaker SFT that measured instruct retention

- **None found.** No CosyVoice3 or CosyVoice2 community speaker SFT reports instruct or emotion accuracy before and after [P/U]. Searched: CosyVoice issues (all 1,891 titles), HF derivatives, the instavar CosyVoice3 LoRA guide (no emotion metrics), and web search in zh and en.
- **The paper's method** (§2.6.2): "high-quality data from the target speaker along with pre-training instruction-following dataset … randomly mask the speaker prompt or the style prompt … helps prevent potential catastrophic forgetting in instructed generation". **No results are reported** for it [P].
- **Related community signals:**
  - #1423: a user fine-tuned CosyVoice2 with custom instructions and it "不响应" (does not respond) [S].
  - #1909 (2026-06): CosyVoice2 training code ignores instruct; only `CosyVoice3LM` uses `instruct_token`. One user patched it and says it works. Another says instruct-trained models become unstable under vLLM, with "prompt污染" (prompt contamination) [S].
- **Code facts for our own SFT** [P code]:
  - `processor.tokenize` encodes a per-row `instruct` field.
  - `CosyVoice3LM.prepare_lm_input_target` prepends it with loss masked (`IGNORE_ID`).
  - **The batch only gets instructs if every sample in it has one** (`torch.tensor([...]).all()`), so give every row an instruct string, using `You are a helpful assistant.<|endofprompt|>` as the "none" value [I].
  - The CosyVoice instruct training data is not public. Replay would be self-generated, e.g. CosyVoice3 instruct outputs on our texts, or our own clips with label-derived instructions [I].

### 3.3 Stored speakers with instruct2 [P code]

- `frontend_instruct2` calls `frontend_zero_shot(tts_text, instruct_text, prompt_wav, sr, zero_shot_spk_id)`. When `zero_shot_spk_id` is set, that function returns `{**spk2info[id]}`, i.e. **the stored prompt transcript, with the instruction discarded**, and then deletes the LLM speech-prompt tokens.
  - Result: instruct2 + a stored speaker ignores the instruction (#1400, 2025-06-25: "合成的语音中将混入 … 注册时的 prompt_text", i.e. the transcript stored at registration gets mixed into the output). Community fix: open PR #1475 (2025-07-23), not merged.
- With an SFT-style `spk2info` entry (only `'embedding'`), `frontend_instruct2` would raise `KeyError` on the `del` [I from code].
- **The CosyVoice3 LLM never uses the speaker embedding.** `Qwen2LM.inference` builds `[sos, text, task_id, prompt_speech_tokens]`; the `embedding` argument is unused. So:
  - `inference_sft(spk_id)` gives the LLM **no identity at all** except what the text says, and gives the flow only `flow_embedding` (no prompt tokens or mel);
  - a reference-free CosyVoice3 mSFT must put the speaker tag in the text: `"<角色名>。请非常生气地说一句话。<|endofprompt|>正文"`. Speaker and instruct share one prefix, which `inference_sft` passes straight through. Text normalisation is skipped whenever `<|…|>` appears, so normalise beforehand [I from code].
- **Whether the flow produces the right timbre from a CAM++ embedding alone after an mSFT is untested** (already E0's smoke test) [U].
- **Choose one instruct path for E4 and use it in both training and inference [I from code]:**
  - **Option 1 (recommended):** put the whole prefix `<角色>。<instruct><|endofprompt|>` inside the `text` field, for training and for `inference_sft`. It is self-consistent, needs no frontend change, and avoids the batch-level `.all()` gate.
  - **Option 2:** train with the separate `instruct` field. It becomes `instruct_token_emb`, prepended before the text with loss masked. At inference that position is `prompt_text`, which `frontend_sft` never sets, so a ~10-line custom frontend would be needed.
  - **Don't mix them.**

---

## 4. Q4: IndexTTS-2/2.5 community trainers (sub-agent, spot-checked)

| Repo | Trains | Emotion path | Data | Results | Runs? | Tag |
|---|---|---|---|---|---|---|
| **instavar/indextts2-finetuning** (★2, last push 2026-08-14) | IndexTTS2 GPT only, full FT, AdamW on all GPT params. s2mel, BigVGAN, conditioners and emotion modules are not trained (conditioning latents and emotion vectors precomputed) | Frozen. The emotion vector fed in training is the **prompt clip's**, not the target's | wav + text + speaker; no emotion labels | One Singapore-English speaker: WER 0.013 → 0.026, ECAPA 0.874 → 0.859; `proves_adaptation_benefit: false`; listening plan excludes "emotion obedience" | Author-only commits; ~77% line-identical to JarodMica's `training_v2` | [P] |
| **SrQingChen/IndexTTS-2.5-Pro** (★7, last push 2026-10-26) | IndexTTS-2.5: PEFT LoRA on GPT attention (± MLP) and CFM attention; a BigVGAN fine-tune script | Frozen (`spk_emb_proj`, `emovec_layer`, `emo_layer`, perceivers not targeted). GPT samples use the clip's **own** emotion vector | Auto-sliced, SenseVoice transcripts; no labels | Smoke runs (21–28 synthetic Mandarin clips). DPO is real code, but its reward is 0.6·(1−WER) + 0.4·SIM with `emo_weight=0.0` | Third-party `feiwell/indextts25-mirror-lora` (81 s Mandarin): CER 0.83% → 0%, CAMPPlus 0.674 → 0.695, no emotion metric | [P] |
| JarodMica `training_v2` (★150) | v2 GPT full FT (ja, kk, vi) | Frozen | — | "seems to retain the emo capabilities fairly well" (2025-11-04), unmeasured | Used by several language fine-tunes | [P] |
| **Issue #369** (2025-09-16) | User replicated stage 3 with emotion/speaker modules frozen, ~500–600 h | — | — | "仅仅微调一两个轮次，模型的情感迁移基本上就退化没了" (after only one or two epochs of fine-tuning, the model's emotion transfer had essentially degraded away). No answer | — | [P, checked] |
| Others: hxyinan (trains with `emo_vec = 0`), risingore, oberkayb ("emotion control intact", no probe), Noth3rnY Murasame VN LoRA (ja), laowang-wy (zh accent LoRA) | — | — | — | No emotion measurements | — | [P/U] |

- **Official position:** "We do not have plans to open-source the training code" (owner, #215, 2025-07-07). #518 has 0 comments [P, checked].
- **Reference-free use:**
  - The GPT side reduces to tensors you could ship: CAMPPlus `style` (192-d), a base emotion vector, and in v2 32×1280 latents.
  - **The s2mel CFM always continues from `ref_mel`** (`prompt_x[..., :prompt_len] = prompt`). Shipping that mel is about the same as shipping audio, because BigVGAN turns it back into sound [P code / I].
  - Dropping the mel prompt would need a per-character fine-tuned CFM, which is untested.

---

## 5. Q5: labelling emotion at scale for Mandarin acted speech

Research by a sub-agent; the C2SER figures were spot-checked by me.

### 5.1 emotion2vec+ large

- **Model card (2024-05):**
  - **no accuracy numbers at all**, only an unlabelled radar chart;
  - 9 classes (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unk);
  - ~300M parameters, fine-tuned on ~42.5k h of filtered pseudo-labelled data [P].
  - Its seed stage used EmoBox academic data, which includes CASIA, M3ED and MER2023, so scores on those sets are not zero-shot [P/I].
- **Mandarin evidence, all weak or indirect:**
  - **EmoBox** ([2406.07162](https://arxiv.org/abs/2406.07162)) does not test emotion2vec+. Its *in-domain-trained* probes reach M3ED (TV drama, 7 classes) UA only 32.8 / WA 49.4 at best (Whisper-large-v3), CASIA (acted) UA 59.6, and MER2023 UA 61.2 [P].
  - **Real-recording accuracy as a TTS judge:** EmoVoice's Secap ground-truth row gives recall **0.800** on 600 Chinese clips (classes averaged unclear) [P].
  - **Naturalistic Mandarin, small sample:** a 65-clip clinical study (arXiv 2609.26054) reports UA 55.4% with near-zero recall for sad, fear, anger and surprise [P, abstract].
- **Known biases:**
  - EmoVoice dropped fear, disgust and surprise as unreliable [P].
  - EME-TTS ([2507.12015](https://arxiv.org/abs/2507.12015)), English: neutral 0.99 but surprise 0.36–0.48 while humans got 0.59–0.70 [P].
  - #1687's third user found it "根本不准" (simply inaccurate) [S].
- **Acted Mandarin anime/galgame speech:** nothing published [U]. So D4 has to calibrate it on our clips.

### 5.2 Other local models

**Zero-shot UA** ([C2SER, 2502.18186](https://arxiv.org/abs/2502.18186), Table VI) [P, checked]:

| Model | CASIA | M3ED | ESD | Emo-Emilia |
|---|---|---|---|---|
| Qwen2-Audio | 48.2 | 19.5 | 56.3 | 39.1 |
| SenseVoice-Small | 33.6 | 23.1 | 52.2 | 63.3 |
| C2SER (implicit CoT) | 53.3 | 36.7 | 96.3 | 69.0 |

- **SenseVoice's own paper** reports CASIA UA 70.0 and ESD 85.5 for Small [P]. The two papers disagree by 36 points on CASIA.
- **SenseVoice's SER labels were pseudo-labels from emotion2vec+ large** [P/I], so the two are **not independent voters**.
- **Qwen3-Omni-30B-A3B** (Instruct and Captioner): the tech report has **no SER numbers** [P].
  - BF16 needs ~79 GB; a community AWQ build needs ≥ 20 GB [S]. So it is an H100 job, not the 5060 Ti.
  - Audio LLMs tend to report the emotion the text implies (VoxParadox, in research_C) [S].
- **Qwen2.5-Omni-7B-AWQ** fits in ~12 GB [P]. Its MELD score (English) is 0.57 in its own report and 0.50 in Kimi's [P].
- **No 2026 open model clearly beats emotion2vec+ on Mandarin** [P/U].
- C2SER is the strongest zero-shot number on Mandarin sets. Whether its weights are usable locally was not checked [U].

### 5.3 Text-only labels from script context

- **M3ED** ([2205.10237](https://arxiv.org/abs/2205.10237)):
  - single-utterance text and audio are close (weighted F1 44.4 vs 46.1);
  - **dialogue context helps text +6 to +9 points but audio only +2 to +4**;
  - human agreement is Fleiss κ 0.59 for 7 classes [P].
- **MERBench** ([2401.03429](https://arxiv.org/abs/2401.03429)): on MER2023, audio 69.9 vs text 62.5 WAF; on MELD, text beats audio by 11 [P].
- **No zero-shot text-LLM vs audio-label agreement figure for Chinese exists** [U].
- **For us [I]:** a local text LLM over the line plus ±3 surrounding lines (and speaker names) is a cheap, *partly independent* second signal. It is also exactly the detector for **"neutral text, emotional delivery"** clips, which are the ones that teach override (§1.2). A script line labelled *neutral* by text and *angry* by audio is a training gold mine, not noise.

### 5.4 Agreement filtering

- **Emo-Emilia** (C2SER): kept Emilia clips where emotion2vec (audio) and GLM-4-9B (text) agreed, sampled 300 per class (4,200), and kept 1,400 after **unanimous review by four experts** [P]. Whether 33% is a survival rate or a per-class cap is unclear [U]. Agreement alone was therefore far from clean.
- **EmoCtrl-TTS:** emotion2vec, non-neutral classes at any confidence, neutral/happy only at confidence 1.0; 27k h kept from 200k h; no precision measured [P].
- **NonverbalTTS** ([2507.13155](https://arxiv.org/abs/2507.13155)): emotion2vec+ labels plus human editing; 8.5% ended with no emotion due to annotator disagreement [P].
- **VoxBox** (Spark-TTS): stores the emotion2vec label and confidence, the SenseVoice label and a Qwen2.5-72B text label, but no combination rule [P].
- **Recommended local pipeline [I]:**
  1. emotion2vec+ large (top-1 + probability);
  2. text-LLM label with context;
  3. a loudness/voicing/HNR feature for soft/whisper and shout (or a small classifier trained on AISHELL6-Whisper; the English-only Vox-Profile has whispered/hushed/soft tags);
  4. keep agreements as labels, keep "text neutral, audio confident non-neutral" as override examples after a listening spot-check, and drop the rest from the *tagged* pool (they still train untagged);
  5. calibrate everything against ~300 human-checked clips (D4).

---

## 6. Q6: a realistic emotion inventory

| Class | CosyVoice3-0.5B documented instruct | CosyVoice2 paper list | IndexTTS2 vector | emotion2vec+ (9) | Qwen CustomVoice (#248 tested) | Labellability | Recommendation [I] |
|---|---|---|---|---|---|---|---|
| neutral | (no instruct) | calm | calm | neutral | — | Easy; the dominant class | **Keep** (the default) |
| happy | 请非常开心地 | happy | happy | happy | happy | Good | **Keep** |
| angry | 请非常生气地 | angry | angry | angry | angry | Good | **Keep** |
| sad | 请非常伤心地 | sad | sad, melancholic | sad | sad | Good, but sad is the weakest on neutral text for CosyVoice3 (0.36–0.60) | **Keep** |
| surprised | — | surprised | surprised | surprised | — | Weak (EmoVoice dropped it from recall as unreliable) | **Keep as experimental**; gate it on the judge's real-clip recall |
| fearful | — | fearful | afraid | fearful | fearful | Weak (same) | **Experimental** |
| disgusted | — | disgusted | disgusted | disgusted | disgusted | Weak; rare in VN scripts [I] | Drop |
| shy / whisper | "Please say a sentence in a very soft voice." (volume, not emotion) | — | — | **no class** | embarrassed (tested) | Needs a phonation/loudness detector, not SER | **Separate style flag** `soft/whisper`, orthogonal to emotion. Label it from level (we keep per-character gain), voicing and HNR, or script cues (小声, ……) |
| shout | "as loudly as possible" | — | — | — | — | Level-based | Optional style flag |

**Intensity** [P/I]:
- **Measured** only in English:
  - EMORL-TTS weak/medium/strong from VAD-derived labels: listener pairwise ordering 0.71 / 0.65 / 0.72 after GRPO, near chance (0.50–0.58) for the baselines [P];
  - Emo-LiPO (2026-06) claims gains "at high intensity levels"; no numbers on the abstract page [P].
- **Continuous knobs that exist:**
  - IndexTTS2 `emo_alpha` and vector magnitude [P];
  - α in x-vector arithmetic on Qwen Base (2606.05367; α ∈ {0…2.5}, trades identity) [P];
  - CosyVoice3's "非常" wording only.
- **For us:** start with **2 levels (normal / strong)** for the 4 core emotions, labelled from the emotion2vec class probability or an arousal estimate. Expose α only if the x-vector route (a2) is used. Do not promise more than coarse intensity [I].

---

## 7. What decides (a) vs (a′) vs (b) vs (c)

**The acceptance metric (pre-register):**
- **Set:** about 40 emotionally neutral held-out lines per main character (chosen by the script-context labeller and checked by ear) × {neutral, happy, angry, sad, (surprised, fearful)} × soft flag.
- **Emotion score:** judge recall per class **divided by the same judge's recall on real clips of that class** from the cast.
- **Gates:** identity (cast confusion matrix, as in §10 of the approach doc) per emotion, and pinyin CER.
- **Human check:** a blind "which emotion?" forced choice on a subset.

| Probe | Where, cost | Decides |
|---|---|---|
| **D1** CosyVoice3-0.5B (and RL) `inference_instruct2` with each main character's best reference clip, the 3 documented emotions + soft, on the neutral set | Local, a few hours | The ceiling (b) starts from in our domain. If text-unrelated accuracy stays at the #1687 level (0.4–0.65), (b)'s built-in instruct is not "explicit" enough without RL [I] |
| **D2** Qwen3-TTS-CustomVoice with our locked mean x-vectors written into rows 3000–3013, `generate_custom_voice(..., instruct="用生气的语气说")`, neutral set | Local, an afternoon | Whether (a′) holds identity (confusion matrix vs Base + the same row) and follows emotion. If yes, (a′) becomes the default Qwen arm, fine-tuned with locked rows + instruct replay |
| **D3** Qwen Base x-vector arithmetic: per-(character, emotion) centroids from labelled clips; measure ‖τ_emo‖ vs inter-character distances; generate an α sweep on the neutral set | Local, a day (needs labels) | Whether (a2) per-emotion rows are viable without identity bleed. Also gives intensity. If τ is small relative to neighbour distances and emotion moves, (a) has a training-light path |
| **D4** Labeller calibration: emotion2vec+ large vs script-context LLM labels vs ~300 human-checked clips across characters | Local, a day of listening | The label noise level and per-class judge recall that every other number is divided by. Decides whether surprised/fearful are in the inventory |
| **D5** (H100) Fine-tunes on identical splits: (a) Base + emotion rows/tokens; (a′) CustomVoice + locked rows + instruct replay; (b) CosyVoice3 mSFT with `<角色>。<instruct><|endofprompt|>` prefix + replay + speaker/style masking. Each scored on the D1–D4 metric before and after | Rented, after D1–D4 | The bake-off. Add a GRPO/DPO stage with an SER reward only to the winner if neutral-text accuracy is short |

**How the probes point [I]:**
- **(c) IndexTTS2 is out for a public reference-free release** unless you accept shipping per-character prompt mels, which is a rights question, not a technical fix. It stays as the zero-shot emotion bar in D1-style comparisons.
- **(b) wins** if D1 shows CosyVoice3 instruct is already strong on our neutral lines *and* the E0 smoke test shows `inference_sft` from a stored embedding produces the right timbre. Its documented inventory is still only 3 emotions + soft; adding surprised/fearful means training them anyway.
- **(a′) wins** if D2 shows CustomVoice keeps identity with written rows and follows emotion on neutral text. It keeps the Qwen pipeline and gives the widest instruct vocabulary.
- **(a) wins** if D2 fails on identity but D3 shows clean emotion offsets. Then (a2) rows, optionally with emotion tokens, trained with the existing trainer.
- **In every branch,** neutral-text control will likely need SER-reward post-training to get from about 0.4–0.6 to 0.7+ (all published SFT → RL gaps) [I from §1.1].

---

## 8. Could not verify

1. Any measured emotion-control result for Qwen3-TTS-Base, or for a new speaker added to CustomVoice/VoiceDesign.
2. Whether CustomVoice behaves well with Base-encoder x-vectors in its speaker slot (Rust engine: no evaluation).
3. Any CosyVoice2/3 speaker SFT with instruct accuracy measured before and after.
4. Whether `inference_sft` from an embedding-only `spk2info` produces correct timbre on a fine-tuned CosyVoice3. Checked today [P]:
   - PR #1475 (stored speaker for sft/instruct2) is **open, unmerged**.
   - PR #1700 ("fix cosyvoice3 training of flow and llm", cited in research_B as the #1731 fix) is **closed, unmerged**.
   - The per-row instruct training path is nevertheless in `main` (`prepare_lm_input_target`). PR #1732 itself was closed unmerged.
5. Emotion retention after any IndexTTS2 community fine-tune (#369 is the only data point, negative).
6. The classifier behind macminix's "emotion accuracy" and the size of its emotion eval.
7. Label-noise tolerance of emotion-token training (no ablation exists).
8. Per-class emotion counts in the Tricolour data (needs labelling).
9. Emo-LiPO and EmoSteer-TTS numbers (abstracts only).

---

## 9. Sources

**Qwen3-TTS** [P]
- Issues/discussions: https://github.com/QwenLM/Qwen3-TTS/issues/25 , /issues/14 , /issues/213 , /issues/217 , /issues/248 ; https://github.com/QwenLM/Qwen3-TTS/discussions/121 , /discussions/231 , /discussions/210 , /discussions/168 , /discussions/218
- vspeech trainer: https://github.com/vspeech/Qwen3-TTS-Train (issues #1, #5, #6)
- Rust engine: https://github.com/cgisky1980/Qwen3-TTS-Rust (`src/tts/prompt.rs`, `src/download.rs`)
- HF cards: https://huggingface.co/macminix/qwen3_voice_design_t8 , /t5 , /t6 ; https://huggingface.co/SpragAI/qwen3-tts-emotion-tags ; https://huggingface.co/scrappylabs/narrator-tts ; https://huggingface.co/lunahr/Celune-1.7B-Calm ; https://huggingface.co/ShinyUser/vocence-miner04 ; https://huggingface.co/Danieli1021/vocence_1.0
- Derivative lists: https://huggingface.co/api/models?filter=base_model:finetune:Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (also `adapter:`, VoiceDesign, Base)
- x-vector emotion arithmetic: https://arxiv.org/abs/2606.05367
- Practitioner write-up [S]: https://dev.to/gabrielemastrapasqua/emotions-on-a-cloned-voice-a-25-mb-graft-a-steering-vector-and-a-lot-of-dead-ends-1bfn

**CosyVoice** [P]
- Code: https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/cli/cosyvoice.py , /cli/frontend.py , /llm/llm.py , /utils/common.py , /dataset/processor.py , /example.py
- Issues: https://github.com/FunAudioLLM/CosyVoice/issues/1687 , /1400 , /1475 , /1909 , /1903 , /1057 , /1423 , /1275 , /1300
- Papers: CV3 https://arxiv.org/abs/2505.17589 ; CV2 https://arxiv.org/abs/2412.10117 ; CV3-Eval https://github.com/FunAudioLLM/CV3-Eval
- [S] https://instavar.com/research/tts/cosyvoice-lora-fine-tuning

**Emotion-control papers** [P]
- EMORL-TTS https://arxiv.org/abs/2510.05758
- Emo-DPO https://arxiv.org/abs/2409.10157
- SFT+GRPO emotion/duration https://arxiv.org/abs/2609.11523
- EmoVoice https://arxiv.org/abs/2504.12867
- EmoCtrl-TTS https://arxiv.org/abs/2407.12229
- Marco-Voice https://arxiv.org/abs/2508.02038
- Emo-LiPO https://arxiv.org/abs/2606.13006
- SAE emotion steering https://arxiv.org/abs/2606.01479
- EmoSteer-TTS https://arxiv.org/abs/2508.03543

**IndexTTS** [P]
- https://github.com/index-tts/index-tts (README emotion list; issues #215, #369, #518)
- https://github.com/instavar/indextts2-finetuning
- https://github.com/SrQingChen/IndexTTS-2.5-Pro
- https://github.com/JarodMica/index-tts/tree/training_v2
- https://huggingface.co/feiwell/indextts25-mirror-lora

**Labelling** [P]
- emotion2vec+ large https://huggingface.co/emotion2vec/emotion2vec_plus_large
- EmoBox https://arxiv.org/abs/2406.07162
- C2SER / Emo-Emilia https://arxiv.org/abs/2502.18186
- SenseVoice https://arxiv.org/abs/2407.04051
- EME-TTS https://arxiv.org/abs/2507.12015
- MERaLiON-SER https://arxiv.org/abs/2511.04914
- Qwen3-Omni https://arxiv.org/abs/2509.17765
- M3ED https://arxiv.org/abs/2205.10237
- MERBench https://arxiv.org/abs/2401.03429
- NonverbalTTS https://arxiv.org/abs/2507.13155
- ParaSpeechCaps https://arxiv.org/abs/2503.04713
- Spark-TTS/VoxBox https://arxiv.org/abs/2503.01710
- Vox-Profile https://arxiv.org/abs/2505.14648
- Qwen2.5-Omni AWQ https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ

**Local** [L]
- `E:\STEINS;GATEVoice\kurisu_v2\research\{qwen_internals_conditions,labeling_sota_2026-10,v3_results,v3_synthesis,finetune_methods_2026-10}.md`
- `F:\Tricolour\research\notes\research_C_eval.md` §5
