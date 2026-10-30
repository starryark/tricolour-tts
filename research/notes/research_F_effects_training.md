# Research F: how training and inference should treat effect-bearing clips

Date: 2026-10-30. Scope: training and inference treatment of clips with baked-in production effects (echo/reverb, telephone/radio/"digital", noise) for the 14-character Mandarin Qwen3-TTS-12Hz-1.7B fine-tune. Detection and restoration models are covered by a sibling note.

Tags: [P] primary source read by me; [S] secondary (search summary, aggregator, or second-hand); [I] my inference; [U] unverified; [L] local measurement in this note (script `effects_probe.py`, results `effects_probe.json`, both in this folder). All dates are arXiv/v1 or page dates.

Section map: Q1 = §2, Q2 = §3, Q3 = §4, Q4 = §5, Q5 = §6, Q6 = §7.

---

## 0. Bottom line

1. **Never train an effect-bearing clip under a character's default row, and never let one into that row's locked x-vector.** Both are supported by evidence. Hsu et al. 2019: with a plain speaker lookup, noisy speakers' output was 11.3 dB WADA-SNR against 18.2 dB for clean speakers, so the noise had become identity [P]. Our own probe on Qwen's speaker encoder [L]: one DSP effect moves a character's locked vector **1.2–2.7× the distance to the nearest other character**. Telephone EQ alone pushes 10 of 14 characters' vectors closer to another character's clean centroid.
2. **Contamination of the mean is linear, and it reaches the output.**
   - With 25% effect clips, the vector moves 0.24–0.25 of the full effect shift for every effect [L].
   - With no training, Base output from that vector moves 0.15–0.20 of the DSP effect, against 0.67–0.79 for a 100% effect vector [L].
   - So 10% contamination predicts about 0.07–0.08 output leak before any fine-tuning, close to the A3′ line of 0.10 [I].
   - Keep the effect share in the reference set ≤ 2%.
3. **Effect deltas are nearly character-independent in x-vector space.** The cross-character cosine of the effect shifts is 0.84–0.96. A leave-one-out shared delta rebuilds each character's effect vector at cosine 0.995–0.999 [L]. A per-(character, effect) row can therefore be built as `v_char + δ_effect` without that character having real effect takes.
4. **Prompting/conditioning vs filtering: the published evidence is split.** It depends on whether the label is exact. Audiobox Aesthetics found prompting beat filtering (WER 2.76–2.87 vs 5.06 at p50) [P]. COT-TTS (2026-09) found filtering beat conditioning on *estimated* quality scores (MOS 4.15 vs 3.60) [P]. For our categorical, intentional effects, an explicit label is right. For unintentional noise, drop the clip (or restore it, per the sibling note).
5. **Effects as controls.** No published study compares the perceptual realism of DSP and learned channel/room effects in TTS. Kurisu's planned blind DSP-vs-row A/B **was never run**. Our probe (§3.4–3.5) [L] found:
   - **Phone.** The speaker slot reproduces a band-limited phone effect at 0.75–0.79 with no training. That equals the **12 Hz codec's own ceiling** (0.74–0.78 after a round trip), so a learned narrowband row cannot do better. DSP after generation gives 0.99.
   - **Large reverb.** The slot carries it at 0.70 untrained, and the codec ceiling is about 0.9.
   - **Echo.** A 180 ms inner-voice echo is not induced by the slot (0.06). The codec keeps only a smear of it (cepstral peak 7.7, against 406 for DSP and 4.9 for clean).
   - **CustomVoice instruct** ("用打电话…", "大厅…回声", "内心独白…回声") produced **no** channel or room effect (projections ≈ 0).
   - **Recommendation:** DSP is the v1 mechanism for every effect. A learned show-style phone row is an optional arm that ships only if it beats DSP blind.
6. **LTAS-based gates (A3′/A5) are blind to reverb and echo.** DSP echo changes the 1/3-octave LTAS by 0.1 dB rms, and small-room reverb by 0.5 dB [L]. Yet both move the x-vector by more than one nearest-character distance.
   - Add an x-vector projection gate for every effect.
   - Use the CED tagger for reverb only: it detects 100% of DSP reverb, but 20–30% of phone and 0% of echo.
   - Use a cepstral-peak detector for echo (§6).

---

## 1. Local precedent (Kurisu v3), summarised once

Sources: `E:\STEINS;GATEVoice\kurisu_v2\research\qwen_internals_conditions.md`, `finetune_methods_2026-10.md` §3.4–3.5, `e0_results.md`, `v3_results.md`, `v3_synthesis.md`, `v3/augment_notes.md`, `release/README_v3.md`, `SELECTION_RULE.md` (A3, A5, Amendments B and C) [L].

| Finding | Number | What it does not settle |
|---|---|---|
| The Base x-vector alone carries device EQ (E0) | Projection 1.04 (Amadeus game, tilt +11.4 dB) and 1.22 (phone, +18.3 dB) | Only EQ-type effects were tested. Reverb and echo never were |
| A mixed clean+filtered set gives a useless row | anime-Amadeus x-vector −0.43. In E5, its row's A5 was −0.23 → dropped | Showed the mixed vector failed. Did not show which share of effect clips breaks a row |
| Separate `amadeus`/`phone` rows reproduce the effect with no leak | A5 1.03–1.14 / 1.16–1.31 (phone in-sample); A3′ ≤ 0.06 | The phone row is about 93% our own DSP (400 augmented vs 20–28 real), so it mostly learned our filter |
| Phone augmentation recipe | Measured LTAS curve (phone − game pool), min-phase FIR, ±2.5 dB tilt and 0.9–1.1 band-edge warp per clip, 15% narrowband (300–3400 Hz, 8 kHz μ-law), 400 clips over 48 strata, each source used once, clean originals kept | Ratio not ablated |
| DSP-vs-learned blind A/B | Planned in `v3_synthesis.md` §2 ("DSP may well win") | **Never run.** No result in `rating_results.md` or `final_report.md` |
| Instruct probe on the published checkpoint (§2.5 of internals) | Planned | **Never run** |
| Background-SFX tokens | 0–7% detection at 5% FPR; A3 failed → dropped | — |
| A3 (raw LTAS vs v2) measured retraining drift, not leakage | E3 rms 1.07 dB failed with no device data → replaced by A3′ (projection onto the device direction ≤ 0.10) | A3′ only sees spectral effects (see §5) |

---

## 2. Q1: keep, filter, restore or condition

### 2.1 Evidence

| Work (date) | Setting | What happened to the default / clean output | Numbers | Tag |
|---|---|---|---|---|
| **Hsu et al., ICASSP 2019** ([pub](https://research.google/pubs/pub47834/), [pdf](https://sls.csail.mit.edu/archives/root/publications/2019/Wei-NingHsu_ICASSP-2019.pdf)) | Tacotron 2, VCTK. Half the speakers had **all** their clips mixed with CHiME-4 noise at 5–25 dB (fully correlated) | Speaker lookup baseline: noise became identity | WADA-SNR of output: clean speakers 18.16 dB, noisy speakers **11.26 dB**. Full model (factorised latent + augmentation + adversarial), clean residual: 18.62 / 18.35. Augmentation only (no adversarial): 18.64 / 17.03. Neither: 18.78 / **9.49** (noise follows the speaker). MOS: baseline 3.22, proposed with clean residual 4.52, with noisy residual 1.32 | [P]. Row labels in the pdftotext output were shifted; I aligned them against the prose ("−adv" = 85% noise on zs, 96.5% on zr) |
| **Environment-aware TTS** ([2110.03887](https://arxiv.org/abs/2110.03887), 2021-10) | VCTK × measured RIRs, 108 speakers × 108 environments, **one environment per speaker** (fully entangled) | Separate speaker and environment embeddings let unseen speaker–environment pairs be generated | Unseen pairs: speaker-similarity MOS 3.18 vs baseline 1.98; environment similarity 3.25 vs 2.33 | [P] |
| **VALL-E** ([2301.02111](https://arxiv.org/abs/2301.02111), 2023-01) | Codec LM, in-context prompt | "VALL-E could synthesize speech with reverberation as well, whereas the baseline outputs clean speech"; it "could learn the acoustic consistency" | Qualitative only | [P] |
| **Parler-TTS** ([2402.01912](https://arxiv.org/abs/2402.01912), 2024-02) | SNR and C50 binned into words | Only ~1% of 45k h clean, yet "very clear" prompts gave PESQ 3.84 | (from Kurisu notes; not re-read) | [P via local note] |
| **Wang et al., noise robustness of FM zero-shot TTS** ([2406.05699](https://arxiv.org/abs/2406.05699), 2024-06) | Noisy prompts | Baseline copies the prompt's noise into the output | Noisy-prompt DNSMOS: baseline 3.01 (clean prompt 3.33); SE front-end 3.28 but SIM-o 0.60; noise-augmented training 3.32, SIM-o 0.65 | [P] |
| **Audiobox Aesthetics** ([2502.05139](https://arxiv.org/abs/2502.05139), 2025-02, §5) | LibriTTS, FM model; filter below PQ percentile vs prefix "Audio quality: yy" | **Prompting beat filtering** | WER: baseline 2.95, filter p25 3.37, filter p50 **5.06**, prompt 2.76–2.87. Net win vs baseline: filter p50 15.8%, prompt p90 r5 45.1%. "Prompting wins all versus filtering" | [P] |
| **COT-TTS** ([2609.22697](https://arxiv.org/abs/2609.22697), v2 2026-09-25) | 0.6B; quality-score (DNSMOS/UTMOSv2) conditioning vs quality filtering + 1M high-quality refinement | **Filtering beat conditioning** | EN: UTMOSv2 2.684 vs 2.943; WER 0.063 vs 0.041; duration error 5.03 vs 1.16; human MOS 3.60 vs 4.15. "improving the training data itself is more effective than introducing estimated quality scores as additional conditions" | [P] |
| Raon-OpenTTS ([2605.20830](https://arxiv.org/abs/2605.20830), 2026-05) | DNSMOS+WER+speech-ratio filter | Removing the bottom 15% beat removing 50% | — | [S] |
| **DAIEN-TTS, extended** ([2608.03011](https://arxiv.org/abs/2608.03011), 2026-08-04; earlier [2509.14684](https://arxiv.org/abs/2509.14684)) | F5-TTS + speech/noise/reverb separation; separate speaker and environment prompts; triple CFG | With a silence environment prompt and a **noisy-reverberant speaker prompt**, output stays clean and keeps more identity | SECS: F5-TTS 0.317, DAIEN-NR 0.486 (clean prompt 0.628); WER 2.80 → 2.24. CFG trade-off: reverb scale 0 → 9 moves FAD 10.38 → 1.48 but WER 5.9% → 25.6%. **No leakage metric, no C50/DRR** | [P] |
| **VoiceDiT** ([2412.19259](https://arxiv.org/abs/2412.19259), ICASSP 2025) | LibriTTS-R + WavCaps noise (SNR 2–10) + RIRs ("with a certain probability"); CLAP-encoded environment | No clean-output leakage test | AC-Filtered: WER 7.09%, CLAP 0.22, FAD 4.60, environment-relevance MOS 3.86 (GT 4.26) | [P] |
| **UmbraTTS** ([2506.09874](https://arxiv.org/abs/2506.09874), 2025-06) | F5-TTS, joint speech+background | Only a background-mix A/B: **87.89% preferred joint generation** over F5-TTS + post-hoc mixing (30 samples, ≥20 raters). RIR or EQ not tested | WER 6.89, CLAP 0.37 | [P] |
| Speak in the Scene / AST-LDM ([2406.12688](https://arxiv.org/abs/2406.12688), 2024-06) | Acoustic scene transfer, CLAP condition | No DSP (RIR or noise) baseline | Env→Clean WER 22.6%, SSM 0.718 | [P] |
| **SwanTale** ([2608.02023](https://arxiv.org/abs/2608.02023), v2 2026-08-04) | Caption "Environment" field: "microphone characteristics, reverberation, background music…" | Reports a "Reverb Consistency" metric (SD of SRMR). I could not extract the SwanTale value | — | [P, partial] |
| **Poly-InstructTTS** ([2608.20387](https://arxiv.org/abs/2608.20387), 2026-08 by ID; the page stamp read as 2026-06-30 [U]) | CosyVoice-style. Training audio **denoised** (Demucs + ClearerVoice) instead of labelled | Evaluated on InstructTTSEval-expanded "In-the-wild (Telephone, Echo…)"; reports "weaknesses under challenging acoustic conditions (e.g., noise, echo)" | Aggregate APS/DSD/RP only | [P] |
| EnvTTS (community, 2026) ([HF card](https://huggingface.co/ChristianYang/LongCat-AudioDiT-Env-TTS-1B-ablation)) | LongCat-AudioDiT-1B LoRA; environment + speaker reference + captions | Not a paper; no leakage test | CER 0.103, SIM 0.702, CLAP 0.711 (base 0.638 / 0.601 / 0.541) | [S] |

### 2.2 Production systems with channel or room controls (2025–2026)

| System | Control | Mechanism | Numbers | Tag |
|---|---|---|---|---|
| **Irodori-TTS** (Aratako, palette file 2026-05-12) | 📢 "エコー / リバーブ" (echo/reverb), 📞 "電話越し / スピーカー越し" (through a phone/speaker) | Learned emoji tokens in text | None; the v4.1 card says the effect of emoji control "varies with context" | [P code: `irodori_tts/gradio_emoji_palette.py`] |
| **Higgs Audio v2** (Boson AI, 2025) | `<|scene_desc_start|>Audio is recorded from a quiet room.<|scene_desc_end|>` in the system prompt | Learned text description | None | [P README_V2] |
| **MiniMax Speech** (platform docs, OpenAPI v1.0.0) | `voice_modify.sound_effects` ∈ {spacious_echo, auditorium_echo, lofi_telephone, robotic}, one at a time | Per-request effect; that it is post-hoc DSP is my reading [U] | None | [P docs] |
| **Fish Audio S2** | The official emotion reference has **no** channel or room tag; only `[audience laughing]`-type SFX | — | — | [P docs]. An aggregator lists `[echo]` [U] |
| **ElevenLabs v3, CosyVoice 2/3, IndexTTS-2, Step-Audio-EditX** | No documented channel or room tag found | — | — | [S] (search only) |
| **Qwen3-TTS** | No documented environment instruct. CustomVoice instruct covers emotion and prosody (issue #248); 12 Hz Base instruct is "relatively unstable" (#25) | — | See §3.4 for our zero-shot probe | [P via local note] |

**Which have numbers?** For channel/room control specifically:
- DAIEN-TTS (SECS, WER, CLAP, FAD; noise and reverb separately);
- VoiceDiT and UmbraTTS (CLAP, FAD, WER);
- Environment-aware TTS (MOS).

None reports how much effect leaks into the clean/default output. Hsu 2019 and Wang 2024 are the only ones that measure an effect bleeding into output that should be clean (WADA-SNR, DNSMOS).

### 2.3 Verdict for Q1 [I]

- **Intentional, categorical effects (phone, radio, echo, reverb):** *condition or drop, never leave unlabelled under the default row.*
  - Audiobox and Parler support conditioning; COT-TTS's counter-result is about conditioning on noisy *estimated* continuous scores.
  - Our labels can be exact: DSP augmentation labels are exact by construction, and real effect clips get detector labels.
  - Given one speaker slot, "condition" means a separate row (or DSP; §3).
- **Unintentional noise:** drop it, or restore it if the sibling's restoration passes its own gates.
  - It is not a feature users will request, and every filter result above was done at much larger scale. Losing a few percent of 22 h is cheap.
- **Restoring effect clips back to "clean" and training them under the default row** is the riskiest option. Dereverberation and bandwidth-extension artefacts become identity in exactly Hsu's sense. Use it only for clips that pass the same metrics (§6) as clean clips. The cheapest is the per-clip δ-projection of §7.3: a restored clip should sit inside the clean spread (SD 0.06–0.16).

---

## 3. Q2: do speaker embeddings and codec LMs absorb channel and room?

### 3.1 Literature

- **x-vectors.** Raj et al., ASRU 2019 ([1909.06351](https://arxiv.org/abs/1909.06351)) [P]: "x-vectors capture spoken content and channel-related information".
  - Session identification reached about 45% (of 848 sessions); errors went to other sessions of the same speaker with similar acoustics.
  - x-vectors trained without augmentation "capture more non-speech information"; augmentation "trades this noise-related information for additional speaker information".
- **Other deep embeddings.** Zhao et al. 2022 ([2212.07068](https://arxiv.org/abs/2212.07068)) [P, abstract]: "all deep embeddings [d-vector, x-vector, ResNet, ECAPA] encoded channel and content information in addition to speaker identity". TTS was "less sensitive to the choice of speaker representations".
- **Codec embeddings.** Grundhuber & Habets, Interspeech 2026 ([2606.31365](https://arxiv.org/abs/2606.31365)) [P, abstract]: in an acoustic-teleportation codec, "acoustics leak into the speech embeddings due to the training objective". Room parameters can be read back blindly to within 0.02 s.
- **Codec LM in context.** VALL-E keeps the prompt's reverberation (above) [P].
- **Qwen specifically:**
  - The speaker encoder is `Qwen3TTSSpeakerEncoder` [P code].
  - The tech report says "we jointly train a learnable speaker encoder with the backbone" ([2601.15621](https://arxiv.org/abs/2601.15621)) [P]. It is therefore optimised for reconstruction, not for channel-invariant verification. One would expect it to keep the recording channel, because that helps reconstruction [I].
  - The report's only data-quality statement is a quality-stratified continual pre-training stage that "alleviates hallucinations caused by noisy data" [P].
  - Kurisu E0: its vector alone reproduces device EQ in Base's output [L].

### 3.2 Local measurement: Qwen's speaker encoder on our cast [L]

**Setup** (`effects_probe.py --stage emb`):
- 14 roster characters, 40 clips each: the first 40 of the 60 references behind `tts_work/xvec_rows.pt`.
- Seven DSP effects were applied to each clip, and locked vectors were computed exactly as `train_sft.py` does.
- Distances are compared with each character's distance to its **nearest other character** (nn).
- Split-half noise floor: two disjoint 20-clip halves of the same clean character differ by **0.46 nn** (median).

The DSP is generic, not fitted to the game:
- `phone_nb`: 300–3400 Hz, 8 kHz μ-law.
- `phone_eq`: 600 Hz 4th-order high-pass, +presence 1.4–5 kHz, 9 kHz low-pass. Harsher than Kurisu's measured show curve.
- `radio`: 400–5000 Hz + tanh saturation + noise at 30 dB.
- `reverb_small`: RT60 0.4 s, DRR +6 dB.
- `reverb_large`: RT60 1.5 s, DRR −3 dB.
- `echo`: feedback delay 180 ms, 4 repeats at 0.5^k.
- `noise15`: pink noise at 15 dB SNR.

| Effect | Shift / nn distance (median, range) | cos(effect, clean) | Identity kept (raw / centred) | Cross-character cos of shifts | Shared-δ rebuild cos (vs clean-only cos) | 25% mix projection | LTAS shift rms (dB) |
|---|---|---|---|---|---|---|---|
| phone_nb | **2.63** (2.20–3.51) | 0.927 | 10/14 · 9/14 | 0.93 | **0.9955** (0.927) | 0.248 | 20 |
| phone_eq | **2.59** (1.98–3.96) | 0.937 | **4/14 · 4/14** | 0.96 | 0.9973 (0.937) | 0.252 | 20 |
| radio | 2.21 (1.76–3.14) | 0.947 | 7/14 · 9/14 | 0.92 | 0.9961 (0.947) | 0.249 | 5 |
| reverb_small | 1.36 (1.13–1.78) | 0.981 | 12/14 · 12/14 | 0.95 | 0.9988 (0.981) | 0.250 | **0.5** |
| reverb_large | 2.43 (1.96–3.04) | 0.937 | 10/14 · 7/14 | 0.93 | 0.9954 (0.937) | 0.245 | 1.9 |
| echo | 1.19 (0.97–1.36) | 0.985 | 11/14 · 11/14 | 0.84 | 0.9974 (0.985) | 0.244 | **0.1** |
| noise15 | 2.73 (2.04–3.29) | 0.926 | 6/14 · 5/14 | 0.95 | 0.9960 (0.926) | 0.247 | 2–7 |

"Identity kept" asks whether the effect vector's nearest clean centroid is still the same character. The residual of the shared-δ rebuild is 0.22–0.42 of the shift. In nn units that is 0.33 (small reverb) to 0.65, i.e. **0.7–1.4× the 0.46 split-half floor**. The shared δ rebuilds a character's effect vector about as well as a second half-sample of the same character's clean clips would rebuild the clean vector.

**Reading [I]:**
1. **Every effect moves the vector further than the gap between neighbouring characters.** Raw cosines (0.93–0.99) look high only because the cast's centroids sit at cosine 0.985–0.993 to their nearest neighbour. Judged in nn units, an all-phone or all-noise vector is a different "speaker".
2. **Contamination is exactly linear** (25% mix → 0.24–0.25 of the full shift). A 10% phone contamination moves the vector 0.26 nn along the phone direction. That is below the split-half noise (0.46 nn), but systematic and aimed at the one direction the decoder renders as EQ. If output projection scales the same way, Kurisu's 1.0–1.2 output projection at 100% implies about 0.10–0.12 output leak at 10% contamination: right at the A3′ line.
3. **Effect shifts are mostly shared across characters** (cosine 0.84–0.96). Two consequences:
   - (a) A per-(character, effect) row can be built as `v_c + δ_e`, with δ_e estimated on other characters' DSP clips (rebuild cosine ≥ 0.995).
   - (b) A character whose real clips are mostly effected can be partly *cleaned* as `v_c,effect − δ_e` [I, untested in output].
4. **Spectral metrics cannot see echo or small-room reverb** (0.1 and 0.5 dB LTAS rms), but the speaker encoder can (1.19 and 1.36 nn). So effect clips the LTAS screen passes can still contaminate the vector.

### 3.3 Consequences for computing the locked vectors [I]

- Build each default vector from **clean-only** references: tier A, passing the sibling's detectors at a conservative threshold, preferring studio-dry scenes.
  - With 60 references per character, one stray effect clip is 1.7%.
- **Check the reference set itself.** Project `v_c − v_c(strict subset)` onto each δ_e (from §3.2). Require ≤ 0.05 of the full shift (≈ 0.1 nn for phone). This is cheap and needs no generation.
- Channel-robust encoders exist, but none helps a locked-vector design:
  - Fujita et al., ICASSP 2024, [2401.05111](https://arxiv.org/abs/2401.05111): SSL adapters fine-tuned with noisy references;
  - Cong et al., Interspeech 2020, [2008.04265](https://arxiv.org/abs/2008.04265): domain-adversarial training;
  - DAIEN's cross-speaker conditioning.
  
  We cannot swap Qwen's encoder: the codec LM was trained on its space. Selecting clean references is the lever.

### 3.4 Output-level probe (Base rows, CustomVoice instruct, codec ceiling) [L]

**Setup** (`effects_probe.py --stage gen/score`; 2026-10-30, RTX 5060 Ti, bf16, `qwen-tts` 0.1.1 with transformers 4.57.3):
- **Characters and texts:** moxiaoju (F), chiguo (F) and luoyan (M) × 10 neutral Mandarin probe texts (`tricolour-tts/eval/probes_neutral_zh.txt`) × 1 seed, paired across conditions. n = 30 per condition.
- **Base, x-vector-only mode, no training.** The row vector is the locked mean over each character's 40 reference clips, either clean or passed through one DSP effect. Two "mix25" vectors use 30 clean + 10 effect clips.
- **CustomVoice:** the clean vector written into a speaker row, plus an instruct: phone "用打电话时从听筒里传出来的声音说", hall "在空旷的大厅里说话，声音有明显的回声和混响", inner voice "像内心独白一样说，声音带着回声". Compared against the same row with no instruct.
- **Metrics:**
  - the LTAS projection and x-vector projection of §6.1, against the DSP direction measured on the real clips;
  - CED detection at 5% FPR, with the threshold set on the 30 reference outputs;
  - **the same DSP applied to the clean-vector outputs**, as the fidelity ceiling.

| Condition | LTAS proj. | x-vector proj. | CED detect @5% FPR | Runaways |
|---|---|---|---|---|
| Base + phone_nb vector | **0.75** | **0.79** | 27% (Telephone/Radio) | 0 |
| Base + phone_eq vector | 0.52 | 0.69 | 23% | 0 |
| Base + reverb_large vector | **0.67** | **0.70** | **67%** (Reverberation / large room) | 0 |
| Base + echo vector | n/a¹ | **0.06** | 10% (Echo) | 0 |
| Base + mix25 phone_nb | 0.17 | 0.20 | 7% | 0 |
| Base + mix25 reverb_large | 0.13 | 0.15 | 3% | 0 |
| CV + phone instruct | 0.01 | 0.00 | 0% | 0 |
| CV + hall instruct | 0.22 | −0.02 | 3% | 0 |
| CV + inner-voice instruct | n/a¹ | −0.02 | 13% (Echo) | 0 |
| *DSP phone_nb on Base clean outputs (ceiling)* | 0.99 | 0.98 | 30% | — |
| *DSP phone_eq on Base clean outputs* | 1.00 | 1.05 | 20% | — |
| *DSP reverb_large on Base clean outputs* | 0.96 | 1.08 | **100%** | — |
| *DSP echo on Base clean outputs* | n/a¹ | 1.09 | **0%** | — |

¹ The echo LTAS direction is 0.1 dB rms, so its projection is division by near-zero.

**Reading:**
1. **The locked x-vector carries large-room reverb, not just EQ.** With no training at all, a reverb-mean vector makes Base produce 0.67–0.70 of the DSP reverb shift, and CED flags 67% of outputs. Kurisu tested only EQ, so this is new [L].
2. **It does not carry the feedback echo** (x-vector projection 0.06). A discrete 180 ms delay is not something the speaker slot can induce zero-shot. Whether the 12 Hz codec can even represent it is in §3.5.
3. **Contamination reaches the output roughly linearly, at slightly under full strength.** A vector with 25% effect clips gives 0.15–0.20 output projection (expected 0.25 × 0.67–0.79 ≈ 0.17–0.20).
   - Untrained, 10% contamination therefore predicts about 0.07–0.08 output leak.
   - Kurisu's fine-tune did not shrink the row effect (E0 1.04–1.22 → v3 1.03–1.31), so training should not be expected to shrink a contamination leak either [I].
4. **CustomVoice instruct produces no channel or room effect zero-shot** (projections ≈ 0, detection at the false-positive floor). The phone, hall and inner-voice instructs are ignored acoustically. The hall instruct's small LTAS change has no x-vector or tagger support, so it is probably a delivery change [I].
5. **CED is a good reverb detector but a poor phone and echo detector here.** On *DSP-ceiling* outputs it detects 100% of reverb, 20–30% of phone and 0% of echo. Use it only for reverb, use LTAS/bandwidth for phone, and the cepstral peak for echo (§6).
6. **No runaways** in any condition (n = 330 generations: 7 Base vectors + 4 CustomVoice conditions × 30).

### 3.5 Codec ceiling: what the 12 Hz tokenizer itself can carry [L]

**Setup** (`--stage codec`, `--stage cep`):
- 30 further real reference clips, the same 3 characters but **not** used in §3.2.
- Each was DSP'd, then round-tripped through `Qwen3-TTS-Tokenizer-12Hz` (encode → decode), and scored against the DSP original.
- A learned row can at best emit what the codec can represent.

| Effect | DSP original: LTAS / x-vec proj. | **After codec round trip**: LTAS / x-vec proj. | Other |
|---|---|---|---|
| phone_nb (300–3400 Hz, μ-law) | 1.01 / 1.01 | **0.74 / 0.78** | Base + phone vector already reaches 0.75 / 0.79 (§3.4) |
| phone_eq (deep bass cut, −45 dB at 100 Hz) | 1.00 / 1.04 | **0.54 / 0.67** | Base + vector 0.52 / 0.69 |
| reverb_large | 0.97 / 0.98 | 0.92 / 0.89 | CED reverb 100% → 100% |
| echo (180 ms feedback) | — / 1.04 | — / 0.93 | Cepstral peak at 180 ms: clean ≈ 4.9; DSP ≈ 406; **codec 7.7** (detected in 87% at the clean 95th percentile) |

**Reading:**
1. **Deep band-limiting is capped by the codec at about 0.55–0.75 of the target.** The decoder regenerates energy outside the passband. Untrained Base with an effect vector already sits *at* that ceiling.
   - A learned narrowband phone row cannot get closer to a true telephone band than about 0.75, while DSP after generation gives 0.99.
   - Kurisu's phone row did fine because the show's phone sound is a gentler shelf (−20 to −24 dB below 500 Hz), not a hard band.
   - So: gentle EQ can be a row; a hard telephone band should be DSP [I].
2. **Reverb survives the codec almost intact (0.89–0.92).** A trained reverb row has headroom over the untrained 0.70. So reverb *can* be a learned condition; whether it should is a leak and realism question (§4).
3. **A discrete echo survives only as a smear.** The speaker encoder still "hears" it (0.93), but the cepstral peak drops from about 406 to 7.7, i.e. the repeats are no longer clean copies. A learned echo row would at best reproduce that smeared doubling, while DSP reproduces the exact delay [L, I].

---

## 4. Q3: effects as controls

### 4.1 The structural constraint

- Qwen3-TTS has **one** speaker slot in the prefix (MD:2166-2172). A shared "effect row" in a second slot does not exist without changing the prefix layout, which breaks `generate_custom_voice` compatibility.
- So the real options are:
  - **(a1) Per-(character, effect) speaker rows.** 14 × k rows; 716 untrained control rows exist, so capacity is not the limit.
  - **(a2) A global text-side effect token** (an atomic added token, like Kurisu's tags). For a whole-utterance channel this is untested on this model.
  - **(a3) CustomVoice instruct.** Usable only if it already responds, because instruct from scratch needs far more data (Kurisu internals §2.4). It does not respond (§3.4).
  - **(b) DSP after generation.**
  - **(c) Both.**

### 4.2 Evidence on DSP vs learned

- **Nothing published** measures perceptual realism of DSP vs learned telephone/reverb/echo in TTS. The nearest result, UmbraTTS's 87.89% preference for joint generation, concerns *background sound* (Lombard and integration effects), not EQ or room [P].
- **Kurisu.** The learned rows matched the real takes' spectral shift, but the phone row was 93% DSP-augmented, and the DSP-only control A/B was never run [L]. So whether DSP-only lost anything perceptually is unknown.
- **What a learned row can add:** delivery correlated with the effect. Inner monologue tends to be softer and closer-miked; phone lines are often more projected. With tens of real effect takes per character, little of that will be learned [I], and delivery is better handled by emotion/style conditioning.
- **What DSP gives:**
  - exact parameters (delay time, feedback, RT60, wet/dry) that a row cannot expose;
  - zero leak by construction;
  - composability with every character and future rows;
  - no training cost.
  
  Visual-novel effects were very probably made by plug-ins in post-production, so fitted DSP *is* the generating process [I]. Fit the parameters from real effect takes: EQ by LTAS ratio as in Kurisu; echo delay by cepstral peak; reverb RT60 and DRR by blind estimation.

- **Practice in games and VN tooling** [S]: inner monologue is typically a post effect (echo, pitch, close-mic dryness). The Audacity forum, Kokoro "Inner Thoughts" presets and Roblox AudioHelper all add reverb or echo as a switchable effect.

### 4.3 What the probe adds [L]

| Option | Phone (narrowband) | Phone (gentle EQ, show-style) | Big-room reverb | Inner-voice echo |
|---|---|---|---|---|
| (a1) per-(char, effect) row | Capped ~0.75 by the codec | Works (Kurisu 1.16–1.31) | Works zero-shot at 0.70; codec ceiling 0.9 | Not induced zero-shot (0.06); codec keeps only a smear |
| (a3) CustomVoice instruct | 0 | 0 | 0 | 0 |
| (b) DSP after generation | 0.99 | ≈1 by construction | 0.96–1.08 | Exact |
| Leak risk to default | Rows: yes if mis-trained. DSP: none | same | same | same |

### 4.4 Verdict for Q3 [I]

- **(b) DSP is the default for every effect in v1.** It is the only option at or near 1.0 for all four effects, with zero leak.
- **(a1) A learned row is justified only where DSP is likely to sound worse:**
  - The candidate is the show-style gentle phone EQ, where real takes may carry delivery.
  - Big-room reverb only if real takes are plentiful. The row's appeal there is room ambience integrated with delivery, which DSP lacks; but no evidence says listeners hear a difference.
- **(c) Both:** train the learned row with DSP augmentation (Kurisu recipe) and A/B it blind against DSP. Ship whichever wins. "DSP wins" is an acceptable result.
- **No shared effect text token in v1.** It is untested on this model, and the Kurisu tag evidence (13–14% detection for point events) is weak. The one-slot constraint makes per-character rows the native route.

---

## 5. Q4: augmentation for decorrelation

The rough size of the problem, from the existing local features (`tts_work/b07b_ltas.jsonl`, `b08_ced.jsonl`) [L, unvalidated]:
- Telephone-like LTAS clips (high bands < −10 dB and low bands < −6 dB vs the character median) number **0–72 per roster character**, under 3% of their clips.
- CED echo or reverberation > 0.3 fires on ≤ 6 clips per character.
- Real effect takes per (character, effect) are therefore tens, not hundreds. Any learned effect row will be mostly augmentation, as Kurisu's phone row was. The sibling's detection note supersedes these counts.

| Work | What is augmented | Ratio / probability | Range | Result | Tag |
|---|---|---|---|---|---|
| Hsu 2019 | **Every** training clip gets one noisy copy (2× data), labelled original/augmented | 1:1 | CHiME-4, 5–25 dB | Augmentation alone: clean-residual output of noisy speakers 17.03 vs 18.64 dB for clean speakers; adversarial term closes it to 18.35 | [P] |
| DAIEN-TTS 2026 | LibriTTS 580 h; noise and RIR applied independently | p_noise = 0.5, p_rir = 0.5 | DNS noise 180 h, −5…15 dB; DNS RIRs 20,248 | see §2.1 | [P] |
| Wang 2024 | Pre-training targets get noise; fine-tuning puts noise **on the prompt only**, target clean | p = 0.5 (both) | DNS noise, 0–20 dB pre-train, −5…20 dB fine-tune | Noisy-prompt DNSMOS 3.01 → 3.32 | [P] |
| VoiceDiT | WavCaps noise (340k clips) + RIR | RIR "with a certain probability" | SNR 2–10 | — | [P] |
| AST-LDM | Speech + environment sound | — | 4–20 dB | — | [P] |
| Environment-aware TTS | VCTK × REVERB/Aachen RIRs | 108 speakers × 1 environment each | measured RIRs | §2.1 | [P] |
| Kurisu v3 phone | 400 clean clips → measured phone EQ (+ jitter, 15% narrowband), labelled `phone`; 80 real rows (20 × 4) | ≈ 4.4% of 10,970 rows for one effect | ±2.5 dB tilt, 0.9–1.1 warp | A5 1.16–1.31, A3′ ≤ 0.06 | [L] |

**What is not published:** a recipe for *"default stays clean, k effects on demand"* in a fine-tune of about 20 h. Every published ratio above comes from systems where the environment is a generation **target** at full scale. **Nothing published**: [U].

**Design rules that do transfer [I]:**
- Augment **all** characters, not only those with real effect takes. Decorrelation is the point (Hsu, Environment-aware).
- Each source clip is used once. The clean original stays under the default row. Augmented copies go only under the effect condition (Kurisu invariant).
- Jitter the effect parameters per clip (Kurisu tilt and warp), so the row learns a family rather than one filter fingerprint.
- Wang 2024's prompt-only augmentation (noisy *conditioning*, clean *target*) maps onto our case as: compute the default vector over clean clips only, and never let an effect clip be a training target under it.

---

## 6. Q5: evaluation of leakage and fidelity

### 6.1 Which metric sees which effect [L, I]

| Effect | LTAS projection (A3′/A5) | x-vector projection | Temporal / signal metric | Tagger (CED AudioSet) | Quality predictor |
|---|---|---|---|---|---|
| Phone / radio (EQ, band limit) | **Yes** (20 dB rms LTAS shift) | Yes | Energy above 4 kHz / below 300 Hz | "Telephone", "Radio" | NISQA *Coloration* |
| Large reverb | Weak (1.9 dB) | Yes (2.4 nn) | C50/DRR (Brouhaha), SRMR | "Reverberation", "Inside, large room or hall" | DNSMOS SIG/OVRL drop |
| Small reverb | **No** (0.5 dB, under retraining drift) | Yes (1.4 nn) | C50 | Weak | — |
| Echo / feedback delay | **No** (0.1 dB) | Yes (1.2 nn) | **Cepstral peak at the delay** (known lag) | "Echo" | — |
| Noise | Partly | Yes (2.7 nn) | Brouhaha SNR, WADA-SNR | "Noise", "Static", "Hum" | DNSMOS BAK, NISQA *Noisiness* |

**x-vector projection** is the one metric sensitive to every effect here, because the speaker encoder moves strongly for all of them (§3.2). It uses the same encoder that builds the rows. Define it like A3′:

```
proj_xv(e) = mean_texts[(xv(output) − xv(reference output)) · δ_e] / |δ_e|²
```

where δ_e = v_c,effect − v_c,clean comes from DSP'd real clips.
- 1.0 means the output sits where an all-effect reference set would put it.
- Leak gate on default rows: ≤ 0.10.
- Fidelity gate on effect rows: ≥ 0.5.
- **Never gate on it alone.** The talker is trained in the conditioning encoder's space, so after fine-tuning a row's output could drift along δ in x-vector space more than the audio itself changes. The probe shows it is not trivially saturated (the echo vector stayed at 0.06). Always pair it with an acoustic metric for the same effect: LTAS, cepstrum, C50 or the tagger [I].

**Caveat (codec artefact):** 12 Hz Qwen outputs have a periodic ~40 ms cepstral peak, 10–21× background ([Qwen3-TTS #386](https://github.com/QwenLM/Qwen3-TTS/issues/386), 2026-10-22) [S]. A cepstral echo detector must use lags well clear of 40 ms (ours: 180 ms).

### 6.2 Thresholds others used

| Metric | Threshold | Source |
|---|---|---|
| LTAS projection onto the device direction, default voice | ≤ 0.10 (A3′) | Kurisu Amendment C [L] |
| LTAS projection, effect row | ≥ 0.5 (A5) | Kurisu [L] |
| Tagger detection | Threshold at 5% FPR on the same checkpoint's untagged outputs; ship if ≥ 50% (A7) | Kurisu A6/A7 [L] |
| C50 bins (Brouhaha) | Edges 10 / 35 / 45 / 55 / 59 / 60 dB → "very distant-sounding" … "very close-sounding" | DataSpeech `v02_bin_edges.json` [P code] |
| SNR bins (Brouhaha) | Edges 17.1 / 25.4 / 33.7 / 41.9 / 50.2 / 58.5 / 66.8 / 75.0 dB → "very noisy" … "very clean" | DataSpeech [P code] |
| DNSMOS BAK | > 3.5 for "clean" speaker prompts, < 2.5 for environment prompts | DAIEN-TTS 2608.03011 [P] |
| DNSMOS OVRL | ≥ 3.0 to keep a segment (`min_dnsmos=3`) | Emilia-Pipe, Amphion `utils/tool.py` [P code] |
| WADA-SNR on outputs | Reported, no threshold (clean ≈ 18–19 dB, leaked ≈ 9–11 dB) | Hsu 2019 [P] |

DNSMOS, UTMOS and NISQA are unreliable on expressive speech (`research_C_eval.md`). Use them only as regression alarms, not as effect-fidelity gates [L].

### 6.3 Recommended measurement set [I]

- **Leak (every default row, paired against a no-effect-data control run):**
  - LTAS projection onto each EQ-type δ;
  - x-vector projection onto each δ;
  - CED detection rate per effect class at 5% FPR, with the threshold set on the control run;
  - echo cepstral peak.
- **Fidelity (every effect condition):** the same metrics against the same DSP reference, plus CER and SIM to the character's clean row.
- **Fidelity ceiling:** apply the fitted DSP to the default outputs and score them. Any learned condition is judged against that ceiling, not against 1.0.

## 7. Q6: recommendation for the 14-character case [I]

### 7.1 The governing constraint

- Qwen has one speaker slot, so every learned effect is a per-(character, effect) row: 14 per effect.
- The effect deltas are nearly character-independent (§3.2), so these rows can be **built** as `v_c + δ_e` and **trained** with DSP-augmented clean clips of all 14 characters.
- That makes a learned row cheap to *construct*, but every one of them is another way to mis-train. DSP needs none of this.

### 7.2 Per-effect decision

| Effect class | Training data treatment | v1 inference | Rows / tokens | Why |
|---|---|---|---|---|
| **Inner voice / echo** | **Drop** every echo-flagged clip from training and from references. Restored versions only if they pass G1 below | **DSP preset(s)** fitted to the games: delay and feedback from the cepstral peak of real echo takes, plus light room and optional slight low-pass | None | Row cannot induce it (0.06); codec smears it; CV instruct ignores it; DSP exact |
| **Big room / flashback reverb** | **Drop** from default rows and references | **DSP convolution** with an IR fitted to real takes (RT60/DRR blind estimates, LTAS match), 1–2 presets | None in v1; optional `<c>_room` arm later if real takes ≥ ~100 per character | A row works (0.70 untrained) but leak-prone; DSP ceiling ≈ 1 |
| **Telephone / "digital"** | Real phone takes: **experimental arm only** → `<c>_phone` rows. Otherwise drop. Never under the default row | **DSP**: fitted show EQ (Kurisu method: LTAS of real phone takes − matched clean takes), optional narrowband + μ-law preset | Optional 14 `<c>_phone` rows, experimental, ship only on G3 | Row works for gentle EQ; codec caps hard band-limit |
| **Radio / PA** | Drop | DSP (band-pass + saturation) | None | Few clips expected; DSP trivial |
| **Unintended noise** | Drop (or restore per the sibling note, then G1) | — | None | Not a user feature |

### 7.3 Locked vectors

- 60 references per character, as in `eval/xvectors.py`: tier A, 2.5–8 s, round-robin over games.
- **Exclude any clip flagged by the sibling's detectors** at a high-recall threshold. Prefer scenes with no effect tag.
- **δ-projection hygiene** [L, new]:
  - The per-clip projection onto each shared δ_e separates DSP'd from clean clips at **AUC 1.00** (phone, radio, large reverb, noise), with a minimum of 0.88 for small reverb and echo.
  - Clean clips spread with SD 0.06–0.16 along δ.
  - Flag references with projection > 0.5 on any δ and **listen before excluding**. Of the current 560 references × 3 deltas, 32 were flagged. Some look like loud or shouted delivery (e.g. low-band LTAS deviation −16 to −30 dB), not a channel effect. Excluding shouting would bias the default toward calm delivery.
- Compute in fp32 and round to bf16 before training (Kurisu).

### 7.4 Augmentation (only for the experimental `phone` arm)

- **Sources:** 70–100 clean, train-split clips per character, so **about 1,000–1,400 rows**, 4–6% of rows. This matches Kurisu's 4.4% for one effect. Augment *all* 14 characters, including those with no real phone takes.
- **Chain:** the fitted show EQ with per-clip jitter (±2.5 dB tilt, 0.9–1.1 band-edge warp); 15% narrowband rows.
- **Invariants:** each source used once; clean originals stay under default rows; stratified by character × length × punctuation.
- **Real phone takes:** route to their character's phone row, repeated ×2–4 as in Kurisu.
- **Row vectors:** `v_c + δ_phone`, with δ estimated from the augmented clips of the *other* characters (leave-one-out), or simply the mean x-vector of that character's augmented clips. Pre-register one.
- **Counts are not evidence-based.** No published minimum exists [U]. Kurisu's single-character 400 is the only data point.

### 7.5 Gates to pre-register (before any training)

| Gate | Definition | Threshold | Notes |
|---|---|---|---|
| **G0 reference hygiene** | Projection of (v_c − v_c from the strictest clean subset) onto each δ_e | ≤ 0.05 of the full shift, every character, every δ | No generation needed |
| **G1 default leak** | Per character, ≥ 15 texts × 2 seeds. Metrics: LTAS projection onto the phone/radio δ; x-vector projection onto every δ; CED reverb detection (threshold at 5% FPR on the reference); echo cepstral detection. **Reference:** for the v1 recipe (all effect clips dropped, DSP only) the main run *is* the no-effect-data run, so it is scored against **Base + the same locked vector, untrained**, on the same texts and seeds (the §3.4 setup). For the phone arm, the reference is the v1 main run | Each projection ≤ 0.10; each detection ≤ reference + 5 pp | **Validity first** (Amendment C lesson): two training seeds of the v1 recipe must pass against each other, or the gate is reported unusable |
| **G2 fidelity** (each learned effect row) | Same metrics against the **DSP-on-default ceiling** | ≥ 0.5 of the ceiling; CER ≤ default + 0.02; zero runaways | Phone projection is in-sample if few real takes; report as such |
| **G3 learned vs DSP** | Blind pairwise preference: row output vs DSP(default output), same text and seed | Learned ships only if preferred with the 95% CI lower bound > 50%, ≥ 8 raters × 25 pairs (`research_C_eval.md` design-effect note) | Otherwise DSP only. Either outcome is acceptable |
| **G4 default quality** | CER and SIM of default rows vs the control | Per the A2-style paired bootstrap | — |
| **G5 DSP preset check** | Fitted DSP on held-out clean clips vs real effect takes | Phone EQ shape error ≤ 1.5 dB over 150 Hz–8 kHz excluding F0 bands (Kurisu); echo delay within ±5 ms; reverb RT60 within ±20% | Documents that the presets match the games |

### 7.6 Release

- The default voice is clean by construction.
- The "phone", "inner voice" and "hall" options are DSP presets in the inference wrapper. They are composable with any character, any emotion instruct and future rows, and documented as post-processing.
- If the phone arm passes G1–G3, expose `<c>_phone` rows as experimental.

---

## 8. Could not verify / nothing published

- **Perceptual DSP-vs-learned comparison** for telephone, reverb or echo in TTS: nothing published. UmbraTTS's 87.89% concerns background mixing only. Kurisu's planned A/B was never run.
- **Effect leakage into default/clean outputs**, measured with a leakage metric in a 2025–26 environment-control paper: none found. DAIEN-TTS, VoiceDiT, UmbraTTS and AST-LDM report WER/SIM/CLAP/FAD only.
- **Minimum augmented clips per effect** for a codec-LM fine-tune: none found.
- **CosyVoice 3 / IndexTTS-2 / Step-Audio-EditX / ElevenLabs v3 channel tags:** none found by search [S]. Fish S2's official list has none. `[echo]` appears only on an aggregator [U].
- **MiniMax `sound_effects` being DSP post-processing:** not stated in the docs [U].
- **Raon-OpenTTS filtering result:** search summary only [S].
- **SwanTale's SRMR "Reverb Consistency" value:** table header not extractable.
- **Qwen speaker-encoder training augmentation:** the tech report says only that it is "jointly train[ed] … with the backbone". A reconstruction-trained encoder would be expected to keep channel information [I], consistent with §3.2.
- **Probe limitations [L]:**
  - Generic DSP, not the games' actual effects. The sibling's detected real takes should replace them for δ estimation.
  - 3 characters × 10 texts × 1 seed. No CER was measured for the probe outputs; the runaway check passed.
  - Untrained Base only. Kurisu's fine-tune kept the row effect at about the E0 level (1.04–1.22 → 1.03–1.31), so a fine-tune should not be assumed to remove contamination.
  - **Reproducibility.** The script ran against a hand-built package directory in the session scratchpad (`qwen-tts==0.1.1`, `sox`, `accelerate`, `transformers==4.57.3` and `huggingface_hub==0.36.0`, all installed `--no-deps --target`), because the system Python has transformers 5.14, which `qwen-tts` 0.1.1 cannot import. That directory will not survive the session. The script's docstring gives the pip line and the `FPROBE_PKG` / `FPROBE_OUT` path overrides.

## Sources

Primary, read (dates are v1 unless noted):
- Hsu et al., ICASSP 2019: https://research.google/pubs/pub47834/ · PDF https://sls.csail.mit.edu/archives/root/publications/2019/Wei-NingHsu_ICASSP-2019.pdf
- Environment-aware TTS (2021-10): https://arxiv.org/abs/2110.03887
- VALL-E (2023-01): https://arxiv.org/abs/2301.02111
- Raj et al., x-vector probing, ASRU 2019: https://arxiv.org/abs/1909.06351
- Zhao et al., probing deep speaker embeddings (2022-12): https://arxiv.org/abs/2212.07068
- Wang et al., FM zero-shot TTS noise robustness (2024-06): https://arxiv.org/abs/2406.05699
- AST-LDM / Speak in the Scene (2024-06): https://arxiv.org/abs/2406.12688
- VoiceDiT, ICASSP 2025 (2024-12): https://arxiv.org/abs/2412.19259
- Audiobox Aesthetics (2025-02): https://arxiv.org/abs/2502.05139
- UmbraTTS (2025-06): https://arxiv.org/abs/2506.09874
- DAIEN-TTS (2025-09): https://arxiv.org/abs/2509.14684 · extended (2026-08-04): https://arxiv.org/abs/2608.03011
- Qwen3-TTS technical report (2026-01): https://arxiv.org/abs/2601.15621
- Grundhuber & Habets, Interspeech 2026 (2026-06-30): https://arxiv.org/abs/2606.31365
- Poly-InstructTTS (2026-08 by ID): https://arxiv.org/abs/2608.20387
- SwanTale (v2 2026-08-04): https://arxiv.org/abs/2608.02023
- COT-TTS (v2 2026-09-25): https://arxiv.org/abs/2609.22697
- Noise-robust zero-shot TTS with SSL adapters, ICASSP 2024: https://arxiv.org/abs/2401.05111
- Brouhaha (2022-10): https://arxiv.org/abs/2210.13248
- DataSpeech bin edges: https://github.com/huggingface/dataspeech (`examples/tags_to_annotations/v02_bin_edges.json`, `v02_text_bins.json`)
- Emilia-Pipe `min_dnsmos=3`: https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/utils/tool.py
- NISQA: https://github.com/gabrielmittag/NISQA
- Irodori-TTS emoji palette (2026-05-12): https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/gradio_emoji_palette.py
- Higgs Audio v2 README: https://github.com/boson-ai/higgs-audio/blob/main/README_V2.md
- MiniMax T2A API: https://platform.minimax.io/docs/api-reference/speech-t2a-http
- Fish Audio emotion reference: https://docs.fish.audio/resources/emotion-reference
- Qwen3-TTS issue #386 (12 Hz 40 ms echo, 2026-10-22): https://github.com/QwenLM/Qwen3-TTS/issues/386

Secondary:
- Cong et al., Interspeech 2020: https://arxiv.org/abs/2008.04265
- Raon-OpenTTS (2026-05): https://arxiv.org/abs/2605.20830
- EnvTTS community model card: https://huggingface.co/ChristianYang/LongCat-AudioDiT-Env-TTS-1B-ablation
- Inner-monologue post effects: https://forum.audacityteam.org/t/distinguishing-spoken-dialogue-from-inner-monologue-and-telepathy-without-dialogue-tags/64984

Local:
- Kurisu files listed in §1.
- `F:\Tricolour\research\notes\effects_probe.py` and `effects_probe.json`. Generated audio is in the session scratchpad `fprobe\`, not kept.
