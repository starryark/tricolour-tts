# Research C: Automatic and listening evaluation for a Mandarin multi-character galgame TTS

Researched 2026-10-29. Tags: [P] primary source I read myself (arXiv HTML, GitHub code or README, HF card); [S] secondary (search snippet or summary site); [I] my inference or recommendation; [U] unverified.

**Bottom line**
- No published evidence covers ASR or speaker verification on *Mandarin* anime or galgame voices. Every anime result found is Japanese (or Japanese and English dubs).
- Decide using measurements on your own held-out **real** clips. Treat each real clip's score as the floor or ceiling that synthetic output is judged against [I].
- Conventional MOS predictors (UTMOS, DNSMOS, Audiobox) are at chance on expressive speech in the one benchmark that tests this (SpeechJudge) [P].

---

## 1. Mandarin ASR for scoring TTS output

### What the benchmarks use
- **Seed-TTS-eval** [P, `run_wer.py`, GitHub, read 2026-10-29]:
  - The zh ASR is FunASR `AutoModel(model="paraformer-zh")`; en uses Whisper-large-v3.
  - zh pipeline:
    1. The hypothesis goes through `zhconv.convert(..., 'zh-cn')`. The reference is not converted.
    2. All `zhon.hanzi.punctuation` and `string.punctuation` characters are stripped from both strings, except `'`.
    3. Each string is split into single characters.
    4. `jiwer.compute_measures` scores the result, so the reported "WER" is really CER.
  - No pinyin handling: a homophone counts as a substitution. https://github.com/BytedanceSpeech/seed-tts-eval
- **CV3-Eval** (CosyVoice 3, arXiv 2505.17589, May 2025) [P, README]: "Whisper-large V3 for English ASR and Paraformer for Chinese ASR". SIM uses ERes2Net (3D-Speaker), and audio quality uses DNSMOS. https://github.com/FunAudioLLM/CV3-Eval
- **Qwen3-TTS report** (arXiv 2601.15621, Jan 2026) [P]: names Qwen3-ASR only for long-form transcription (§4.2.6). The zero-shot WER sections do not name an ASR. SIM uses "a WavLM-based speaker verification model". UTMOS is reported only for the tokenizer.
- **Conclusion** [P/I]: Paraformer-zh is still the de facto TTS-paper standard in 2026, inherited through Seed-TTS-eval and CV3-Eval. I found no evidence that the community has moved to Qwen3-ASR or FireRedASR2 for TTS scoring.

### Accuracy on ordinary Mandarin (vendor-reported, not on TTS or expressive speech)
- **Qwen3-ASR-1.7B** [P, arXiv 2601.21337, v2 30 Jan 2026]:

  | Model | AISHELL-2 | WenetSpeech net / meeting | SpeechIO | CV-zh | Tongue-twister (internal) |
  |---|---|---|---|---|---|
  | Qwen3-ASR-1.7B | 2.71 | 4.97 / 5.88 | 2.88 | 5.35 | 2.44 |
  | Whisper-large-v3 | 5.06 | 9.86 / 19.11 | 7.56 | 12.91 | 16.63 |

  The report has no emotional or expressive test set.
- **FireRedASR2** (arXiv 2603.10420, 11 Mar 2026) [P]: average CER over four Mandarin test sets:

  | Model | Avg CER |
  |---|---|
  | FireRedASR2-LLM (8B+) | 2.89 |
  | FireRedASR2-AED (1B+) | 3.05 |
  | Doubao-ASR | 3.69 |
  | Qwen3-ASR | 3.76 |
  | Fun-ASR | 4.16 |
  | Fun-ASR-Nano | 4.55 |

  On AISHELL-1, AED scores 0.57 against 1.48 for Qwen3-ASR. On singing (Opencpop), AED scores 1.17 against 2.57 for Qwen3-ASR. The report has no emotional or expressive evaluation, and the comparison is self-reported.
- **Paraformer-Large** (0.2B): 4.56 average in the FireRedASR v1 table (AISHELL-1 1.68) [S]. The FunAudioLLM paper reports AISHELL-1 CER of 1.95 for Paraformer-zh, 2.09 for SenseVoice-Large, 2.96 for SenseVoice-Small and 5.14 for Whisper-large-v3 [S].
- **Dolphin** (DataoceanAI/Tsinghua, arXiv 2503.20212): 40 Eastern languages and 22 Chinese dialects, built for Eastern languages. The only numbers found are multilingual WER averages; I found no Mandarin CER [S].

### Which ASR is best on expressive or anime speech?
- **No evidence exists** for Mandarin anime, galgame or acted speech [I, after searching].
- Indirect hints:
  - Qwen3-ASR is far ahead of Whisper on tongue-twisters, kids' speech and singing [P].
  - FireRedASR2 beats Qwen3-ASR on singing [P].
  - SpeechJudge's annotators found expressive clips harder [P].
- Whisper-large-v3 is clearly the weakest of these on Mandarin. That fits your observed median real-clip CER of 0.08 [I].

### Pinyin-space CER
- No standard TTS benchmark uses it. Seed-TTS-eval and CV3-Eval score characters only [P].
- **Only TTS precedent found**: arXiv 2601.22661 (StepFun et al., 30 Jan 2026, role-play TTS) reports "Pinyin WER" next to CER, with a Step-Audio ASR [P]. It does **not** define the metric (no tone handling, no tool named) [P].
- Older precedent: a Blizzard 2009 Mandarin entry reported pinyin error rate without tone (12.2%), with tone (17%) and CER (23%) side by side [S]. Pinyin error rate is also used in Mandarin ASR and lip-reading papers [S].

### A trap for a pre-registered selection rule
- arXiv 2607.08256 (KAIST, v2 5 Aug 2026; English, F5-TTS) [P]:
  - Best-of-N verifier rankings flip depending on which ASR family does the evaluation.
  - A verifier and evaluator from the same family look better, a bias similar to an LLM judge favouring its own output.
  - Recommendation: report WER under "at least two ASR families with disjoint training lineages", and use cross-family rank ensembles for selection.
- Applied to this project [I]:
  - Paraformer, SenseVoice and Fun-ASR share Alibaba FunAudioLLM lineage. Qwen3-ASR (Alibaba Qwen) and FireRedASR2 (Xiaohongshu) are separate lineages. Whisper is a third.
  - If you select checkpoints by Paraformer CER, do not use Paraformer CER alone as the confirmation metric.

### Recommendations for Q1 [I]
1. Score real held-out tier-A clips with Paraformer-zh, Qwen3-ASR-1.7B and FireRedASR2-AED. Pick the primary ASR by lowest real-clip error **on your data**. Real-clip error is also the floor that TTS output is judged against.
2. Compute three numbers on Seed-TTS-eval-normalised text:
   - raw CER, for comparability with papers;
   - toneless pinyin error rate;
   - tone-aware pinyin error rate.
   For pinyin, use `pypinyin` `lazy_pinyin` with `Style.NORMAL` for toneless and `Style.TONE3` for tone-aware, plus `neutral_tone_with_five`. Score per syllable with jiwer.
3. pypinyin's own polyphone errors (多音字) on the *reference* side cancel out only if hypothesis and reference both go through the same G2P. Run both through pypinyin. Do not use a hand-labelled reference pinyin.
4. Use two ASR families: one inside the selection rule and a different one as the held-out confirmation, or a rank ensemble of both, following 2607.08256.
5. Toneless pinyin error rate is the main homophone-robust metric. Tone-aware pinyin error rate catches tone and sandhi errors, but ASR tone decisions are themselves noisy on acted speech. Report it, but weight it less.

---

## 2. Speaker-similarity models

### What the benchmarks use
- **Seed-TTS-eval SIM**: "WavLM-large fine-tuned on the speaker verification task" (checkpoint `wavlm_large_finetune.pth` from Google Drive) [P]. It is commonly described as the UniSpeech WavLM-large with an ECAPA-TDNN head [U].
- **CV3-Eval SIM**: ERes2Net through 3D-Speaker [P].
- **3D-Speaker official table** [P]. The page does not state the training data for these rows. CAM++'s 6.78% CN-Celeb figure matches the about-3k-speaker number on the mirrored card, which suggests these are not the 200k models [I].

  | Model | Params | VoxCeleb1-O EER | CN-Celeb EER | 3D-Speaker EER |
  |---|---|---|---|---|
  | CAM++ | 7.2M | 0.65% | 6.78% | 7.75% |
  | ERes2NetV2 | 17.8M | 0.61% | 6.14% | 6.52% |
  | ERes2Net-large | 22.46M | 0.52% | 6.17% | 6.34% |

  Chinese 200k-speaker checkpoints:
  - `iic/speech_campplus_sv_zh-cn_16k-common`
  - `iic/speech_eres2netv2_sv_zh-cn_16k-common`
  - `iic/speech_eres2net_sv_zh-cn_16k-common`

  A mirrored model card reports CAM++ CN-Celeb EER of 6.78% when trained on about 3k speakers and **4.32%** when trained on about 200k speakers [S].
  https://github.com/modelscope/3D-Speaker
- **Qwen3-TTS speaker encoder** [P]: "we jointly train a learnable speaker encoder with the backbone". The paper does not describe its architecture. The repo has **no standalone embedding-extraction API**; only `create_voice_clone_prompt(..., x_vector_only_mode=True)` uses it internally. An encoder trained jointly with the generator is also a bad judge of that generator's output [I]. Not recommended.

### Anime and character-voice evidence (all Japanese)
- **KyaraBench / KyaraEmbed** (Spellbrush, arXiv 2610.06013, 5 Oct 2026) [P]:
  - Data: 85 human-audited identities from Anim-400K JA/EN anime dubs.
  - EER results:

    | System | Mono | XLing | Mono-VA |
    |---|---|---|---|
    | VoxCeleb ECAPA | .138 | .367 | .305 |
    | WavLM-base+ SV | .201 | .314 | .288 |
    | ReDimNet | .156 | .383 | .295 |
    | Qwen3-Omni audio tower | .387 | .455 | .417 |
    | Domain-fine-tuned ECAPA | .149 | .271 | .252 |
    | KyaraEmbed | .155 | .215 | .207 |

    Mono = same character, same language. XLing = the character's dubbed performer in the other language. Mono-VA = impostor characters who share a voice actor.
  - Listeners: Mono BER .226 (worse than ECAPA's .143). Humans also accept same-actor impostors more often than different-actor ones.
  - CAM++ and ERes2Net were not tested.
  - XLing changes the **performer** as well as the language, so the Mono-to-XLing jump does not isolate language bias. The supportable point is Mono: VoxCeleb ECAPA reaches 13.8% EER on same-actor, same-language anime speech, against under 1% on VoxCeleb1-O. That is a large domain gap for acted anime speech [P/I].
  - Model: https://huggingface.co/spellbrush/kyaraembed
- **litagin anime_speaker_embedding** [P, HF card]:
  - SpeechBrain ECAPA with GroupNorm, trained on the OOPPEENN Japanese VN dataset.
  - `char` variant: 7,357 characters, separates characters even when they share a voice actor. `va` variant: 989 actors, validation EER 0.41%.
  - MIT licence; `pip install anime_speaker_embedding`.
  - The card notes that same-speaker cosines are deliberately low.
  - It is trained on **Japanese**, so it may carry the same language bias you are trying to escape [I].
  - https://huggingface.co/litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm
- **Voice-actor identification floor** (arXiv 2607.15694, Jul/Aug 2026) [P]:
  - Data: 1,168 Japanese voice actors.
  - A closed-set nearest-centroid misidentification floor of about 2.6% remains after calibration, score normalisation and re-ranking (13% session-disjoint). The author attributes it to embedding geometry: crowded embedding space and multi-style actors.
  - A domain-matched encoder greatly reduces misattribution. A single global threshold cannot balance false accusations against misses, so per-speaker calibration is recommended.
- **Chinese anime or galgame SV EER**: **none found**.

### Why your Mandarin audit (median same-speaker ≈ 0.5) may be low
This is the opposite failure from the Japanese saturation above 0.9.
- Carbonneau et al. (arXiv 2507.02176, Jul 2025) [P]:
  - Speaker-verification embeddings encode **clip duration**, **SNR** and **equalisation**, and can separate identical speech by length alone.
  - They capture timbre and pitch range, but miss speech rate and pitch or loudness variability.
  - Mitigations: match durations, measure SNR on both sides, re-equalise.
- Rule these confounds out before comparing models [I]:
  - very short or very long clips;
  - sample-rate or resampling chains;
  - BGM or reverb in game rips;
  - shouts and whispers inside tier A.
- Also consider the anime domain gap (KyaraBench Mono: 13.8% EER for VoxCeleb ECAPA), Mandarin-versus-VoxCeleb language mismatch (not directly measured anywhere) and a VoxCeleb model's limited range of young female voices [I].

### Recommendations for Q2 [I]
1. Run a **bake-off on real held-out clips** before choosing a SIM model:
   - candidates: Seed-TTS-eval WavLM-large-SV, CAM++-200k-zh, ERes2NetV2-200k-zh, litagin `char`, and optionally KyaraEmbed;
   - for each model, compute EER, nearest-centroid identification accuracy and confusion pairs on tier-A real clips, with raw and globally centred cosines.
   - Pick the model that best separates the **real** cast. No published result answers this for Mandarin galgame voices.
2. Prior: CAM++ or ERes2NetV2-200k-zh are the best bets for the Mandarin language match. litagin `char` is the best bet for the anime domain and character-level separation. An ensemble that averages z-normalised scores is reasonable.
3. Report Seed-TTS-eval WavLM SIM as well, but only for comparability with papers.
4. Always centre and normalise:
   - subtract the global mean embedding of the real cast and re-L2-normalise, or
   - use adaptive s-norm against a cohort of the other characters.
   This was the fix in your Japanese project, and it is standard practice in speaker verification [I].

---

## 3. Speaker-confusion (identity confusion matrix) metric

### Precedent
- No TTS paper found uses exactly "classify each output among the cast centroids, then report the own-speaker rate and the confusion matrix". Closest:
  - **Deep Voice 2** (arXiv 1705.08947, 2017): speaker-classifier accuracy on synthesised VCTK samples, 99.9% against 98.8% for ground truth. Proposed as an automatic measure of how distinguishable the voices are [S].
  - **Unsupervised Polyglot TTS** (arXiv 1902.02263): top-1 and top-5 speaker-ID accuracy per language pair [S].
  - **NIST speaker de-identification** (arXiv 2508.14012): CMC@k hit rates [S].
  - **Voice-actor floor paper** (2607.15694): one L2-normalised 8-segment centroid per speaker, nearest-centroid closed-set top-1 error, open-set detection at a fixed false-alarm rate [P/S].
  - 3D-Speaker and other speaker-verification toolkits use EER/minDCF. Speaker identification uses top-1 accuracy or CMC [S].
- The metric is established in spirit (speaker-ID accuracy on synthetic speech) but has no standard name or protocol for TTS [I].

### Threshold setting [I]
1. **Relative to real data.** Build centroids from real **train** clips. Classify real **held-out** clips the same way. Their per-character own-centroid rate R_c, and the real confusion matrix, are the ceiling. The gate is generated rate G_c ≥ R_c − δ, for example δ = 10 points, rather than a fixed absolute rate. Characters the real data cannot separate get no penalty.
2. Use the same embedding model, centring and segment length for real and generated clips. Match generated durations to real held-out durations (Carbonneau).
3. Report:
   - the macro-averaged own-speaker rate, so large characters don't dominate;
   - the worst-character rate;
   - the off-diagonal pairs whose generated confusion exceeds the real confusion. These are *new* confusions introduced by training.
4. Add a **margin** statistic per clip: cos(own) − max cos(other). It is continuous and so less noisy than a hit/miss rate. Gate on the median margin, as well as top-1.
5. Put bootstrap CIs over clips on all of these. With about 20 probes per character, a single character's rate has a 95% CI of about ±20 points. Gate on the cast macro average and use per-character rates only to diagnose.
6. Pre-register δ, the embedding model and the centring before seeing checkpoint results.

---

## 4. Naturalness / MOS predictors on expressive Chinese

### Strongest evidence: SpeechJudge
- arXiv 2511.07931, ICLR 2026, Amphion/CUHK-SZ [P].
- SpeechJudge-Eval has 1,000 human naturalness-preference pairs: 400 zh, 400 en, 200 mixed.
- The expressive subset uses emotional, accented and whispered references, plus **video-game character voices (Genshin Impact)**.
- Agreement with human preference (chance = 50%):

  | Judge | Regular | Expressive |
  |---|---|---|
  | DNSMOS | 61.0 | 55.8 |
  | UTMOS | 54.0 | 53.5 |
  | Audiobox-Aesthetics CE | 69.3 | 55.2 |
  | Audiobox-Aesthetics PQ | 61.3 | 54.3 |
  | WER | 59.3 | 57.0 |
  | SIM | 47.5 | 42.5 |
  | Qwen2.5-Omni-7B | 62.0 | 59.7 |
  | Kimi-Audio-7B | 65.5 | **68.0** |
  | Gemini-2.5-Flash | 73.5 | 66.2 |
  | GPT-4o-audio | 71.5 | 64.7 |

  Trained judges, total accuracy: SpeechJudge-BTRM 72.7, **SpeechJudge-GRM 77.2** (79.4 with voting@10).
- Gemini-2.5-Flash scores 81.6% on zh-to-zh regular pairs, against 59.4% on en-to-en [P].
- GRM release: `RMSnow/SpeechJudge-GRM`, Qwen2.5-Omni-7B base, **pairwise** (`compare_wavs(text, wavA, wavB)`). Licence not stated. https://github.com/AmphionTeam/SpeechJudge [P]
- NISQA was not tested.

### Other evidence
- **Audio Turing Test** (Meituan/SJTU, arXiv 2505.11200, May 2025) [P]:
  - A Chinese human-likeness test with Human / Unclear / Machine labels and 857 listeners. Its dimensions include paralinguistics and emotion, but no role-play.
  - Auto-ATT is Qwen2-Audio LoRA. Trap-item F1: Auto-ATT 0.92, UTMOSv2 0.14, DNSMOS-Pro 0.00.
- **Audiobox Aesthetics** reports good results on the VMC22 out-of-domain Chinese Blizzard-2019 set, which is read speech, not expressive [S].
- UTMOS's out-of-domain track was Chinese synthetic speech (VMC22) [S].
- **Chinese-trained MOS models**: the Tencent Chinese MOS corpus exists [S]. I found no Chinese expressive-speech MOS predictor with published human correlation.

### Recommendations for Q4 [I]
- UTMOSv2, DNSMOS and Audiobox are not usable to rank expressive checkpoints, only as regression or artefact alarms. Example: flag a checkpoint if its median drops by more than X relative to real clips.
- For a learned naturalness signal:
  - run SpeechJudge-GRM pairwise between candidate checkpoints on the same text and speaker; or
  - use Gemini or Kimi-Audio as an A/B judge.
- Validate either against your own listening test before trusting it. Galgame Mandarin acting is out of domain for all of them.
- Label all of these "predicted", per Cooper et al. 2025.

---

## 5. Emotion fidelity: emotion2vec+ large

- **CV3-Eval already does exactly this** [P, `scripts/eval_emotation.py`]:
  - `pipeline(Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")` at 16 kHz, utterance granularity.
  - The prediction is the argmax over {angry, disgusted, fearful, happy, neutral, other, sad, surprised, unk}. The intended emotion is the utterance-ID prefix.
  - sklearn `classification_report` gives per-class P/R/F1, split by high and low intensity.
  - Reference audio comes partly from EmoBox (Chinese and English).
- **Model card** [P]: about 42.5k hours of fine-tuning data, filtered pseudo-labels. Languages are **not stated**, and there are no per-language numbers. Embedding extraction (`extract_embedding=True`, utterance or 50 Hz frame) is supported.
- **Mandarin evidence**:
  - emotion2vec base was best among SSL models on M3ED (Mandarin TV drama), but only at WA 49.2 / UA 27.0, with a frozen probe [P].
  - An emotion2vec-family model is reportedly strongest on Chinese and ESD test sets in zero-shot cross-lingual SER [S].
  - No published accuracy for emotion2vec+ large on Mandarin *acted* or anime speech [I].
- **Emotion-embedding cosine is discouraged**:
  - EmoVoice (arXiv 2504.12867) found emotion similarity "not sensitive enough" for fine-grained comparisons [S].
  - "The False Resonance" (arXiv 2604.26347, Interspeech 2026, NTU/USC) [P, abstract]: emotion2vec-style embeddings "classify emotion accurately" but are "unsuitable for zero-shot similarity evaluation", because speaker and linguistic information interferes. The metric rewards acoustic mimicry over emotional synthesis.
  - IndexTTS 2.5 and others still report emotion2vec cosine [S].
- **Labelling training data**: EmoBox built its cross-corpus test set by keeping only utterances where the human label **agreed** with emotion2vec's pseudo-label [S].

### Recommendations for Q5 [I]
1. Use **classification agreement**, not cosine. For each eval probe with an intended emotion, check the argmax label, or the probability of the intended class. Report macro F1, and score each emotion against the same metric on real clips of that emotion.
2. For emotion-balanced probes, auto-label the training data with emotion2vec+ large. Keep a clip only if:
   - top-1 probability ≥ about 0.6 (tune on a hand-checked sample); and
   - the label agrees with a second signal: the script or scene context, the VN's sprite or expression tags, or a second model such as SenseVoice's emotion tag.

   Hand-check about 20 clips per class first. Galgame "happy", "teasing" and "embarrassed" acting often lands in "other" or "neutral" [I/U].
3. Expect weak classes (disgusted, fearful) and pre-register which classes the gate covers. A reasonable set is neutral, happy, sad, angry and surprised.

---

## 6. Listening-test design for many speakers

### Guidance found
- **Cooper, Le Maguer, Klabbers & Yamagishi**, "Good practices for evaluation of synthesized speech" (arXiv 2503.03250 v2, Oct 2025) [P]:
  - Gives **no numeric minimums**. It requires reporting listener count, samples per system, ratings per sample, the statistical test and the exact question.
  - "True" MUSHRA needs anchors, which TTS usually lacks.
  - Suggested tests: Wilcoxon or Mann-Whitney for MOS and MUSHRA; binomial or Clopper-Pearson for pairwise tests. Prefer an omnibus test with post-hoc comparisons over Bonferroni.
  - Non-significance does not establish equivalence. Include baselines in your own test.
- **Wester, Valentini-Botinhao & Henter, Interspeech 2015** [S, abstract only; PDF unreadable]:
  - More than 60% of Interspeech 2014 TTS papers used fewer than 20 listeners.
  - Re-analysing Blizzard 2013 showed that rank order and the number of significant pairs change with listener count and sentence coverage.
  - I could not verify a specific minimum listener number from it [U].
- **Speaker-similarity precedent**: a MUSHRA-like 0–100 similarity test with 23 listeners and 24 speaker sets, analysed with Wilcoxon and Holm (arXiv 2506.20190) [S].
- KyaraBench's human study used 48 listeners for 320 trials, and listeners were worse than ECAPA on same-character verification (BER .226) [P]. Untrained listeners who don't know the characters are poor at character verification.

### Power numbers [I, my exact-binomial calculation, alpha .05, power .8]
- **4-AFC speaker ID** (target plus 3 foils, chance 25%), trials needed to show above-chance identification:

  | True accuracy | Trials |
  |---|---|
  | 40% | ≈ 62 |
  | 50% | ≈ 26 |
  | 60% | ≈ 14 |

- Comparing **two checkpoints' ID accuracy** (unpaired):

  | Accuracies | Trials per arm |
  |---|---|
  | 60% vs 75% | ≈ 150 |
  | 50% vs 70% | ≈ 90 |
  | 60% vs 70% | ≈ 350 |

- **AB preference between two checkpoints** (two-sided), total trials:

  | True preference | Trials |
  |---|---|
  | 75/25 | ≈ 30 |
  | 70/30 | ≈ 50 |
  | 65/35 | ≈ 90 |
  | 60/40 | ≈ 200 |

- Rater clustering inflates these by a design effect of 1 + (m−1)·ICC. For example, 40 trials per rater at ICC 0.05 gives ≈ 3×. Use many raters with fewer trials each, and analyse with a mixed-effects logistic model or bootstrap over raters.

### Recommended design [I]
1. **Primary checkpoint decision: blind AB or ABX preference** between the two or three finalists, from the same text, character and emotion probe.
   - Test emotion- and character-matched pairs. For 2 finalists, aim for at least 8 raters × 25 pairs ≈ 200 trials. That detects a 60/40 preference only if raters are independent. At ICC ≈ 0.05 the design effect is ≈ 2.2, giving about 90 effective trials, which is just enough for a 65/35 preference. Add raters if ICC is higher.
   - Report the binomial CI per rater-bootstrap.
2. **Speaker-ID test** ("which character is this?"):
   - Full 15–23-way forced choice is unrealistic for raters who don't know the cast.
   - Use **4-AFC**: real reference clips of the target plus 3 foils. Take the foils from the **most-confused pairs in the real-data confusion matrix**, plus one random foil.
   - Include real held-out clips as a control. The real-clip ID rate is the ceiling, as in §3.
   - Raters should be familiar with the cast, for example people who have played the VN, or should hear a short reference reel first.
3. **MUSHRA-like** (0–100, hidden real reference, no true anchor) is useful for ranking 3 to 5 checkpoints on similarity and naturalness at once. Use Wilcoxon or Friedman, then Holm.
4. Small-team minimum:
   - **at least 8–12 native Mandarin raters**, ideally 15 or more;
   - every rater hears every condition;
   - at least 2–3 clips per character for each high-priority character, and at least 1 for others, with randomised order and level-normalised clips (P.56);
   - 2–3 attention or trap items, such as a real clip in a "which is synthetic" slot.
   - These numbers are judgement calls, not published standards.
5. Pre-register the analysis, alpha and decision rule, consistent with the automatic rule.

---

## Source list (all accessed 2026-10-29)
- Seed-TTS-eval repo and `run_wer.py`: https://github.com/BytedanceSpeech/seed-tts-eval
- CV3-Eval repo and `scripts/eval_emotation.py`: https://github.com/FunAudioLLM/CV3-Eval ; CosyVoice 3: https://arxiv.org/abs/2505.17589
- Qwen3-TTS report: https://arxiv.org/abs/2601.15621 ; repo: https://github.com/QwenLM/Qwen3-TTS
- Qwen3-ASR report: https://arxiv.org/abs/2601.21337
- FireRedASR2S: https://arxiv.org/abs/2603.10420 ; https://github.com/FireRedTeam/FireRedASR2S
- FunAudioLLM / SenseVoice: https://arxiv.org/abs/2407.04051
- Dolphin: https://arxiv.org/abs/2503.20212
- ASR-family confound: https://arxiv.org/abs/2607.08256
- Role-play TTS with Pinyin WER: https://arxiv.org/abs/2601.22661
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- KyaraBench: https://arxiv.org/abs/2610.06013 ; https://huggingface.co/spellbrush/kyaraembed
- litagin anime speaker embedding: https://huggingface.co/litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm
- Voice-actor identification floor: https://arxiv.org/abs/2607.15694
- SV confounds and U3D: https://arxiv.org/abs/2507.02176
- EER vs perceptual alignment: https://arxiv.org/abs/2609.33999
- Deep Voice 2: https://arxiv.org/abs/1705.08947
- SpeechJudge: https://arxiv.org/abs/2511.07931 ; https://github.com/AmphionTeam/SpeechJudge
- Audio Turing Test: https://arxiv.org/abs/2505.11200
- emotion2vec+ large: https://huggingface.co/emotion2vec/emotion2vec_plus_large ; emotion2vec paper: https://arxiv.org/abs/2312.15185
- False Resonance: https://arxiv.org/abs/2604.26347 ; EmoVoice: https://arxiv.org/abs/2504.12867
- EmoBox: https://arxiv.org/abs/2406.07162
- Good practices: https://arxiv.org/abs/2503.03250
- Wester et al. 2015: https://www.isca-archive.org/interspeech_2015/wester15c_interspeech.html
