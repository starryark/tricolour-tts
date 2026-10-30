# Research E: detecting, quantifying and restoring production effects (2026-10-30)

**Scope.** This note covers detection, quantification and restoration of intentional production effects in the Tricolour game-voice corpus:
- echo and reverb;
- telephone, radio and PA ("digital") processing;
- background noise.

How a kept clip is then conditioned (tokens, speaker rows, augmentation) belongs to the sibling note on training-side handling.

**Tags.**

| Tag | Meaning |
|---|---|
| [P] | I read the primary source myself. |
| [P, agent] | A research sub-agent read the primary source, and I did not re-check it. |
| [S] | Secondary source or search snippet. |
| [I] | My inference. |
| [U] | Unverified. |
| [L] | Measured locally on this corpus. |

All web sources were accessed 2026-10-30.

**Already settled in the Kurisu project, and not repeated here.** Sources: `E:\STEINS;GATEVoice\kurisu_v2\research\labeling_sota_2026-10.md` §1.4 and §9, `qwen_internals_conditions.md`, and `v3\augment_notes.md`.
- The 12 Hz codec preserves channel and environment, so an untagged effect leaks into the default voice.
- A measured LTAS-ratio EQ curve applied by DSP is the most reliable way to get a phone sound.
- Generic G.711 is not a stand-in for a stylised "show phone".
- Echo is either tagged or excluded.
- Emilia filters at DNSMOS ≥ 3.0.
- Brouhaha, SQUIM, DNSMOS and NISQA exist and are what Data-Speech and ParaSpeechCaps use.
- Irodori-TTS auto-tags 📞 and 📢 with a fine-tuned Qwen3-Omni.

---

## 0. Bottom line

1. **This corpus does have measurable effects, and they cluster by scene** [L, unvalidated by ear]. I ran a cheap DSP pass over all 25,425 tier-A/B clips (§1). It flags:

   | Flag | Clips | Hours | Where they sit |
   |---|---:|---:|---|
   | Narrowband | 369 (1.45%) | 0.33 | 61 scenes |
   | Discrete echo | 106 (0.42%) | 0.11 | all in 10 scenes |
   | Long tail | 407 (1.6%) | — | weak detector |
   | Bass-cut | 860 (3.4%) | — | confounded with F0 |

   - The union of all four flags is 1,711 clips (6.7%, 1.27 h). That is an upper bound.
   - When a clip is flagged narrowband, the next line in the same scene is also flagged 58% of the time, against a 1.5% base rate. For echo the figures are 82% against 0.4%.
   - **So detection and labelling should be done per scene, not per clip.**
2. **The exported `wavs/` have their reverb tails cut.** Stage 1 trimmed every clip at −45 dBFS frame RMS plus 25 ms of padding. The median cut after the last kept frame is 0.51 s [L, `build/stage1_measure.py`].
   - Tail-based reverb measures (RT60, C50, decay) run on `wavs/` will under-detect.
   - Run reverb and echo detectors on the **source files** (`source` field, roots in `build/common.py`).
3. **The existing `rolloff_hz` field is a weak bandwidth proxy** [L]. Of the 531 clips under 4 kHz, only 250 are flagged narrowband by a speech-frame LTAS test, and 119 narrowband clips have `rolloff_hz` ≥ 4 kHz.
   - The cause is that the field is the 99.9% cumulative-energy point of the whole clip, noise floor included. A soft line with no fricatives ("……我……", 1.76 kHz) lands under 4 kHz with no processing at all.
4. **This game's phone sounds like a real telephone band, unlike Kurisu's anime phone** [L, unvalidated]. Flagged clips have:
   - a −40 dB knee at about 4 kHz. The flag rule itself requires `bw_hz` ≤ 6.4 kHz, so this is partly selected for, not a finding;
   - median 80–300 Hz energy 33.5 dB below the 0.3–3.4 kHz band, against −4.5 dB corpus-wide. The rule does not use this feature, so this is the independent evidence.
5. **No off-the-shelf detector is validated for this job.**
   - Open audio-LLMs identify the degradation type at near chance. On MRMAD (Aug 2026), with 4 options (25% chance), Qwen3-Omni scores 35–37% and Audio Flamingo 3 26% [P].
   - Brouhaha C50 was trained on read LibriSpeech with room RIRs only, and on 6 s windows [P].
   - Nothing learned is trained on discrete delay or echo [P, agent].
   - The best tools are classical DSP (cepstrum, LTAS band ratios), plus Brouhaha C50 and SIGMOS for diffuse reverb. All of them need calibrating on a small human-labelled sample.
6. **Restore little or nothing.**
   - The evidence that restoration helps TTS is all from noisy found data: LibriTTS-R, Sidon/F5-TTS, AutoPrep [P].
   - Generative restorers measurably move speaker identity. In Google's own table, Miipher-1 scores SPK 0.585 against 0.945 for a discriminative model [P].
   - Nobody has tested restoring *intentional* effects, or expressive acted speech [P, agent: "no evidence found"].
   - At an effected share of about 2–7%, dropping or labelling costs about 1 h out of 23 h.
7. **Effects can be estimated separately for re-application by DSP:**
   - echo: delay and gain from the cepstral peak;
   - phone: an EQ curve from the pooled LTAS ratio (the Kurisu method);
   - reverb: an RIR estimate from BUDDy, VINP or Rec-RIR. These are 16 kHz research code trained on English simulated rooms, and pooling per scene beats per-clip estimates [I].

---

## 1. Local pilot on this corpus [L]

**Files** (all in `F:\Tricolour\research\notes\`):

| File | Contents |
|---|---|
| `fx_pilot.py` | The script |
| `fx_pilot_feats.csv` | Per-clip features and flags |
| `fx_label_sample.csv` | Blind 100-clip labelling sheet |
| `fx_label_key.csv` | Strata key for the sheet; keep it closed while labelling |

**Run.** CPU only, 10 workers, about 2 min for 25,425 clips. It reads the untrimmed source files at 24 kHz.

### 1.1 Features

Every threshold below is a starting point, not a validated value.

| Feature | Definition | Target effect |
|---|---|---|
| `low_db`, `hi_db`, `vhi_db` | LTAS over active frames (within 30 dB of the p95 frame energy). Energy in 80–300 Hz, 3.4–8 kHz and 8–11.5 kHz, each relative to 0.3–3.4 kHz | Phone or radio band-limiting |
| `bw_hz` | Highest 1/3-octave band within 40 dB of the LTAS maximum. The grid tops out at 12.7 kHz | Band-limiting and codec |
| `cep_z`, `cep_lag_ms` | Real cepstrum of the clip plus 0.5 s after it. Robust z-score of the largest peak at 40–600 ms quefrency | Discrete echo or delay |
| `tail_ms`, `fill_db`, `gap_db` | In the source file, time from the trim point down to the file floor + 6 dB; pause level above the floor; p95 − p10 frame range | Diffuse reverb and noise |

### 1.2 Synthetic sanity check

**Method.** I took 300 tier-A clips of 2–6 s and applied four synthetic effects to each:
- echo: delay 80–400 ms, gain 0.3–0.6, feedback 0–0.4;
- exponential-noise reverb: RT60 0.5–1.5 s;
- narrowband: 300–3400 Hz, 8 kHz, μ-law;
- Kurisu-like bass-cut EQ.

**What this shows.** That each feature *can* see its effect. It says nothing about accuracy on the game's real effects.

**The "dry" baseline is not verified.** It is 300 random tier-A clips, and some may already carry real effects. That would explain why the dry upper percentiles are wide (dry `tail_ms` p90 = 600 ms). It does not change the AUCs much.

| Effect vs dry (AUC; 0.5 is chance, below 0.5 means lower under the effect) | Best features |
|---|---|
| Echo | `cep_z` **1.00** (dry p90 = 14, echo p10 = 114); `tail_ms` 0.91 |
| Reverb | `tail_ms` 0.95; `gap_db` 0.13; `fill_db` 0.87; `cep_z` 0.83 |
| Narrowband phone | `vhi_db` **0.00**; `bw_hz` 0.00; `hi_db` 0.02 |
| Bass-cut "show phone" EQ | `low_db` 0.04; `hi_db` 0.78 |

### 1.3 Corpus results (unvalidated)

| Flag (starting rule) | Clips | Of which ≥ 1.5 s | Next line in the scene also flagged (vs base rate) | Scenes touched (of 606) | Notes |
|---|---:|---:|---|---|---|
| Narrowband: `vhi_db` < −45 and `bw_hz` ≤ 6.4 k | 369 (1.45%), 0.33 h | 278 | **0.58** (vs 0.015) | 61; the top 10 hold 60% | Knee at 4 kHz in 214 clips. Median `low_db` −33.5. Chapter 5m/5w (骆衍, 迟耀), HM `dlc_maid_cafe`, SLS `Act4:1320` |
| Echo: `cep_z` > 30 and lag ≥ 55 ms | 106 (0.42%), 0.11 h | 84 | **0.82** (vs 0.004) | **10; all hits** | SLS `Act5:5280` (徐阿姨): 8 lines with a constant **66 ms** echo, z = 75–134. TRI `chapter4:04e_02` (路人): 200 ms echo, band-limited, so PA-like. Lags of 40–50 ms are pitch artefacts, hence the 55 ms floor |
| Long tail: `tail_ms` > 800 and `fill_db` > 30 | 407 (1.6%) | 168 | 0.14 (vs 0.016) | 72 | Weak and noisy. Needs Brouhaha C50 or SIGMOS REVERB (§2) |
| Bass-cut, per speaker: robust z of `low_db` < −3, full band | 860 (3.4%) | 557 | 0.24 (vs 0.034) | 162 | **Confounded with F0.** High-pitched voices (迟菓) and shouts have little energy below 300 Hz. Must be normalised per speaker, then validated |
| **Union** | 1,711 (6.7%), **1.27 h** | | | | Upper bound. The 4 flags overlap very little |

**Caveats.**
- No one has listened to any flagged clip yet.
- The pilot ran on all tier-A/B clips (25,425, 23.5 h, including bit-part banks), not only the 14-character, 21,665-clip training subset.
- The cepstral echo test needs about 0.8 s of audio and is undefined for shorter clips.
- About 25% of clips are under 1.6 s. For these, scene-level pooling is the only reliable signal [I].

---

## 2. Q1: detectors, state of the art as of late 2026

### 2.1 Room acoustics (reverb)

| Tool | Id, licence, compute | What it measures | Published accuracy | Failure modes for us |
|---|---|---|---|---|
| **Brouhaha** | `pyannote/brouhaha` (gated, OpenRAIL) [P, agent]. Small SincNet + BiLSTM, so CPU is fine. Runs on Windows through pyannote.audio [U] | Frame-level (16 ms) VAD, SNR, and C50 in −10…60 dB [P] | arXiv 2210.13248 [P]: C50 MAE **3.5 dB** on synthetic test data (multi-task), SNR MAE 4.1 dB. On real rooms (BUT ReverbDB, 1,000 utterances), C50 **R² 0.85, MAE 1.1 dB** | Trained on LibriSpeech read speech with 385 real *room* RIRs and a 6 s window [P]. Clips of 1–3 s, shouts, whispers and DAW plate or delay effects are out of distribution [I]. A discrete echo at ≥ 50 ms should lower C50, but the output is uncalibrated for that [I]. **Use it on source files, never on trimmed wavs** |
| **BERP** | github.com/Alizeded/BERP (arXiv 2405.04476, TASLP 2025). Whether weights and a licence exist is [U] | T60, EDT, C50, C80, D50, STI and more; no DRR [P, agent] | C50 MAE 0.22 dB on real data [P, agent]. This is implausibly tight next to Brouhaha, and the test rooms are probably in-distribution [I] | LibriSpeech only |
| **VINP** | github.com/Audio-WestlakeU/VINP (arXiv 2502.07205 v3, Sep 2025). Licence not stated [P] | Dereverberated speech plus an RIR waveform, from which RT60 and DRR follow [P] | SimACE Table III [P]: RT60 MAE **0.089–0.103 s**, DRR MAE 2.4–3.3 dB. FiNS 0.113 s / 2.15 dB; BUDDy 0.124 s / 3.93 dB | 16 kHz. Trained on 200 h of English with gpuRIR-simulated rooms, RT60 0.2–1.5 s. "Relatively simple noise conditions" [P] |
| **Rec-RIR** | github.com/Audio-WestlakeU/Rec-RIR. MIT; a `ckpt/epoch35.tar` is referenced (arXiv 2509.15628, v3 Jun 2026) [P] | Blind RIR via a CTF, plus RT60, DRR and C50 | The abstract claims state of the art; the numbers are only in a figure [P] | Same lab and data style as VINP [I] |
| **SRMR** | torchmetrics `SpeechReverberationModulationEnergyRatio`; needs the gammatone package [S]. CPU | Ratio of low to high modulation-band energy, non-intrusive | Not checked | Classic diffuse-reverb measure. Expressive speech rate and short clips shift it [I] |
| **Cepstral echo test** (this note §1) | Plain numpy | Delay and strength of a discrete echo | Synthetic AUC 1.00 [L] | Pitch rahmonics below ~50 ms. Needs ≥ 0.8 s |

### 2.2 Quality predictors

| Tool | Id, licence | Outputs | Evidence on reverb, band-limiting or noise | For us |
|---|---|---|---|---|
| **SIGMOS (ITU-T P.804)** | microsoft/SIG-Challenge `ICASSP2024/sigmos`, ONNX, 48 kHz native, CPU. Licence [U] [P, agent: `sigmos.py`] | **MOS_COL, MOS_DISC, MOS_LOUD, MOS_NOISE, MOS_REVERB, MOS_SIG, MOS_OVRL** | No per-dimension accuracy found [U] | The only released predictor with an explicit **reverb** head and a **coloration** head, which is the dimension most likely to catch phone EQ [I]. Rank by it; do not use it as an absolute value |
| **DNSMOS P.835** | Microsoft DNS-Challenge ONNX, 16 kHz, CPU | SIG / BAK / OVRL | Clip-level SIG PCC ≈ 0.71 [S, agent]. In CHiME-7 UDASE, DNSMOS ranked reverberant-data systems opposite to listeners [S, agent] | Blunt. Low BAK finds noise. Already shown to be at chance for ranking expressive TTS (`research_C_eval.md`) |
| **NISQA v2** | gabrielmittag/NISQA. Code MIT, **weights CC BY-NC-SA 4.0**, and the torchmetrics wrapper uses the same weights [P, agent] | MOS, Noisiness, **Coloration**, Discontinuity, Loudness | Trained on simulated codecs and packet loss plus live Zoom, Skype and WhatsApp calls, so it is the most telephone-aware. No reverb in training [P, agent] | Coloration is a second phone signal. Non-commercial licence |
| **TorchAudio-SQUIM** | `torchaudio.pipelines.SQUIM_OBJECTIVE`, 16 kHz | STOI / PESQ / SI-SDR estimates | Accuracy on reverb or band-limiting not checked [U] | Data-Speech uses it. Redundant with SIGMOS here [I] |
| **Audiobox Aesthetics** | `facebook/audiobox-aesthetics`, 0.1B, mostly CC-BY 4.0 [P, agent] | PQ, PC, CE, CU | Nothing published on reverb or bandwidth [P, agent] | PQ will likely drop on phone clips, but untested [I] |
| **UTMOS / UTMOSv2** | — | Naturalness MOS | Not a channel detector | Skip for detection |
| **Uni-VERSA, ARECHO, QualiSpeech** | ESPnet `universa` / `universa_plus` branch (arXiv 2505.20741, 2505.24518 v2 Oct 2025), QualiSpeech (2503.20290) [P, agent] | ARECHO predicts 87 metrics, including "RIR Room Size" and NISQA coloration | Per-metric numbers only in appendices I did not read [U] | Newest multi-metric predictors. Worth a look only if SIGMOS and Brouhaha disagree badly |

### 2.3 Bandwidth and codec detection

**Three effective-bandwidth estimators share one rule:** the highest bin within 50 dB of the peak of the time-averaged power spectrum.

| Estimator | Settings and use | Source |
|---|---|---|
| URGENT `utils/estimate_audio_bandwidth.py` | n_fft 512 at 16 kHz-equivalent | [P, agent] |
| NeMo SDP `BandwidthEstimationProcessor` | n_fft 512, `frequency_threshold −50`, Blackman-Harris window | [P, agent] |
| HiFiTTS-2 | Same rule. Keeps ≥ 11 kHz for the 22 kHz set and ≥ 13 kHz for the 44.1 kHz set | [P, agent] |

**FireRedTTS** keeps clips whose 99.5% roll-off is above 7 kHz [P].

**Caveats** [I]:
- Whole-clip averaging dilutes the result with silence and noise floor.
- Vorbis has its own low-pass.
- **Measure over active frames only**, as `fx_pilot.py` does.

**Pretrained telephone, channel or codec classifiers: none found.**
- What exists is classical work without weights [S, agent].
- Blind estimation of a channel's magnitude response from the long-term average speech spectrum is the classical route (Gaubitch, Brookes & Naylor, EUSIPCO 2009; TASLP 2013) [S, PDFs unreachable].

### 2.4 Audio taggers

**AudioSet label quality** [P, agent; research.google.com/audioset]:

| Class | Estimated label accuracy | Balanced-train data |
|---|---|---|
| "Echo" | 50% | about 0.2 h |
| "Reverberation" | 67% | about 0.2 h |

- CED, BEATs, PANNs, EfficientAT and AST all inherit these labels and their 10 s clip-level weak labels.
- **Per-class AP for Telephone, Echo or Reverberation: not found** [U].
- The Kurisu project found AST unable to recognise crowd and wind even in clean sources (`augment_notes.md`). Expect the same for echo.
- **CLAP zero-shot** ("telephone speech", "reverberant speech"): no evaluation found [P, agent: searched].

**RemFX** effect classifier (Rice, Steinmetz et al., WASPAA 2023, arXiv 2308.16177) [P]:
- PANNs trained at 48 kHz.
- Per-effect accuracy: reverb **84.5%**, delay **74.3%**, distortion 78.0%, compression 80.7%, chorus 75.1%.
- This was on **singing, guitar, bass and drums, not speech**, so the numbers do not transfer.
- Code and pretrained models are linked from csteinmetz1.github.io/RemFX.

### 2.5 Audio-LLM captioners

**MRMAD** (arXiv 2608.22236 v2, 31 Aug 2026; Bose/Northeastern; github.com/Bose/MRMAD) [P]:
- Nine degradations, including reverberation, echo, low-pass, band-pass, MP3 and noise.
- Speech clips average 4.7 s, at 16 kHz.
- Degradation-type identification has 4 options (25% chance):

| Model | Accuracy |
|---|---|
| Qwen2.5-Omni | 29.1% |
| Audio Flamingo 3 | 26.3% |
| Qwen3-Omni-Instruct | 35.3% |
| Qwen3-Omni-Thinking | 36.6% |
| MiMo-Audio | 33.6% |
| GPT-Audio-1.5 | 28.1% |
| Gemini 3.1 Pro | **86.2%** |
| Non-expert humans | 80.6% (on a 168-question subset) |

The paper says models do worse on band-pass and reverberation than on low-pass or packet loss.

**Models checked:**
- `Qwen/Qwen3-Omni-30B-A3B-Captioner`: Apache-2.0, audio ≤ 30 s, about 32B parameters total. It does not fit 16 GB in BF16; 4-bit is about 17–19 GB [P card / U size].
- `mispeech/midashenglm-7b`: Apache-2.0, about 16 GB in BF16, tight [P, agent].
- Kimi-Audio and Step-Audio: not checked.

**Verdict:** a local audio-LLM is not worth the VRAM for this job. Irodori-TTS's fine-tuned Qwen3-Omni tagger (Kurisu notes) is the only effect-tagging precedent, and it published no accuracy.

---

## 3. Q2: what large TTS pipelines do (with thresholds)

**Every pipeline below treats reverb, phone and noise as defects to filter or clean.** None keeps them as a labelled style, except that MiniMax keeps steady-state noise and Kimi-Audio mixes original and enhanced audio 1:1 [P, agent].

| Pipeline (date) | Detectors and thresholds (verbatim where possible) | Source |
|---|---|---|
| **Emilia** (2407.05361; 2501.15907 v2 Oct 2025) | DNSMOS P.835 OVRL **≥ 3.0** ("the lowest 'fair' quality anchor"). **Emilia-Large/YODAS incremental data: 2.4.** Language ID ≥ 80%. Per-character duration within 1.5 × IQR. Segments 3–30 s. UVR-MDX-Net Inst_HQ_3 separation. Reverb is named as a problem in §1 of 2407.05361, but no stage handles it [P, agent] | Thresholds [P] 2501.15907; code [P, agent] |
| **AutoPrep** (2309.13905, ICASSP 2024) | USR-BSRNN SE ("only deals with background noise"); DNSMOS OVRL < 2.4 discarded; speaker cosine ≥ 0.5 to the cluster centre. DurIAN ablation: MOS 2.90 → 3.10 and SIM 0.55 → 0.60 with SE | [P, agent] |
| **FireRedTTS** (2409.03283) | DNSMOS P.835 OVRL **> 3.3**; **roll-off (99.5% energy) > 7 kHz**; ASR confidence ≥ 0.8; music separation + DeepFilterNet. 624k h → 248k h | [P] |
| **WenetSpeech4TTS** (2406.05763) | MBTFNet SE on everything; tiers by **DNSMOS P.808**: Premium > 4.0, Standard > 3.8, Basic > 3.6 | [P, agent] |
| **WenetSpeech-Yue** (2509.03959) | TTS subset: DNSMOS > 2.5 and Brouhaha **SNR > 25 dB** | [P, agent] |
| **HiFiTTS-2** (2506.04152) | Bandwidth ≥ 11 / 13 kHz (−50 dB rule); CER and speaker-similarity filters; no SNR, C50 or SE | [P, agent] |
| **NeMo SDP TTS recipe** | SQUIM and bandwidth computed **as metadata only**, no filtering | [P, agent] |
| **LibriTTS-R** (2305.18802) | Restore everything with Miipher, drop nothing. Restored for noise, reverb, codec and down-sampling | [P] |
| **CosyVoice 3** (2505.17589) | MossFormer2 noise reduction; agreement across 3 ASR systems (WER < 15%); no DNSMOS or bandwidth gate | [P, agent] |
| **Kimi-Audio** (2504.18425) | BSRNN SE for noise and reverb; **original or enhanced chosen 1:1** | [P, agent] |
| **MiniMax-Speech** (2505.07916) | "Original steady-state noise … preserved" | [P, agent] |
| **Qwen3-TTS** (2601.15621) | The S2 high-quality stage "alleviates hallucinations caused by noisy data"; no thresholds | [P, agent] |
| **MOSS-TTS** (2603.18090) | DNSMOS computed *before* denoising (enhancement inflates it), > 2.8 | [S, agent] |
| Fish-Speech, IndexTTS2, Spark-TTS | No effect handling described | [P, agent] |

**Ablations on filtered vs raw data:**
- Microsoft 2406.05699, flow-matching TTS [P, agent]. The DNSMOS pre-training threshold peaks at **2.8**: SIM-o 0.749, against 0.719 with no filter and 0.746 at 3.0. Enhancing *prompts* lowered SIM-o from 0.71 to 0.68.
- 2510.03111 (Oct 2025): "denoising variants with permissive filtering provide the best overall compromise" [P, agent, abstract only].
- Raon-OpenTTS (2605.20830, May 2026): 15% filtering beat 50% [S, agent].

**Anime and game corpora** [P, agent]:
- litagin's `Galgame_Speech_ASR_16kHz` (5,354 h, which trained anime-whisper) does **no** effect filtering.
- OOPPEENN/Galgame_Dataset and moe-speech cards are gated, so no evidence either way.
- GPT-SoVITS's UVR5 guide recommends MDX-Net then **DeEcho-Aggressive**, and warns that `onnx_dereverb` "cannot remove single-channel reverb" and that DeReverb leaves heavy high-frequency plate reverb. These are community music-vocal models without published speech evaluations.
- **No source treats intentional effects as a category.**

---

## 4. Q3: restoration models and the evidence

### 4.1 Models (late 2026)

| Model | Weights / licence | Rate | Handles | Speed (estimate for 21.7k × 3 s on 16 GB) | Verdict |
|---|---|---|---|---|---|
| **Sidon v0.1** (arXiv 2509.17052 v2, Sep 2025; `sarulab-speech/sidon-v0.1`) | **Yes, MIT** (HF card) [P] | w2v-BERT 2.0 features from 16 kHz input [P: RTF measured on "16 kHz input"], 48 kHz HiFi-GAN output [P] | Reverb RT60 0.1–2.0 s; resampling to 8–48 kHz; noise SNR −5…20 dB; MP3; clipping; packet loss [P] | RTF 0.002 on an H200 [P]; about 1–2 h wall-clock here [U] | The only open, multilingual, fast universal restorer. **Resynthesises the 8–24 kHz band from 16 kHz features**, so timbre detail is regenerated [I]. Not text-conditioned: WER 0.079 → 0.095 on test-other [P, agent] |
| Miipher-2 (2505.04457) | **No**: "code and checkpoints will not be released" [P] | — | Noise, reverb, codec | RTF 0.0078 | Unavailable |
| Miipher-1 (LibriTTS-R) | No | 24 kHz | — | — | Unavailable. Open reimplementation `Wataru-Nakata/miipher` has CC-BY-NC weights, EN/JA, and "does not reproduce" the original [P, agent] |
| **SGMSE+ EARS-Reverb** (sp-uhh/sgmse) | Yes, MIT [P, agent] | 48 kHz | **Dereverb**. Training set EARS includes emotional speech and non-verbal sounds [P, agent] | Diffusion, about 50 steps; roughly 0.5–2 h per 1k clips [U] | Best fit for an expressive dereverb *trial* |
| **ClearerVoice MossFormer2_SE_48K** | Yes, Apache-2.0 [P, agent] | 48 kHz | **Noise only**; a sibling MossFormer2_SR_48K does super-resolution | about 1–3 h total [U] | Discriminative, so lowest identity risk. Use for noise if needed |
| Resemble Enhance | Yes, MIT code [P, agent] | 44.1 kHz | Noise + bandwidth; dereverb not stated | Slow CFM [U] | Generative, hallucination risk; awkward on Windows (deepspeed) [U] |
| VoiceFixer | Yes, MIT | 44.1 kHz | Noise, reverb, bandwidth, clipping | — | Mel resynthesis, so timbre changes [U]. Beat Demucs as a TTS cleaner in Sidon's Table 4 |
| LLaSE-G1 | Yes, Apache-2.0 | X-Codec2 | — | — | **SpkSim 0.42 vs 0.77 for the noisy input** (PASE paper) [P, agent]. Skip |
| AnyEnhance | Only a reduced 45.7M baseline | — | — | — | Full model unreleased [P, agent] |
| UNIVERSE++ (open-universe), BUDDy, StoRM | Yes, Apache/MIT | **16 kHz** | Noise / dereverb | — | Rate too low for a 24 kHz codec |
| AP-BWE (bandwidth extension) | Yes, MIT | 16 or 48 kHz | Band-limited only | 292× real time on a 4090 [P, agent] | Fast, but bandwidth extension of a phone band *invents* the 4–12 kHz timbre band [I] |
| FlowSE, PASE (2511.13300, AAAI 2026), DialogueSidon (2604.09344), SE-MSB (2609.26000) | FlowSE code; PASE code (cisco-open/pase), weights not stated | — | — | — | PASE: lowest-hallucination generative SE, WER 7.49 vs 14.35 noisy [P, agent]. Newest seen: SE-MSB, title only |

URGENT 2026 (arXiv 2601.13531): the top systems are hybrid generative + discriminative [P, agent].

### 4.2 Evidence that training TTS on restored audio helps or hurts

**LibriTTS-R** (2305.18802) [P]:
- TTS naturalness MOS on Train-960 rises from **2.59–3.35** on LibriTTS to **4.06–4.31** on LibriTTS-R, per speaker (Table 2).
- **No speaker similarity was reported for TTS outputs.**
- Ground-truth corpus speaker similarity fell slightly: test-clean 0.784 → 0.762, test-other 0.755 → 0.745.
- 2 of 640 restored samples were "distorted due to the failure of SR"; the training splits "may also contain a small number".

**Identity drift caused by enhancement.** Do not compare across papers: they use different speaker encoders.

| Source | Result |
|---|---|
| Miipher-2 Table 2 (500 LibriTTS test-other) [P] | SPK: TF-GridNet (discriminative) **0.945**, Miipher-1 **0.585**, Miipher-USM 0.722, Miipher-2 0.744 |
| Sidon paper (different encoder) [P] | Miipher 0.942 / 0.930 vs Sidon 0.971 / 0.961 on LibriTTS clean / other. FLEURS average: Sidon 0.979 = Miipher-2 0.979 |
| PASE (2511.13300) [P, agent] | Several generative SE models make WER *worse* than the noisy input: StoRM 45.9, LLaSE-G1 36.6, FlowSE 30.1, vs 14.35 noisy. LLaSE-G1 SpkSim 0.42 |

**TTS trained on enhanced data:**
- **Sidon, Table 4** [P]: F5-TTS trained on TED-LIUM 3. Sound-quality MOS: original 3.254, Demucs 3.265, VoiceFixer 3.771, Sidon **4.248**. No SIM or WER reported.
- **Yamauchi et al., 2601.12254 (Jan 2026)** [P]: Genhancer SE, Matcha-TTS on TITW-hard. The authors conclude that "enhancement errors negatively affect the performance of the subsequently trained TTS model". Hallucinated clips can score well on UTMOS: one scored 4.01, in the top 39%.

  | Training data | UTMOS | WER (%) |
  |---|---:|---:|
  | Noisy | 2.73 | 21.31 |
  | Enhanced, all | 3.64 | 20.45 |
  | Enhanced, top 80% by confidence | **3.80** | 18.79 |
  | Enhanced, top 70% | 3.76 | **18.14** |
  | Enhanced, top 50% | 3.68 | 20.18 |

- **Meta 2309.10795** [P, agent]: high SE scores "doesn't always align" with downstream TTS quality, and spectral mapping added "unexpected noises at the end of each utterance".

**Explicit gaps** (sub-agent searched; I agree):
- **No study shows a codec-LM TTS reproducing restoration artefacts.** It is plausible, because the codec preserves "acoustic environmental features" (Kurisu notes) [I].
- **No study measures restoration damage on acted speech** (shouts, sobs, whispers).
- **No study starts from clean studio audio with deliberate effects.** All the "restoration helps" results start from noisy found data, where the alternative was keeping the noise, not dropping 5% of clips.

---

## 5. Q4: separating the effect from the dry voice

### 5.1 Reverb

**Joint dereverberation plus RIR estimation exists:**

| Method | Status | What it gives | Source |
|---|---|---|---|
| **BUDDy** (arXiv 2408.07472 v2, Mar 2025; Moliner, Gerkmann et al.) | Unsupervised, diffusion prior | Dry speech plus a parametric RIR ("exponential decay for each frequency subband") | [P] |
| **VINP** | Code on GitHub | Anechoic speech plus an RIR waveform from a pseudo sine-sweep measurement. CTF length 30 frames, 16 kHz, simulated English rooms | [P] |
| **Rec-RIR** | MIT | Blind RIR | [P] |
| **Gencho** (arXiv 2602.09233, ICASSP 2026, Adobe/UIUC) | No code found | Generates complex-spectrogram RIRs from reverberant speech; "integrates modularly … for acoustic matching" | [P] |
| **DARAS** (2507.08135) | — | Mamba-based blind RIR estimation | [S] |

**Implications** [I]:
- An exponential-decay subband RIR model **cannot represent a discrete echo**.
- These are 16 kHz English research tools. Their RIRs are good enough to *re-apply* (convolve at inference), not to recover a 24 kHz dry voice for training.
- A game reverb is most likely one DAW preset reused per scene. Fitting a small parametric reverb (pre-delay, RT60, damping, wet/dry) to the pooled decay of a scene's clips is cheaper and probably closer.

### 5.2 Discrete echo or delay

- The real cepstrum peak gives the delay (66 ms and 200 ms in §1). Its height and rahmonics give gain and feedback [S: cepstrum echo detection since Bogert et al. 1963; I for the application].
- This is fully DSP-reapplicable: a comb filter with the measured delay and gain.
- **Removal is the hard direction.** RemFX delay removal gained only about +2.2 dB SI-SDR on its (non-speech) test set, and the authors call delay and chorus "challenging" [P]. UVR DeEcho is the community option, with no speech evaluation [P, agent].

### 5.3 Telephone, radio or PA EQ

- **Blind magnitude response** = LTAS(effected clips) / LTAS(same speaker's clean clips). This is the Gaubitch LTASS method [S], and it is what Kurisu did.
  - It needs pooling: Kurisu's 28-clip median had a bootstrap SD of 1–4 dB per 1/6-octave band, and its F0 bands are content-dependent (`augment_notes.md`).
  - Pool **per scene or per effect cluster**, never per clip [I].
- **DeepAFx-ST** (Steinmetz, Bryan, Reiss; arXiv 2207.08759; github.com/adobe-research/DeepAFx-ST) predicts parametric-EQ and compressor settings that map an input onto a reference's style [P].
  - Its speech experiments use LibriTTS, and checkpoints are in release v0.1.0 [P].
  - The licence is the "Adobe Research License" (terms not read, likely non-commercial) [U].
  - This is the nearest thing to "estimate the phone chain's parameters" and re-apply them by DSP.
  - Peladeau & Peeters (ICASSP 2024, arXiv 2310.11781) blind-estimate an EQ → compressor → clipper chain [S].
- **Nonlinear parts are not covered by an EQ curve:** codec, μ-law, saturation. The Kurisu rule applies: add a nonlinear stage only if EQ-only copies miss a measured statistic (HNR, flatness) by more than the dry-domain difference.

### 5.4 Environment-aware TTS (dry + effect as two outputs)

| Work | Status |
|---|---|
| DAIEN-TTS (2509.14684, ICASSP 2026) | A speech-environment separation module splits clean and environment mels. No code linked [P] |
| IDEA-TTS (2412.16977, ICASSP 2025) | Environment mask plus enhanced spectrogram. No code [P] |
| DiffRENT (2401.08102) | Recording-environment transfer [S] |
| Environment-aware TTS (2110.03887) | The original task definition [S] |

None is usable off the shelf. All target noise or rooms, not stylised effects [P/I].

**Verdict for Q4.**
- A model that outputs both a 24 kHz dry estimate and the effect, for stylised game effects, **does not exist off the shelf**.
- What does work today is *estimating the effect parameters from pooled clips* and re-applying them by DSP:
  - echo: delay and gain;
  - phone: EQ curve, optionally plus 8 kHz μ-law;
  - reverb: parametric or estimated RIR.
- The dry voice comes from the clean clips the character already has [I].

---

## 6. Q5: recommendation for Tricolour [I]

### 6.1 Detectors to run (all on source files at native rate, speech frames only)

1. **`fx_pilot.py` features**, already computed in `fx_pilot_feats.csv`: narrowband, cepstral echo, per-speaker bass-cut.
2. **Brouhaha** (`pyannote/brouhaha`, gated, CPU or GPU): per-clip median C50 and median SNR over VAD frames. For clips under 2 s, pad with source context, since the source files have 0.1–1.3 s of margin.
3. **SIGMOS** (ONNX, 48 kHz): MOS_REVERB, MOS_COL and MOS_NOISE.
4. **Optional:** NISQA coloration (non-commercial weights) as a second phone signal; DNSMOS BAK as a noise alarm.
5. **Skip:** audio-LLM captioners (near chance on MRMAD) and AudioSet taggers (50–67% label accuracy for Echo and Reverberation).

**Cost** [U]: Brouhaha + SIGMOS over 25k short clips is well under 1 GPU-hour, and feasible on CPU overnight.

### 6.2 Starting thresholds (to be replaced by values fitted on the labelled sample)

| Effect | Rule (start) | Basis |
|---|---|---|
| Telephone / narrowband | `vhi_db` < −45 **and** `bw_hz` ≤ 6.4 kHz; cross-check with the URGENT/NeMo −50 dB bandwidth < 7 kHz on active frames | Synthetic AUC 0.00 [L]; FireRedTTS 7 kHz and HiFiTTS-2 11 kHz rules [P] |
| Stylised phone EQ (full band, bass-cut) | Per-speaker robust z of `low_db` < −3, **and** SIGMOS COL or NISQA COL in the speaker's bottom 5% | F0 confound [L] |
| Discrete echo / PA | `cep_z` > 30 with lag ≥ 55 ms; > 50 is "certain" | Synthetic dry p90 = 14 [L] |
| Diffuse reverb | Clip in the bottom 5% of **its own speaker's** Brouhaha C50 distribution **or** SIGMOS REVERB distribution, and confirmed by the other | No transferable absolute value exists [P/I] |
| Noise | Brouhaha SNR < 25 dB (the WenetSpeech-Yue TTS rule) or DNSMOS BAK < 3 | [P, agent] / [I] |

**Scene propagation** (the most important step):
- Aggregate each score over (series, scene) and contiguous `order` runs.
- If ≥ 30% of a scene's clips from one speaker are flagged with the same effect, flag all of that speaker's clips in the scene for review.
- **Effects are per speaker, not per scene.** In a phone call only the far end is processed. Never propagate across speakers [I; §1 shows clustering, not yet the asymmetry].

### 6.3 Restore, drop, or keep-and-label

**Decision rule** [I, following 2601.12254 and the absence of any "restore intentional effects" evidence]:
- Restoration only pays where an effect class holds a large share of one **character's** data (more than about 10%) **and** the effected lines carry something the clean lines lack, such as a scene's emotional register.
- At the measured ~0.4–1.5% per class, drop or label.

| Class | Measured scale [L] | Action |
|---|---|---|
| Narrowband phone/radio | 369 clips, 0.33 h | **Do not restore.** Bandwidth extension invents the 4–12 kHz band, where much of the character timbre lives. Either drop the clips from the default voice and render the phone by DSP at inference (300–3400 Hz band-pass plus the pooled LTAS curve; this game's phone looks like a true telephone band), or keep and label them (sibling's design). Never leave them in untagged |
| Discrete echo / PA | 106 clips, 0.11 h, 10 scenes | **Drop or label; do not restore.** No speech-trained delay remover exists. Re-apply by comb filter if the effect is wanted |
| Diffuse reverb | Unknown until Brouhaha/SIGMOS run | If under 5% per character, drop or label. If larger for some character, run a **restoration trial** (§6.4) with SGMSE+ EARS-Reverb (expressive training data) and Sidon. Adopt only if it passes the drift checks |
| Noise | Probably rare in studio stems | MossFormer2_SE_48K (discriminative) if flagged; otherwise leave it |
| Bass-cut "maybe-EQ" | 860 flags, heavily F0-confounded | Validate first. Most are probably genuine voice [I] |

**Rules for any restored or effected clip:**
- Never use one as `ref_audio` or as a speaker-embedding reference. In 2406.05699, enhancing prompts lowered SIM-o [P, agent].
- Keep the clean reference clips from `reference_clips.csv`.

### 6.4 Validation

**A. Detector validation with the ~100 labelled clips.** Use `fx_label_sample.csv`, which is blind; the strata are in `fx_label_key.csv`.

1. **Sample design** (all clips ≥ 1.0 s, spread round-robin across scenes):

   | Stratum | Clips | Stratum size |
   |---|---:|---:|
   | Narrowband flags | 15 | 323 |
   | Echo flags | 10 | 105 |
   | Long-tail flags | 12 | 224 |
   | Bass-cut flags | 13 | 660 |
   | Unflagged clips within ±2 lines of a flag | 20 | 3,010 |
   | Unflagged random | 30 | 17,575 |

2. **Labelling.** Listen to the wav and mark echo/reverb, phone/radio, noise and other, each 0/1.
   - For suspected reverb, also listen to the source file: the tail is cut in `wavs/`.
   - Keep the key closed until done.
   - **Hide the `text` column on the first pass.** Lines such as "喂？" or PA announcements cue the listener toward "phone" or "echo". Use the text only on a second pass to resolve doubtful clips.
3. **Precision** per detector is the positive rate within its stratum.
4. **Recall and prevalence:** weight the positives found in each unflagged stratum by stratum size ÷ sample size (inverse-probability weighting).
   - With 30 random clips and 0 positives, the 95% upper bound on hidden prevalence is about 10% (rule of three). That is coarse.
   - If the neighbour stratum turns up positives, add a second round of scene-level checks: 2–3 clips per flagged scene, about 70 scenes, which is cheaper than more random clips.
5. **Refit the thresholds.** Choose the threshold that maximises F1, or the lowest threshold with precision ≥ 0.8, per detector. Report the confusion between echo and reverb, and between phone and bass-cut.
6. **Then run Brouhaha and SIGMOS** and add their per-speaker percentile rules to the same evaluation.

**B. Restoration trial** (only if §6.3 calls for one): 200 flagged clips plus 200 clean controls from the same speakers.
- Speaker cosine of the restored clip against (i) its own original and (ii) the clean-speaker centroid, using the encoder behind `spk_score`. Clean controls set the noise floor of the measure.
- Mandarin CER against the script (Paraformer or SenseVoice; pinyin-normalised as in `research_C_eval.md`).
- Rerun the detectors: is the effect actually gone?
- Drop the worst 20–30% of restored clips by CER and cosine (2601.12254 found top-70–80% best).
- A/B listening on shouts, sobs and whispers.
- Do not trust DNSMOS or UTMOS for this: hallucinations score well [P].

**C. After training (leak check).**
- Generate the default (untagged) voice on 100 texts per character and run the same detectors.
- Each effect's flag rate must stay at the clean-corpus base rate (below 0.5% for echo and narrowband).
- This is the cheapest test that untagged effects did not leak into the default voice [I].

---

## 7. Not checked or open

- SQUIM accuracy on reverb or band-limiting: the PDF was unreadable.
- SIGMOS per-dimension accuracy and licence.
- Per-class AP for Telephone, Echo and Reverberation in PANNs, CED and BEATs.
- Kimi-Audio and Step-Audio as captioners.
- BERP weights.
- Rec-RIR metric values.
- Sidon on Windows: not stated; uv plus torch ≥ 2.8 [P, agent]. Use WSL2 if native fails.
- Whether the brief's 21,665-clip training subset excludes any of the flagged scenes. Re-filter `fx_pilot_feats.csv` by `speaker_id` to check.
- **Not one flagged clip has been listened to.** Every count in §1 is a detector count, not a verified effect count.

## 8. Sources (newest first)

- MRMAD, arXiv 2608.22236 v2 (31 Aug 2026): https://arxiv.org/html/2608.22236v2 · https://github.com/Bose/MRMAD
- Raon-OpenTTS, arXiv 2605.20830 (May 2026) [S]
- Rec-RIR, arXiv 2509.15628 (v3 Jun 2026): https://arxiv.org/abs/2509.15628 · https://github.com/Audio-WestlakeU/Rec-RIR
- DialogueSidon, arXiv 2604.09344 [S] · SE-MSB, arXiv 2609.26000 [S]
- Gencho, arXiv 2602.09233 (Feb 2026): https://arxiv.org/abs/2602.09233
- MAEB, arXiv 2602.16008 [S]
- URGENT 2026 overview, arXiv 2601.13531: https://arxiv.org/html/2601.13531
- Confidence-based filtering for SE-curated TTS data, arXiv 2601.12254 (Jan 2026): https://arxiv.org/html/2601.12254
- Qwen3-TTS report, arXiv 2601.15621
- PASE, arXiv 2511.13300: https://arxiv.org/html/2511.13300v1
- Emilia journal version, arXiv 2501.15907 v2 (Oct 2025): https://arxiv.org/html/2501.15907
- ARECHO, arXiv 2505.24518: https://arxiv.org/html/2505.24518 · Uni-VERSA, arXiv 2505.20741 · QualiSpeech, arXiv 2503.20290
- Preprocessing pipelines for in-the-wild TTS, arXiv 2510.03111
- Sidon, arXiv 2509.17052: https://arxiv.org/html/2509.17052v2 · https://huggingface.co/sarulab-speech/sidon-v0.1 · https://github.com/sarulab-speech/Sidon
- DAIEN-TTS, arXiv 2509.14684: https://arxiv.org/abs/2509.14684
- WenetSpeech-Yue, arXiv 2509.03959 · MiDashengLM: https://huggingface.co/mispeech/midashenglm-7b
- VINP, arXiv 2502.07205 v3: https://arxiv.org/html/2502.07205v3 · https://github.com/Audio-WestlakeU/VINP
- DARAS, arXiv 2507.08135 [S] · URGENT P.808 study, arXiv 2507.11306
- HiFiTTS-2, arXiv 2506.04152: https://arxiv.org/html/2506.04152
- Miipher-2, arXiv 2505.04457: https://arxiv.org/html/2505.04457
- CosyVoice 3, arXiv 2505.17589 · MiniMax-Speech, arXiv 2505.07916 · Kimi-Audio, arXiv 2504.18425 · IndexTTS2, arXiv 2506.21619 · Spark-TTS, arXiv 2503.01710
- BUDDy, arXiv 2408.07472 v2 (Mar 2025): https://arxiv.org/abs/2408.07472
- BERP, arXiv 2405.04476: https://arxiv.org/abs/2405.04476
- Audiobox Aesthetics: https://huggingface.co/facebook/audiobox-aesthetics · Qwen3-Omni Captioner: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner
- AnyEnhance, arXiv 2501.15417 · LLaSE-G1: https://github.com/Kevin-naticl/LLaSE-G1
- IDEA-TTS, arXiv 2412.16977: https://arxiv.org/abs/2412.16977
- FireRedTTS, arXiv 2409.03283: https://arxiv.org/html/2409.03283
- Microsoft flow-matching noise and DNSMOS study, arXiv 2406.05699 · WenetSpeech4TTS, arXiv 2406.05763 · EARS, arXiv 2406.06185
- Emilia, arXiv 2407.05361 · code: https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia
- DiffRENT, arXiv 2401.08102 [S] · Peladeau & Peeters, arXiv 2310.11781 [S]
- AutoPrep, arXiv 2309.13905 · Meta low-resource SE for TTS, arXiv 2309.10795
- RemFX, arXiv 2308.16177 (WASPAA 2023): https://arxiv.org/html/2308.16177v1 · https://csteinmetz1.github.io/RemFX
- LibriTTS-R, arXiv 2305.18802: https://arxiv.org/html/2305.18802
- TorchAudio-SQUIM, arXiv 2304.01448
- Brouhaha, arXiv 2210.13248 v3: https://arxiv.org/html/2210.13248v3 · https://huggingface.co/pyannote/brouhaha
- DeepAFx-ST, arXiv 2207.08759: https://github.com/adobe-research/DeepAFx-ST
- DNSMOS P.835, arXiv 2110.01763 · NISQA: https://github.com/gabrielmittag/NISQA · SIGMOS: https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos
- URGENT bandwidth estimator: https://github.com/urgent-challenge/urgent2024_challenge/blob/main/utils/estimate_audio_bandwidth.py
- SGMSE: https://github.com/sp-uhh/sgmse · ClearerVoice-Studio: https://github.com/modelscope/ClearerVoice-Studio · AP-BWE: https://github.com/yxlu-0102/AP-BWE
- GPT-SoVITS UVR5 notes: https://github.com/RVC-Boss/GPT-SoVITS (`tools/uvr5/webui.py`)
- litagin Galgame ASR set: https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz
- AudioSet Echo and Reverberation pages: https://research.google.com/audioset/dataset/echo.html · https://research.google.com/audioset/dataset/reverberation.html
- SRMR in torchmetrics: https://lightning.ai/docs/torchmetrics/stable/audio/speech_reverberation_modulation_energy_ratio.html
- Gaubitch, Brookes & Naylor, blind channel magnitude response from LTASS (EUSIPCO 2009; TASLP 2013) [S]
