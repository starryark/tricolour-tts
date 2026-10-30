# Research A: Multi-speaker SFT for a Mandarin VN cast on Qwen3-TTS-12Hz-1.7B-Base

Date: 2026-10-29. Tags: [P] primary source I read myself (paper, code, weights, or GitHub thread); [S] secondary claim (forum, community card, vendor docs without data); [I] my inference; [U] unverified.
I read numbers from arXiv HTML pages through a fetch-and-extract tool. I mark them [P], but they were extracted by a small model, so check any number before it drives a decision. I read code and weights directly with `gh` and HTTP range requests.

---

## 0. Bottom line (one paragraph per question)

1. **Joint vs per-speaker.** No 2025–2026 codec-LM paper runs a clean joint-vs-separate comparison at equal data. What exists:
   - Older TTS work [P] says joint training helps low-data speakers and does not hurt high-data speakers.
   - Seed-TTS [P] trained its 5-speaker SFT jointly.
   - On Qwen3-TTS specifically, community mSFT reports [S] say speaker switching works after the known fixes, but audio quality, pronunciation and prosody come out "slightly worse" than single-speaker SFT. Nobody gives numbers.
   - The "speaker mismatch" reports in Qwen3 mSFT forks come from numerical bugs (bf16 accumulation of averaged x-vectors) and train/infer mismatch, not from inherent timbre bleed [P]/[I]. A locked mean x-vector avoids both.
2. **Minimum data.** No published SIM/CER-vs-minutes curve exists for codec-LM TTS in the 1–30 min range. The closest data points:
   - **XTTS-v2 (AR), Hungarian.** Fine-tuning on 12 min gives SECS 0.643, about equal to 10 s zero-shot at 0.635. But listeners preferred every fine-tuned system over zero-shot (p<0.001). 1 h gives 0.706 and 13 h gives 0.727 [P].
   - **BnTTS (XTTS).** A joint fine-tune of 4 speakers at 20 min each improved SECS by 0.02 and cut CER from 0.052 to 0.034 versus zero-shot [P].
   - **Mega-TTS 2 (2023, not an LLM).** Prompting with the same audio beats a fine-tuning baseline at 10 s and 60 s, and ties at 300 s [P].

   [I] For an x-vector-row voice the crossover is probably at a few minutes; below about 3–5 min, ICL with a good reference is likely the better choice.
3. **Imbalance handling.** No TTS-specific temperature or √-frequency ablation exists.
   - The only TTS imbalance study (Luong 2019, ~12:1 imbalance) [P] found that plain concatenation worked well, over-sampling every speaker to the largest speaker's size degraded the majority speakers, under-sampling gave nothing, and a 3-model ensemble was best.
   - Multilingual NMT evidence (cross-domain):
     - Upsampling converges faster but carries "a higher risk of overfitting", and a "Cooldown" schedule (high T early, lower later) works [P, abstract].
     - The specific claims that T=3–5 overfits low-resource directions and that T=1–2 often beats T=5 for large models come from search snippets [S].

   [I] Use about T=2 (√-hours weighting) with a per-speaker repeat cap.
4. **Qwen3-TTS specifics.**
   - **PR #384** is open, has 0 reviews and no results. It exports the *first-seen* x-vector per speaker, not a mean, uses plain shuffle and keeps speaker_encoder in AdamW [P].
   - **Community multi-speaker checkpoints on HF** (scan of 827 repos):
     - techladderai/Qwen3-TTS-kashufa (387 speakers)
     - FelysNeko/…-Delta-me13 (7 Honkai: Star Rail young-female characters in Mandarin, the closest analogue to your cast)
     - TaoLiveAIGC TBLive-CustomVoice (≥19 speaker rows written, 6 exposed)
     - MahmoudIbrahim/A3S (3 speakers)
     - karthik-1599 Kannada (2 speakers)

     None publishes a speaker-distinctness evaluation [P].
   - **TaoLive weights** [P]: rows at 2800+id with x-vector-scale norms (~19), and 13 unreleased rows overwritten with copies of one released speaker's vector. The method is consistent with frozen x-vectors written into rows [I].
   - I found no report or fix for x-vector-only collapse between similar voices.
5. **Conditioning design.**
   - Ablations exist only outside fine-tuned codec LMs:
     - MiniMax-Speech (2025) [P]: a pretrained SV embedding gives SIM equal to or higher than its learnable encoder, but WER is worse (1.400 vs 1.252).
     - Chien et al. (ICASSP 2021) [P]: pretrained + learnable lookup embedding gave the best similarity, especially at 5 utterances per speaker.
   - CosyVoice's SFT recipe conditions on a per-speaker *mean* x-vector [P, code], and CosyVoice 3 uses clustered "timbre centers" [P]. These are design precedents for your locked mean.
6. **Per-speaker LoRA vs joint.**
   - VoxCPM docs claim LoRA r=32 reaches ~98% of full-FT SIM for a single speaker [S] and support hot-swapping [S].
   - JoyVoice recommends LoRA-SFT under about 1 h [P, no numbers].
   - NeuTTS/Qwen-0.5B LoRA (2026) [P]: a joint 6-speaker model using only 11–22% of each speaker's data came within 7–9% SIM of dedicated per-speaker models (confounded by data amount). Joint training improved MOS on *unseen* speakers by up to +0.29.
   - Nothing exists for GPT-SoVITS multi-character quality.

---

## 1. Joint multi-speaker SFT vs per-speaker fine-tunes

### Primary evidence
- **Seed-TTS (2024-06)** [P]. "5 speakers (3 female and 2 male), each with speech data ranging from 1 to 10 hours". Fine-tuned *jointly*, with "an additional speaker index token". The baseline was ICL with a 20 s prompt. Results: WER 3.15→2.83, SIM 0.779→0.789, CMOS +0.37. https://arxiv.org/abs/2406.02430
- **CosyVoice 3 (arXiv 2505.17589, 2025-05)** [P].
  - Speaker SFT mixes target-speaker data with the pretraining instruction data. Identity goes in a natural-language prompt ("You are Speaker A…"), and the speaker or style prompt is randomly masked "to enhance the model's transfer capability".
  - Sec. 5.7: "we utilize an unsupervised clustering method to identify the timbre centers for each speaker… used as speaker embeddings in the conditional flow matching model."
  - No SIM table, no data amounts, and no single-vs-multi comparison; Fig. 5 shows WER/CER only. https://arxiv.org/abs/2505.17589
- **JoyVoice (arXiv 2512.19090, 2025-12)** [P].
  - Reproduces mSFT: 5 speakers at 2–3 h each (~12 h total), a speaker prompt tag in the system prompt, constant lr 1e-5, and both full SFT and LoRA-SFT.
  - It recommends LoRA-SFT when target data is under about 1 h.
  - The only result is Fig. 5. It reports no SIM, CER or confusion numbers and no comparison with zero-shot. https://arxiv.org/abs/2512.19090
- **BnTTS (NAACL Findings 2025, XTTS-based)** [P, PDF read].
  - "We randomly selected 20 minutes of audio for each speaker and fine-tuned the model in a multi-speaker" setting: 4 studio speakers, with the audio and speaker encoders frozen.
  - Versus zero-shot BnTTS-0: SECS(ref) 0.529→0.548, SECS(prompt) 0.576→0.586, CER 0.052→0.034, SMOS 4.456→4.624.
  - There is no per-speaker baseline. https://aclanthology.org/2025.findings-naacl.279.pdf
- **Purwar & Choudhary, "When Fine-Tuning Fails and when it Generalises" (arXiv 2603.10904, 2026-03)** [P].
  - Setup: NeuTTS (Qwen-0.5B LM backbone plus NeuCodec), LoRA r8 on q/k/v, conditioned on reference audio, not speaker IDs.
  - "Pure FT" means one LoRA per speaker on all data (3.5–18.5 h). "Mix FT" means one joint model on 6 speakers using 2/9 of the HiFi-TTS data and 1/9 of the LibriHeavy data for each speaker ("only 11-22% of the per-speaker training data").
  - SIM averages: Pure 0.774 vs Mix 0.721 (HiFi-TTS speakers), and Pure 0.554 vs Mix 0.506 (LibriHeavy). The paper calls this "within 7% and 9%".
  - Per speaker (Pure → Mix): 0.790→0.723, 0.794→0.768, 0.737→0.669, 0.503→0.442, 0.678→0.643, 0.481→0.432.
  - **Confounded:** Mix used far less data per speaker, so the gap is not a pure joint-vs-separate effect.
  - Transfer evidence: a 3-speaker joint LoRA (2 h each) improved DNS-MOS on *unseen* speakers from 3.513 to 3.806 (+0.293) versus Pure FT. https://arxiv.org/html/2603.10904v1
- **Luong et al., Interspeech 2019, "Training multi-speaker neural TTS using speaker-imbalanced corpora"** [P].
  - 10 Japanese female speakers with 735 to 8,750 utterances each, conditioned on a one-hot speaker code.
  - Combining all data (MU) beat speaker-dependent models. "speakers with less data benefit the most", and high-data speakers did not degrade (AB test: 997 sessions, 175 listeners). https://arxiv.org/abs/1904.00771
- **Low-data speakers in a pooled fine-tune without IDs.** In the Greek Parler-TTS study (arXiv 2609.10022, 2026-09) [P], "A critical failure mode emerges after full fine-tuning on the multi-speaker pool." The Common Voice part had 174 speakers and 15.5 h in total, which works out to about 5 min per speaker (my arithmetic) [I]. The paper also says "many speakers contribute only a few minutes each".
  - Identity there comes from text descriptions, not IDs or embeddings. That makes it a warning about pooled training *without* a strong per-speaker conditioner, not about x-vector rows [I]. https://arxiv.org/abs/2609.10022

### Qwen3-TTS community evidence on mSFT
- **QwenLM/Qwen3-TTS #179, 2026-03-26** [S]. CamellIyquitous used the vspeech mSFT fork with the label-shift and text_projection fixes: "Speaker switching works as expected. However… msft seems slightly worse in: audio quality; pronunciation accuracy for certain characters; pause/prosody."
- **Same thread, 2026-03-31** [S]. sleepbro: "I also experimented with msft using a large-scale dataset. Although the model does not severely collapse, I observe a similar drop in quality." No numbers. https://github.com/QwenLM/Qwen3-TTS/issues/179#issuecomment-4160696443
- **vspeech/Qwen3-TTS-Train #3 "Speaker mismatch in the output" (2026-03-30)** [S]. The fix offered was to accumulate the speaker-embedding mean in fp32 rather than bf16 ("bf16 precision causes truncation issues"). Another user reported that sequentially fine-tuning speaker after speaker sped up speech; this is the label-shift bug era. https://github.com/vspeech/Qwen3-TTS-Train/issues/3
- **vspeech README** [S] recommends "At least 30 minutes per speaker for multi-speaker" and "At least 1 hour" for a single speaker. No evidence is given. https://github.com/vspeech/Qwen3-TTS-Train
- **Aliyun Bailian CosyVoice tuning docs** [S, vendor]: "训练数据混合多个发音人会导致音色还原度下降" (mixing multiple speakers lowers timbre fidelity). This is about *unlabeled* mixing in a single-voice product, not ID-conditioned joint SFT. They suggest 1–10 h and at least 150 clips, with little gain past 10 h. https://help.aliyun.com/zh/model-studio/fine-tune-speech-synthesis-model-by-api
- **so-vits-svc community docs (folklore)** [S]: "training with multiple speakers may lead to worsened leaking of voice timbre", with a recommended maximum of about 5 speakers. This is a VITS-style SVC, not a codec LM. Relevance is low.

### [I] Synthesis for your cast
- The direct evidence (Luong, Seed-TTS, BnTTS, and the Qwen community saying switching works) supports one joint model for the 6 mains and roughly 8 side characters. Expect a small quality cost versus dedicated single-speaker SFT, as the community reports.
- "Bleed" between similar voices is unmeasured in every source. Nobody has published a speaker-confusion matrix for an ID/x-vector-conditioned codec LM.

---

## 2. Minimum data per speaker: fine-tuned identity vs zero-shot ICL

- **Székely et al., Interspeech 2025 (XTTS-v2, AR GPT, Hungarian; single speaker)** [P, PDF read]. ECAPA2 SECS against held-out spontaneous speech:

  | System | Data | SECS |
  |---|---|---|
  | FT-Read | 195 sentences, 12 min | 0.643 |
  | FT-Spont-Short | 1 h | 0.706 |
  | FT-Spont-Long | 13 h | 0.727 |
  | ZS-Typical | 10 s reference | 0.635 |

  - MUSHRA (30 listeners): "each of the three fine-tuned systems was preferred over each zero-shot system (p<0.001)". FT-Spont-Short vs FT-Read was not significant (p=0.213).
  - Takeaway: at 12 min the objective SIM is a tie with zero-shot, but listeners prefer fine-tuning. https://www.isca-archive.org/interspeech_2025/szekely25_interspeech.pdf
- **BnTTS** (above) [P]: joint fine-tuning at 20 min per speaker beats zero-shot by a small margin on SECS (+0.019) and clearly on CER and SMOS.
- **Mega-TTS 2 (arXiv 2307.07218; ICLR 2024; not an LLM)** [P]. Fine-tuning baseline vs multi-sentence prompting on the same audio, LibriSpeech:

  | Audio | Fine-tune SIM | Prompting SIM | Fine-tune WER | Prompting WER |
  |---|---|---|---|---|
  | 10 s | 0.895 | 0.905 | 4.74% | 2.28% |
  | 60 s | 0.923 | 0.926 | 4.18% | 2.24% |
  | 300 s | 0.934 | 0.932 | 3.11% | 2.23% |

  Prompting wins up to 60 s and ties at 5 min. https://arxiv.org/abs/2307.07218
- **Qwen3-TTS community.**
  - #179, fumyou13 (2026-02-09) [S], after the label-shift and text_projection fixes, with 5k samples: "By increasing the training epochs (about 20 epochs), the speaker similarity gets progressively improved and performs better than zero-shot voice cloning."
  - #72 (Jan 2026, before the fixes) [S]: x-vector-only fine-tunes sounded flatter than ICL cloning.
  - #343 (2026-07-10) [S]: a LoRA on 21 min of single-speaker English gave wrong timbre or occasional gender flips in about 15% of *short* utterances, concentrated in the first 1–2 s. A commenter fixed it "pretty much perfectly" by passing the Base model's x-vector for the same speaker as `speaker_embeddings` at inference. https://github.com/QwenLM/Qwen3-TTS/issues/343
- **Vendor and doc guidance** [S]: GPT-SoVITS claims about 1 min; VoxCPM suggests 5–50 clips (README: 5–10 min); Aliyun suggests 1–10 h; vspeech suggests 30 min per speaker in multi-speaker training.
- **No published curve of SIM/CER against minutes exists for codec-LM TTS at 1–30 min.**
- [I] For your tiers:
  - 6 mains (1.9–5.9 h): clearly fine-tune them.
  - About 8 side characters (15–30 min): fine-tuning is likely at least equal to ICL on SIM and better on stability and subjective quality (Székely, BnTTS).
  - About 9 at 3–8 min: borderline, roughly where Mega-TTS 2 shows prompting catching up. Train rows, but A/B each against ICL from the best 10–20 s reference.
  - Tail (a few clips): use ICL or the x-vector only, with no dedicated training emphasis.
  - Your prior project's 3-min replay speakers "worked" as regularizers. That tells you nothing about their own perceived quality.

---

## 3. Data imbalance handling

- **TTS-specific evidence (only one study).** Luong 2019 [P] compared:
  - UN: under-sample to 735 per speaker
  - MU: all 32,076 utterances
  - OV: over-sample to 8,750 per speaker
  - E1–E3/EN: three models at 3,000 per speaker with replacement, then an ensemble

  Results: UN gave no gain even for the smallest speaker. OV was better than speaker-dependent overall, but majority speakers showed "noticeable F0 correlation degradation". MU was consistently good. The EN ensemble was best. Imbalance was about 12:1; yours is above 100:1 (5.9 h vs 3 min) [I].
- **ZeSTA (arXiv 2603.04219, Interspeech 2026; VITS, not an LLM)** [P]. Each real utterance repeated 3× ("OS"), combined with zero-shot-TTS synthetic data. OS alone gave "limited and unstable improvements".
  - Naive synthetic augmentation *lowered* SECS: real 10% gave 0.818; adding Fish-Speech synthetic data gave 0.765. A domain embedding plus OS recovered 0.815.
  - [I] If you consider cloning extra lines for tail characters with zero-shot Qwen, expect identity dilution unless the clones are flagged. https://arxiv.org/html/2603.04219
- **Cross-domain (multilingual MT) evidence** [P, but transfer to TTS is I]:
  - Li et al., "Upsample or Upweight?" (NAACL 2025, arXiv 2410.04579): temperature sampling and loss upweighting are equivalent under full-batch GD but differ under SGD. Sampling has lower gradient variance, which means "faster convergence but a higher risk of overfitting". Their **Cooldown** schedule starts with heavy upsampling and gradually reduces it. https://arxiv.org/abs/2410.04579
  - Shaham et al. (ACL 2023, arXiv 2212.07530): interference mainly matters when the model is small relative to the data, and temperature tuning is "key". The search-result summary says T=1–2 often beats the common T=5 and that larger models degrade at T≥3 [S for those specifics, abstract only]. https://arxiv.org/abs/2212.07530
  - arXiv 2312.06134: τ>1 is common, and τ≈3.33 was best for low-resource in their tuning [S, snippet].
- **Qwen3 trainers.** PR #384 and the vspeech fork both use `shuffle=True` with no speaker balancing [P].
- [I] Recommended starting point:
  - Sample with p(s) ∝ hours_s^0.5 (T=2), and cap each speaker's effective epochs (for example, no clip repeated more than about 8–10 times over the run).
  - Optionally use Cooldown: T=2 for the first ~30% of steps, then T≈1.4.
  - With your numbers (~22.5 h total), α=0.5 gives:

    | Speaker | Natural share | Sampled share | Change |
    |---|---|---|---|
    | 5.9 h main | 26% | 13.5% | 0.5× |
    | 15–30 min side | ~1.7% | ~3.4% | ~2× |
    | 3–8 min | ~0.4% | ~1.7% | ~4× |

  - Loss reweighting with the same α is theoretically equivalent but noisier (Li et al.).
  - Monitor held-out loss *per speaker*; small speakers overfit first. Watch for text-ignoring or verbatim replay, the overfitting sign the VoxCPM docs describe.

---

## 4. Qwen3-TTS multi-speaker specifics

### PR #384 (QwenLM/Qwen3-TTS, opened 2026-10-17 by Anson-Trio) [P]
- **Status:** OPEN as of 2026-10-29, last updated 2026-10-17. One commit (898bd15), 0 reviews, 0 comments, no results reported. https://github.com/QwenLM/Qwen3-TTS/pull/384
- **Design, from the diff:**
  - Optional `"speaker"` field.
  - The speaker x-vector is computed live from each sample's `ref_audio` (`.detach()`) and placed at `input_codec_embedding[:, 6, :]`.
  - Export uses the **first-seen** embedding per speaker (`if spk_key not in target_speaker_embeddings`), not a mean. Rows are assigned as `3000 + idx` over sorted speaker names.
  - The README tells users to "use the same `ref_audio` for all samples of a given speaker".
  - No speaker balancing (`shuffle=True`).
  - The optimizer is unchanged: AdamW over all model parameters with weight_decay 0.01, so speaker_encoder is not explicitly frozen.
- **Differences from your trainer [I].** Yours uses a locked mean over clean takes and a frozen encoder. #384 uses the first-seen or live x-vector, so if `ref_audio` varies per sample, training conditions on per-utterance vectors while inference uses a single first-seen vector.

### Related PRs and issues
- **PR #351 (2026-07-24, open)** [P] claims AdamW weight decay silently shrinks the detached speaker_encoder and that averaged embeddings collapse as ‖e0‖/(epoch+1). https://github.com/QwenLM/Qwen3-TTS/pull/351
  - [I] Caveat: PyTorch AdamW skips parameters whose `.grad` is None, weight decay included, and a detached encoder never gets grads under `zero_grad(set_to_none=True)`. The bf16-accumulation explanation (vspeech #3) fits the 1/(epoch+1) collapse better: a bf16 running sum saturates while the count grows.
  - Either way, a precomputed fp32 locked mean is immune.
- **vspeech/Qwen3-TTS-Train** (2026-03, 119★) is the main community mSFT fork [P].
  - `--multi_speaker`, rows from 3000.
  - It trains on per-utterance x-vectors and saves the *running average across the run*, accumulated in the model dtype.
  - Users report speaker mismatch (fix: fp32 accumulation) and an mSFT quality drop. https://github.com/vspeech/Qwen3-TTS-Train
- **QwenLM Discussions** [P]: #143 (2026-01, Chinese) asks when multi-speaker FT will be supported; #348 (2026-07) asks how to do multi-speaker FT and was never resolved. Neither reports confusion.

### Community multi-speaker checkpoints (my scan, 2026-10-29) [P]
I scanned 827 HF repos matching Qwen3-TTS or derived from the Base checkpoints, read `config.json` from 515, and found 122 with `spk_id` entries for more than one speaker. Excluding mirrors of the official 9-voice CustomVoice:

| Repo | Speakers | Rows | Data / notes |
|---|---|---|---|
| techladderai/Qwen3-TTS-kashufa (2026-07) | 387 (IndicVoices speaker IDs) | 3000+ | Card is the stock Qwen README. No results. |
| FelysNeko/Qwen3-TTS-12Hz-1.7B-Delta-me13 (2026-09) | 7 HSR characters (Cyrene, Castorice, Aglaea, Hyacine, Cipher, Hysilens, Cerydra), all `/chinese(prc)` | 3000–3006 | Empty card. Data from project-felys/delta-me13 game voice lines: Mandarin line counts 1852 / 1627 / 954 / 797 / 629 / 432 / 385. Corpus totals are text-length units, not durations. [I] At ~4–5 chars/s that is about 2.5–3 h for Cyrene down to about 0.6–0.7 h for Cerydra. Whether all lines were trained is unknown. No distinctness evaluation. |
| TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice | 6 exposed (3 female, 3 male live-commerce voices) | 2801, 2804, 2826, 2880, 2896, 2915 (=2800+id) | No training repo found. The card points to a "companion training repository" without a link. |
| MahmoudIbrahim/A3S (0.6B, 2026-10) | 3 Arabic (omar, fahad, hanouf) | 3000–3002 | No evaluation. |
| karthik-1599/qwen3-tts-kannada-1.7b-experiment (2026-09) | 2 (syspin male/female, about 89 h Kannada plus 38 h synthetic English replay) | 3000–3001 | Reports held-out loss only. |
| starrydark/Makise_Kurisu_Qwen3_TTS | 1 character, 3 rows (kurisu / amadeus / phone) | 3000–3002 | Same pattern as your earlier project. |

**No public multi-speaker Qwen3-TTS fine-tune reports speaker-distinctness or confusion results.**

### How TaoLive did it (weight forensics; I read `talker.model.codec_embedding.weight` by HTTP range) [P]
- **TBLive-Base vs Qwen Base:** 2041 of 2048 codec rows changed (full continual pretraining). TBLive-Base has no `spk_id`.
- **TBLive-CustomVoice vs TBLive-Base:**
  - 2041 codec rows changed again (full fine-tune).
  - Exactly 19 rows changed in the speaker range: 2801, 2802, 2804, 2806, 2813, 2815, 2817, 2823–2827, 2835, 2842, 2845, 2848, 2880, 2896, 2915.
  - The 13 unexposed rows are **bit-identical copies of speaker "4"** (row 2804). [I] The unreleased voices were scrubbed after training, so at least 19 speakers were trained jointly and 6 were exposed.
- **Speaker-row norms:**
  - TBLive: ~18.8–19.4
  - FelysNeko: ~17.7
  - Official Qwen CustomVoice: 13.9–15.6
  - Codec rows: median 0.72
  - Unused rows in Base: ~0.023

  [I] All three use x-vector-scale vectors written into the row, consistent with the official frozen x-vector recipe. Whether training conditioned on a per-utterance or a mean x-vector cannot be determined from weights.
- **Official CustomVoice vs Base:** *all* 3072 codec rows differ, and the speaker rows sit at 2861–3066 inside the 3072-row vocab [P]. [I] CustomVoice is not a simple SFT of the released Base weights.

### x-vector geometry of similar voices [P numbers, I interpretation]
Raw cosines between speaker rows are 0.90–0.99 everywhere because x-vectors share a large mean component. After subtracting the mean of all 35 rows written across the three models:

| Set | Mean off-diagonal centered cosine | Max pair |
|---|---|---|
| Qwen CustomVoice, 9 mixed voices | 0.53 | 0.71 (serena–ono_anna) |
| TBLive, 3 female + 3 male | 0.32 | 0.75 (male promo–male warm) |
| Felys, 7 young-female HSR characters | 0.69 (range 0.57–0.81) | 0.81 (cyrene–hyacine) |

- [I] Same-gender, same-age characters sit noticeably closer in x-vector space. The talker must turn input differences of this size into distinct timbres.
- No source links these numbers to perceptual confusion. Run the same check on your cast's locked means and A/B-test pairs above ~0.8 centered.
- **Fixes for similar-voice collapse** (learned delta, low-lr rows, speaker-ID token): no Qwen3-specific report or fix found. Issue #213 (known to you) is the only data point against unconstrained row training.

---

## 5. Speaker conditioning design: frozen x-vector vs learnable vs x-vector + residual

- **MiniMax-Speech (arXiv 2505.07916, 2025-05), Table 4** [P], trained on "a substantial subset of our Chinese speech data":

  | Conditioning | Mode | WER | SIM |
  |---|---|---|---|
  | Learnable encoder | zero-shot | 1.252 | 0.730 |
  | Learnable encoder | one-shot | 1.243 | 0.746 |
  | Pretrained SV embedding | zero-shot | 1.400 | 0.746 |
  | Pretrained SV embedding | one-shot | 1.704 | 0.744 |
  | Prompt only | one-shot | 1.207 | 0.726 |

  - The pretrained SV embedding matches SIM but costs intelligibility.
  - Their "Professional Voice Clone" optimizes *only* the speaker conditional embedding with the AR model frozen. It is claimed better than SFT or LoRA on cost, but no numbers are given. https://arxiv.org/abs/2505.07916
- **Chien et al., ICASSP 2021 (arXiv 2103.04088; FastSpeech 2 / Tacotron 2)** [P].
  - SV accuracy, Track 1 (100 utterances per speaker) / Track 2 (5 utterances): x-vector .973/.623, VC-pretrained .980/.837, learnable lookup .988/.490, VC + lookup **.983/.937**.
  - The extraction tool flagged that the combined row's label was inferred from the text.
  - Similarity MOS (Track 2): x-vector 3.25, lookup 3.27, VC + lookup **3.38**. Quality MOS: lookup 3.65, combined 3.55.
  - Pretrained encoders stayed frozen. Inference used the *average* embedding over a speaker's utterances, which the authors found "more stable". https://arxiv.org/abs/2103.04088
- **CosyVoice recipe** [P, code].
  - `tools/extract_embedding.py` builds `spk2embedding` as the mean of utterance embeddings.
  - The configs say `use_spk_embedding: False # change to True during sft`, so SFT conditions on the speaker mean, not per-utterance x-vectors (examples/libritts/cosyvoice2/conf/cosyvoice2.yaml). https://github.com/FunAudioLLM/CosyVoice
  - CosyVoice 3 uses clustered timbre centers in SFT [P].
- **Qwen3 #343 workaround** [S]: overriding a fine-tuned speaker row with the Base model's x-vector at inference stabilized onsets.
- **No ablation of x-vector vs learnable vs x-vector + residual exists for fine-tuned AR codec-LM TTS.**
- [I] If similar young-female voices collapse, the evidence-weighted option is x-vector + residual:
  - The residual is zero-initialized and trained at about 0.1× the base lr, with L2 toward zero (that is, toward the x-vector) and a norm cap.
  - Chien supports the combination for low-data speakers; MiniMax PVC shows embedding-only optimization can work.
  - #213 says unconstrained row training is unstable.
  - A speaker-ID text token, Seed-TTS style, is the other precedent.

---

## 6. Per-speaker LoRA adapters vs one joint model

- **VoxCPM docs** [S]:
  - "5–50 clips" to clone a single speaker with LoRA; 1–3 epochs ("going further often hurts").
  - LoRA r=32 "achieved approximately 98% of the speaker similarity of full fine-tuning" in internal single-speaker benchmarks, with about half the VRAM.
  - Hot-swap with `load_lora()`, `unload_lora()` and `set_lora_enabled()`, compatible with torch.compile.
  - No multi-speaker guidance. https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
- **JoyVoice** [P]: LoRA-SFT recommended below about 1 h of target data. No numbers.
- **NeuTTS LoRA study (2603.10904)** [P]: dedicated per-speaker LoRAs beat one joint LoRA on SIM by 7–9%, but the joint model used 11–22% of the data. The joint model gave better MOS on unseen speakers.
- **Qwen3 #343** [S]: a single-speaker LoRA (r16, 21 min) showed onset timbre instability on about 15% of short utterances. That matters for VN-style short lines.
- **VoiceGuider (arXiv 2409.15759)** [S, snippet]: PEFT speaker adaptation "may decline compared to full-finetuned counterparts, especially for out-of-domain speakers".
- **GPT-SoVITS per-character models vs multi-speaker:** I found no comparison. Per-character models are the community norm [S].
- [I] Per-speaker LoRAs mainly buy isolation (no bleed, independent retraining). They give up cross-speaker transfer, which matters most for the 3–30 min speakers. A hybrid is a reasonable fallback: one joint full-FT model for everyone, plus optional per-speaker LoRAs only for mains that sound diluted.

---

## Sources (primary first)
- Seed-TTS: https://arxiv.org/abs/2406.02430
- CosyVoice 3: https://arxiv.org/abs/2505.17589 · CosyVoice code: https://github.com/FunAudioLLM/CosyVoice
- JoyVoice: https://arxiv.org/abs/2512.19090
- MiniMax-Speech: https://arxiv.org/abs/2505.07916
- Chien et al. 2021: https://arxiv.org/abs/2103.04088
- Luong et al. 2019: https://arxiv.org/abs/1904.00771
- Székely et al. 2025: https://www.isca-archive.org/interspeech_2025/szekely25_interspeech.pdf
- BnTTS: https://aclanthology.org/2025.findings-naacl.279.pdf
- Mega-TTS 2: https://arxiv.org/abs/2307.07218
- Purwar & Choudhary 2026: https://arxiv.org/abs/2603.10904
- ZeSTA: https://arxiv.org/abs/2603.04219
- Greek Parler-TTS: https://arxiv.org/abs/2609.10022
- Li et al. (Upsample or Upweight): https://arxiv.org/abs/2410.04579 · Shaham et al.: https://arxiv.org/abs/2212.07530
- Qwen3-TTS tech report: https://arxiv.org/abs/2601.15621. It describes speaker fine-tuning (Aiden voice) but gives no per-speaker data or joint-vs-separate details, and its intro leaves the preset-voice count as "x".
- QwenLM/Qwen3-TTS PR #384, PR #351, issues #179, #343, #72; Discussions #143, #348: https://github.com/QwenLM/Qwen3-TTS
- vspeech/Qwen3-TTS-Train (issues #3, PR #6): https://github.com/vspeech/Qwen3-TTS-Train
- HF: TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice, TaoLiveAIGC/Qwen3-TTS-TBLive-Base, FelysNeko/Qwen3-TTS-12Hz-1.7B-Delta-me13, techladderai/Qwen3-TTS-kashufa, MahmoudIbrahim/A3S, karthik-1599/qwen3-tts-kannada-1.7b-experiment
- project-felys/delta-me13 (data pipeline only): https://github.com/project-felys/delta-me13
- VoxCPM docs: https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html · Aliyun CosyVoice tuning: https://help.aliyun.com/zh/model-studio/fine-tune-speech-synthesis-model-by-api

Scratch artifacts (scan output, extracted rows): `<local scratch folder, not published>`, `<local scratch folder, not published>` (.npy codec_embedding matrices), scripts in `<local scratch folder, not published>`.
