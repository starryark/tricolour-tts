# Tricolour cast voices: the best approach (research as of 2026-10-29)

The goal is one character voice for each named character in `voice_dataset_game`: native Mandarin, no reference audio needed at inference. This report decides how to get there.

It builds on the STEINS;GATE Kurisu work (`E:\STEINS;GATEVoice\kurisu_v2\research\*.md`, v3 published 2026-10-29) and does not repeat its trainer-bug analysis, accent research or model survey. Supporting notes are in `research/notes/`:
- `research_A_multispeaker.md`: joint vs per-speaker training, minimum data, imbalance, conditioning;
- `research_B_mandarin_models.md`: the model re-rank for Mandarin;
- `research_C_eval.md`: Mandarin evaluation tools and listening-test design;
- `spk_separability.py` with `spk_sep.json`: today's local speaker-separability measurement.

**Tags:**
- **[P]** primary source (paper, repo code, issue/PR, model card, weights) read by me or a research sub-agent;
- **[S]** secondary source (blog, forum, search snippet, community claim);
- **[I]** inference;
- **[U]** unverified;
- **[L]** measured or read locally today.

Some arXiv numbers came through sub-agents' page fetches. Check any figure before it drives a decision.

---

## 0. Bottom line

1. **Train one joint multi-speaker model: a full fine-tune of Qwen3-TTS-12Hz-1.7B-Base with the Kurisu v3 trainer, one locked mean x-vector row per character.** Serve it as a `custom_voice` checkpoint: `generate_custom_voice(text, speaker="chiguo", language="Chinese")`. There are 22 rows, from the 6 mains down to the 3-minute side characters.
2. **Don't inherit v3's recipe wholesale.** Language ids, native replay, pseudo-Mandarin, accent metrics and device rows all existed to make *a Japanese voice speak Chinese*. Mandarin is already this cast's native language and Qwen3-TTS's strongest one, so that whole axis disappears.
   - **What carries over:** the fixed trainer, the data hygiene (edges, loudness, text normalisation, ASR-verified transcripts), locked speaker rows, and pre-registered checkpoint selection.
3. **The new problem is the cast: 22 speakers with a 120:1 data imbalance** (墨小菊 345 min against 叶志雄 2.7 min).
   - **Pooled training is the evidenced choice.** It helps low-data speakers most without hurting the big ones (Luong 2019), and production systems train their fixed voices jointly (Seed-TTS, CosyVoice) [P].
   - **Qwen's own speaker encoder separates the whole cast [L]:**
     - 5-clip means are classified to the right character 100% of the time;
     - single clips about 93% (median).
     - So locked x-vector rows can keep the characters apart.
   - **One pair needs listening:** 文绉/朱特 sit far closer than any other pair (§5). That is consistent with one actor voicing both roles.
4. **Run CosyVoice3-0.5B as a challenger arm, not as the default.**
   - **For it:** its zero-shot Mandarin similarity is higher (CV3-zh SS 80.0 vs 73.0 in IndexTTS-2.5's third-party harness) [P], and it is the only candidate with a documented way to keep emotion instruct after a speaker fine-tune [P].
   - **Against it:** its reference-free multi-speaker inference path has never been tested on CosyVoice3, and its recipe is DIY [P/U]. Zero-shot similarity is also not post-fine-tune similarity.
   - **Decide it with a pre-registered bake-off** on the same data and splits (§11, E4).
5. **Every trained voice must beat zero-shot cloning on its own held-out lines.**
   - **Known curves:** for XTTS-v2, a 12-minute fine-tune *tied* 10-second zero-shot on similarity (0.643 vs 0.635), 1 h reached 0.706, and listeners preferred every fine-tune [P]. On Kurisu, the fine-tune added +0.06 similarity in Chinese over Base driven by the same x-vector (0.568 → 0.626), and +0.08 in Japanese over the Base zero-shot clone (0.584 → 0.667) [L].
   - **The 3–8 min band** (6 characters) is therefore A/B'd against zero-shot before it ships.
6. **Sample by √hours with a repeat cap** rather than proportionally, as an ablation against plain pooling (§7). Proportional sampling gives 墨小菊 27% of all steps.
7. **Cost:**
   - **Local** (RTX 5060 Ti, 16 GB): data prep, codec extraction, every zero-shot baseline and all scoring. A LoRA r32 also fits [L].
   - **Rented:** the full fine-tune needs ~37 GiB (one H100).
   - **Rough numbers:** at v2's measured speed one data-epoch of this cast is ~10 min of H100 time [I from L]. Expect a budget of the same order as Kurisu v3 (~$42) [I].
8. **Main risks:**
   - bleed between similar voices;
   - wrong timbre on very short lines (Qwen #343; 14% of clips here are < 1 s);
   - overfitting the 3-minute characters;
   - no emotion control in a fine-tuned Base (#25).

9. **Updates after 2026-10-30 supersede parts of this list:**
   - §14: decisions, data build, emotion calibration;
   - §15: production effects (echo, phone, reverb), handled DSP-first.
   - Where they differ, §14–15 win.
10. **The first training round is done:** see `T1_results.md`. Selected model: C1 step 834 (CustomVoice fine-tune), with identity 0.936, identity under instruction 0.942 and CER 0.0096. In a blind A/B, angry and sad instructions were followed and happy was borderline. Cost $12.13.

---

## 1. What carries over from Kurisu, and what doesn't

| Kurisu v3 element | For Tricolour | Why |
|---|---|---|
| Fixed trainer `kurisu_v2/pod/train_sft.py` (label alignment, `text_projection`, frozen speaker encoder, per-row speaker table, language-id prefixes) | **Keep, as is** | Official `sft_12hz.py` is still single-speaker and still has the label bugs (PR #278 open; PR #384 multi-speaker open with 0 reviews) [P]. Ours already trained 58 speaker vectors in one run [L] |
| Locked mean x-vector per row, never trained (#213) | **Keep** | PR #384 exports the *first-seen* x-vector and keeps the encoder in AdamW; the vspeech fork averages in bf16 and users report speaker mismatch [P/S]. A precomputed fp32 mean avoids both [I] |
| Full FT, fp32 masters, lr 5e-6, ~1 epoch, batch 32 | **Keep as the starting point** | Nothing published beats it on target quality [L, finetune_methods §2] |
| Language ids (25% `nothink`) | **Keep, cheaply** | Train rows as `chinese` with p_auto 0.25, so both `language="Chinese"` and `Auto` are in-distribution. There is no accent problem to solve [I] |
| Native zh/en replay, pseudo-Mandarin, accent metrics (M1, shared-char SER) | **Drop** | Their job was the Japanese→Chinese accent |
| Device rows, `[sigh]`/`[nv]` tags, SFX | **Drop** | No device voices in this cast. Tags reached only 13–14% realisation on Kurisu [L] |
| Edge recut 0.12 s / 0.25 s, 5 ms fades, −23 LUFS | **Keep** (§8) | The Tricolour export left only 25 ms of padding at a −45 dBFS gate [L] |
| Two-ASR transcript verification in reading space | **Keep, in pinyin space** (§8) | Mandarin's analogue of the kana comparison |
| Pre-registered selection, paired bootstrap, blind listening | **Keep** | It caught noise-level "wins" on Kurisu [L] |

---

## 2. The cast and its data [L]

Usable under a training filter: tier A+B, 1–15 s, named characters (`kind=character`). The table shows usable minutes.

| Band | Characters (usable min) | Rows |
|---|---|---|
| **Mains, 110–345 min** | 墨小菊 `moxiaoju` 345 · 文芷 `wenzhi` 223 · 骆衍 `luoyan` 173 · 迟菓 `chiguo` 125 · 迟耀 `chiyao` 112 · 叶心 `yexin` 110 | 6 |
| **Side, 12–27 min** | 朱特 `zhute` 27 · 黄丽丽 `huanglili` 26 · 墨叔 `moshu` 25 · 丁老师 `dinglaoshi` 17 · 琳姐 `linjie` 17 · 文绉 `wenzhou` 14 · 徐阿姨 `xuyayi` 13 · 刺儿头 `ciertou` 13 | 8 |
| **Small, 4–8 min** | 林海 `linhai` 7.5 · 雯雯 `wenwen` 6.7 · 胖子 `pangzi` 5.3 · 张宏宇 `zhanghongyu` 5.2 · 唐静雨 `tangjingyu` 4.6 · 迟校长 `chixiaozhang` 4.3 | 6 |
| **Tiny, ~3 min** | 邱诚 `qiucheng` 2.9 · 叶志雄 `yezhixiong` 2.7 | 2 |
| **Excluded** | `ql` (5 clips, 0.2 min); 小白 the cat (meows); 小说 narrator (transcripts include unspoken prose); 14 bit-part banks (several actors per bank, so no single voice) | — |

**Total:** 1,279 min, 20,545 clips, 22 rows.

**Other facts that shape the plan [L]:**
- **3,552 usable clips (14%) are under 1 s.** VoxCPM's docs call clips under 1 s unstable [P], and Qwen #343 reports wrong timbre on short utterances after a LoRA [S].
- **Text hazards:**
  - `…` in 51% of clips and ASCII `...` in 1.4%. The tokenizer makes `……` and `......` different single tokens.
  - `——` in 21%, `～` in 4.7% (both single tokens).
  - Digits in 145 clips, tokenised one per digit: `2005年` → `2,0,0,5,年`.
  - Latin in 194 clips.
  - Spaces in 23 clips, mostly song lyrics (`当你碰壁依然 倔强地绽放笑颜`) and whole English/Italian lines.
- **Level differs by game:**

  | Game | Median RMS |
  |---|---:|
  | TRI | −22.1 dBFS |
  | SLS | −25.2 dBFS |
  | HM | −24.1 dBFS |

  No gain was applied in the export.
- **The most common tier-B reason is `acoustic_disagrees` (3,316 clips).** It comes from a VoxCeleb ECAPA audit that §5 shows is too weak for this cast. Those clips are probably fine.
- **HM is 44.1 kHz upsampled to 48 kHz.** Irrelevant here, because Qwen's codec runs at 24 kHz [I].

---

## 3. Base model for Mandarin (re-rank)

The Japanese survey (`kurisu_v2/research/sota_models_2026-10.md`, 2026-10-28) chose Qwen for Japanese fidelity, ja→zh and the only evidence of multilingual retention after a fine-tune. **None of those reasons applies here.** The re-rank is about Mandarin quality, multi-speaker fine-tuning and reference-free inference (full detail in `notes/research_B_mandarin_models.md`).

**Zero-shot Mandarin.** Seed-TTS-eval test-zh / test-hard, CER % / SIM. Each model's own report unless noted.

| Model | test-zh | test-hard | FT path for a multi-speaker cast | Reference-free | Licence |
|---|---|---|---|---|---|
| **Qwen3-TTS-1.7B-Base** | 0.77 (own, no SIM); 1.22 / 77.0 in third-party tables of unknown origin [U] | 6.76 / 74.8 (third-party) | **Ours works** (58 vectors trained in v3) [L]. Official: single-speaker only [P] | **Yes** (`custom_voice`) [P] | Apache-2.0 |
| **Fun-CosyVoice3-0.5B-2512** | 1.21 / 78.0; RL 0.81 / 77.4 | 6.71 / 75.8; RL **5.44** / 75.0 | mSFT by design ("Speaker A<\|endofprompt\|>") [P CV2 §2.7]. Recipe is DIY; it writes `spk2embedding.pt`, you build `spk2info.pt` yourself [P] | Code path `inference_sft(spk_id)` exists, **untested on CV3**; bistream variant crashed (#1731) [P] | Apache-2.0 |
| dots.tts SOAR | 0.94 / **81.0** | 6.60 / **79.5** | Official script; no speaker field, so one 2B checkpoint per character [P] | No [P] | Apache-2.0 |
| FireRedTTS3-Base | 1.01 / 80.9 | 6.50 / 78.4 | **No training code** [P] | — | Apache-2.0 |
| LongCat-AudioDiT-3.5B | 1.09 / 81.8 | 6.04 / 79.7 | No training code [P] | — | MIT |
| VoxCPM2 | 0.97 / 79.5 | 8.13 / 75.3 | Official LoRA, one adapter per character, hot-swappable [P] | Yes, via LoRA [P] | Apache-2.0 |
| IndexTTS-2.5 | (strong; own harness) | — | **None official** (#518 still open); community trainers unverified [P] | No, always needs a prompt [P] | bilibili licence |
| GPT-SoVITS v2ProPlus / V5 | no benchmark | — | Per-character training; the Chinese community default [S] | Reference marked required in the API [P] | MIT |

**Clean like-for-like similarity:** IndexTTS-2.5's harness gives CV3-zh SS 73.0 for Qwen3-TTS against **80.0 for CosyVoice3-0.5B** [P]. Qwen is mid-pack on Mandarin similarity.

**Verdict [I]:**
- **Qwen3-TTS-1.7B-Base stays the default.** That is the conclusion of this re-rank, not an inheritance:
  - the multi-row trainer exists and is validated twice under pre-registered gates;
  - inference needs no reference;
  - one model serves the whole cast;
  - the remaining gap is zero-shot similarity, which is exactly what a fine-tune changes. Kurisu: Base zero-shot clone 0.584 → v2 0.667 (Japanese); Base + the same x-vector 0.568 → 0.626 (Chinese) [L].
- **CosyVoice3-0.5B is the one challenger worth a run.** It has the highest Mandarin similarity among open models with any training path, plus instruct.
  - Its failure modes are known: repetition and prompt leakage (#1703) [P], and higher WER after speaker SFT (CosyVoice2: 5.34 → 7.15) [P].
  - CSP-FT measured full FT tripling WER on Fun-CosyVoice3 [P].
- **IndexTTS-2.5, CosyVoice3 and Qwen Base zero-shot are the bars** every fine-tuned voice must beat. IndexTTS-2.5 is also the community favourite for zero-shot emotional dubbing [S].
- **Set aside:**
  - dots.tts: one 2B model per character;
  - FireRedTTS3 / LongCat: no training code;
  - VoxCPM2: weak test-hard similarity;
  - GPT-SoVITS: needs a reference, no measured comparison. It is a cheap baseline to A/B against, the way Style-Bert-VITS2 was for Kurisu.

---

## 4. One joint model vs per-character models

| Evidence | Finding |
|---|---|
| Luong et al. 2019 ([1904.00771](https://arxiv.org/abs/1904.00771)), 10 speakers, ~12:1 imbalance [P] | Pooling all speakers beat per-speaker models. **Low-data speakers gained most.** High-data speakers did not degrade. Oversampling everyone to the largest speaker *hurt* the big speakers' pitch accuracy |
| Seed-TTS ([2406.02430](https://arxiv.org/abs/2406.02430)) [P] | 5 speakers (1–10 h each) jointly with a speaker-index token: WER 3.15 → 2.83, SIM 0.779 → 0.789 vs separate fine-tunes |
| BnTTS [P] | 4 speakers × 20 min jointly fine-tuned beat its zero-shot baseline (SIM +0.02, CER 0.052 → 0.034) |
| NeuTTS LoRA study ([2603.10904](https://arxiv.org/abs/2603.10904)) [P] | Joint model within 7–9% of dedicated per-speaker models on SIM, but with only 11–22% of each speaker's data (confounded); joint training improved unseen-speaker quality |
| Qwen community [S] | vspeech #179: with the label fixes, multi-speaker switching works, but quality, pronunciation and pauses are "slightly worse" than single-speaker. No numbers. The fork averaged x-vectors in bf16, a known cause of speaker mismatch |
| Production precedent [P] | TaoLive TBLive-CustomVoice: ≥ 19 speaker rows written, 6 exposed, full fine-tune, rows at x-vector magnitude (consistent with frozen x-vectors). FelysNeko Delta-me13: 7 young-female Honkai: Star Rail characters in Mandarin, the closest analogue to this cast. Neither reports speaker separation |

**Verdict [I]:** one joint model. Per-character full fine-tunes of a 1.7B model would mean 22 × 3.4 GB of checkpoints, and they would throw away the transfer that matters most for the 3–27 min characters.
- **Fallback:** if the joint model blurs a specific pair, give those characters a LoRA on top of the joint model. That is cheaper than splitting the whole cast.

---

## 5. Speaker conditioning and separability

The joint model tells characters apart **only** through each row's locked x-vector. If two centroids coincide, no training can separate them. I measured this today with Qwen3-TTS-Base's own speaker encoder, exactly as `train_sft.py` computes rows [L].

**Method (`notes/spk_separability.py`):**
- Up to 60 tier-A clips of 2.5–8 s per character, resampled to 24 kHz.
- Split into halves; centroids from half A.
- Half-B clips classified by nearest centroid.
- Cosines computed after subtracting the mean of the centroids ("centred"), because raw cosines saturate.

| | Value |
|---|---|
| Raw cosine to the **nearest other** character's centroid | 0.978–0.994 for every character (useless, as on Kurisu) |
| Centred cosine, own half-A vs half-B centroid | 0.83–0.99 |
| Centred cosine to nearest other centroid | 0.27–0.87 |
| Per-clip nearest-centroid accuracy | 0.60–1.00, median ≈ 0.93 |
| 5-clip-mean accuracy | **1.00 for all 22** (only 1–6 groups per character, so coarse) |

**Closest pairs** (centred cosine between full centroids):

| Pair | Cos |
|---|---:|
| 文绉 `wenzhou` – 朱特 `zhute` (both TRI) | **0.857** |
| 文绉 – 迟校长 | 0.693 |
| 迟菓 – 黄丽丽 | 0.660 |
| 黄丽丽 – 丁老师 | 0.651 |
| 墨小菊 – 丁老师 | 0.628 |
| 迟菓 – 琳姐 | 0.627 |

**Reading [I]:**
- **The rows are distinguishable** for every character.
- **文绉/朱特 is an outlier:** 0.857 against 0.693 for the next pair. That is consistent with one voice actor doubling both roles, which is common for side roles in Chinese VN casting [U].
  - If so, their rows will sound alike. The difference has to come from acting, which a fine-tune learns from each character's lines.
  - List this pair in the listening test.
- **The cast's mains are no harder to separate than FelysNeko's 7-character Mandarin cast.**
  - Its written rows have mean centred closeness 0.69, max 0.81 [P].
  - The centring methods differ slightly, so compare loosely.
- **The weakest single-clip accuracy was 张宏宇 `zhanghongyu` (0.60).** Short, expressive clips are noisier, but the 5-clip mean was correct.

**If bleed appears after training** (identity confusion in §10):
- **Fix:** give the affected rows a zero-initialised residual on top of the x-vector, with a low learning rate and a norm cap [I].
- **Support:** pretrained + learnable embeddings gave the best similarity at 5 utterances per speaker (Chien 2021) [P]. MiniMax-Speech found a pretrained SV embedding matched a learnable encoder on SIM but cost WER [P].
- **Caution:** #213 warns against *unconstrained* row training.
- **Status:** a contingency, not a default.

---

## 6. Minimum data and the roster bands

| Source | Data | Fine-tuned vs zero-shot |
|---|---|---|
| XTTS-v2, Székely et al., Interspeech 2025 [P] | 12 min / 1 h / 13 h | SIM 0.643 / 0.706 / 0.727 vs zero-shot (10 s) 0.635. **12 min ties on SIM**, yet listeners preferred every fine-tune (p < 0.001) |
| BnTTS [P] | 20 min | SIM +0.02, CER 0.052 → 0.034 |
| Mega-TTS 2 (2023, not an LLM) [P] | 10 s / 60 s / 5 min | Prompting beat fine-tuning at 10 s and 60 s; tied at 5 min |
| Kurisu (Qwen3-TTS, 10 h) [L] | 10.7 h | SIM 0.584 (Base zero-shot clone) → 0.667 (v2), JA |
| Kurisu v3 replay speakers [L] | ~3 min × 55 | Trained stably with locked x-vectors. Never evaluated for identity |

**There is no published curve below 30 min for an LLM/codec TTS** [U].

**Decision [I]:**
- **Mains and side characters (≥ 12 min):** trained rows, shipped by default.
- **Small (4–8 min) and tiny (~3 min):** trained rows too, because a row costs nothing and joint training is where their gain would come from. **Each ships only if it beats zero-shot on its own held-out lines** (E3); otherwise that character is served by zero-shot from its best reference clip.
- **Excluded speakers and bit banks:** no rows.
  - Bit-bank lines may be added as **per-clip-x-vector rows** (the Base's own voice-clone regime, like v3's replay) to regularise. That is an ablation, never a named voice.

---

## 7. Imbalance: how to sample

Share of training samples, and how often each clip repeats, at one data-epoch of steps [L, computed]:

| Character | Proportional (plain pooling) | √hours | hours^0.3 |
|---|---|---|---|
| 墨小菊 (345 min) | 27.0% / ×1.0 | 14.2% / ×0.53 | 9.6% / ×0.36 |
| 叶心 (110 min) | 8.6% / ×0.9 | 8.0% / ×0.88 | 6.8% / ×0.75 |
| 朱特 (27 min) | 2.1% / ×1.3 | 4.0% / ×2.5 | 4.5% / ×2.9 |
| 林海 (7.5 min) | 0.6% / ×1.0 | 2.1% / ×3.5 | 3.0% / ×5.2 |
| 叶志雄 (2.7 min) | 0.2% / ×0.9 | 1.3% / ×5.5 | 2.3% / ×9.9 |

**Evidence:**
- **Luong 2019 is the only TTS study** [P]: plain pooling was fine, and oversampling to the largest speaker hurt.
- **Machine translation:** up-sampling converges faster but overfits; a cooldown schedule (heavy up-sampling early, less later) helps ([2410.04579](https://arxiv.org/abs/2410.04579)) [P].
- **ZeSTA:** padding a small speaker with zero-shot synthetic clones lowered similarity unless the clones were flagged [P]. So **don't clone extra lines for the tail.**

**Plan [I]:**
- **E1:** plain pooling.
- **E2:** √hours with a hard cap of **3 repeats per clip**, so the tail gets ≤ 3×.
  - **Implement it as row repetition in the mix step**: each clip appears k ∈ {1, 2, 3} times, following the `--event_repeat` pattern in `kurisu_v2/pod/mix_v3.py`.
  - Then **run more steps**, so the mains still see every clip at least once.
  - Never downsample the mains. The trainer stays untouched, and "cap 3" means exactly ≤ 3 copies of a row.
  - The table above shows shares at a fixed step budget; with repetition, the mains' share falls without dropping any of their clips.
- Track held-out loss and identity per character, and adopt E2 only if it beats E1 on the small bands without costing the mains.

---

## 8. Data preparation: what changes from the export

All items are [I] unless tagged; the counts are [L].

1. **Edges.**
   - Re-cut every clip to speech bounds, with 5 ms fades and an exactly silent **0.12 s lead / 0.25 s tail**: v2's policy.
   - VoxCPM: trailing silence > 0.5 s is "the most common cause" of runaway generation [P].
   - The export's 25 ms pads are too tight for decays and breaths.
2. **Loudness.**
   - Target −23 LUFS, peak ≤ −1 dBFS. The games differ by ~3 dB [L].
   - **Recommendation: one gain per (series, character).** It keeps whispers quiet and shouts loud *within* a character. VN acting has wide dynamics, and per-clip normalisation flattens them.
   - v2 used per-clip normalisation clamped to [−6, +9] dB [L]; the earlier `kurisu_qwen3tts` build used per-source gain. Either is defensible; this is a judgement call, not a measured result.
3. **Text normalisation.**
   - ASCII `...`/`......` → `……` (353 clips). Collapse runs longer than `…………`.
   - Keep `——` and `～`: single tokens that carry delivery.
   - Write digits as spoken hanzi (145 clips; verify each against ASR: `2005年` → 二零零五年).
   - Remove spaces next to Chinese characters.
   - Keep inline English words: the Base is multilingual.
4. **Drop:**
   - whole-line foreign speech (e.g. `Ma da quanto tempo`, `Unbelievable! Such a big Crisis!`);
   - sung lines (song lyrics: lines with spaces between Chinese phrases; listen to confirm).
5. **Short clips.**
   - Keep < 1 s clips only when the transcript is lexical (not `嗯`/`啊`-only).
   - Cap them at ~5% of training rows, chosen by a seeded draw, not by order.
   - Keep 1–15 s otherwise.
   - Add short lines to the eval probes, because of #343.
6. **Transcript verification in pinyin space.**
   - Transcribe with two unrelated ASR families: Qwen3-ASR-1.7B (cached locally) and Paraformer-zh or FireRedASR2.
   - Score toneless pinyin error with pypinyin on both sides; raw CER is inflated by homophones (Whisper median 0.08 here).
   - Drop clips that both systems reject, with the threshold calibrated by the dataset's own control: same speaker, different line.
   - This mirrors Kurisu's two-ASR kana check, which removed 266–304 take/script mismatches [L].
7. **Tier B: re-audit, don't drop wholesale.**
   - `acoustic_disagrees` came from the weak VoxCeleb ECAPA. Re-score it with Qwen's encoder (centred) and drop only clips far from their own centroid.
   - `name_conflict` (347: mostly HM's maid-café persona swap, where 叶心's actor plays 迟菓) is a borderline style case. Leave it out of the first run.
   - Check `hot`.
8. **Speaker rows.**
   - Mean of ≤ 60 clean tier-A clips of 2.5–8 s per character, spread across games and scenes.
   - Accumulate in fp32 and rescale to the clips' typical norm, as `speaker_embedding()` does.
9. **Splits.** Keep the dataset's scene-held-out val split, and carve a test split the same way. Training clips never share a scene with test probes.

---

## 9. Training recipe (first run)

- **Init:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Trainer `kurisu_v2/pod/train_sft.py`, unchanged:
  - label alignment fixed;
  - `text_projection` applied;
  - speaker encoder frozen;
  - per-row speaker table;
  - per-row language prefix.
- **Rows:** 22, written at 3000–3021 with the names in §2. Language `chinese` with p_auto 0.25.
- **Optimiser:**
  - full fine-tune, fp32 master weights, bf16 autocast;
  - AdamW β (0.9, 0.95), lr 5e-6, 3% warm-up, cosine to 10%;
  - weight decay 0.01, effective batch 32 (v2/v3).
- **Exposure:**
  - ~20.5k clips ≈ 640 steps per data-epoch.
  - Run ~1.5 data-epochs (E1) or the equivalent step count under the capped √ sampler (E2).
  - Checkpoint every ~0.25 epoch.
  - v2's best was at 1 epoch on 10 h of one speaker [L]. With 22 speakers the right point is unknown, so select by the rule, not by loss.
- **No tags, no replay, no pseudo-data.**
- **Inference:**
  - default sampling (temperature 0.9, top-k 50, repetition penalty 1.05);
  - `language="Chinese"`;
  - `max_new_tokens` ≈ 2× the expected frames, with one regeneration on a cap hit.
  - Greedy decoding loops [L].

---

## 10. Evaluation and the selection rule (pre-register before training)

Detail in `notes/research_C_eval.md`.

**No published validation exists** for Mandarin ASR or MOS models on anime or galgame speech, or for speaker verification on Chinese VN voices [U]. So **each tool is chosen by how well it scores the cast's real held-out clips**, and generated audio is judged against that real-clip baseline, not against fixed numbers [I].

| Axis | Metric | Notes |
|---|---|---|
| **Intelligibility** | **Toneless pinyin error rate** (primary), raw CER (for comparison with papers), tone-aware pinyin error | Score with **one ASR family, confirm with another**: checkpoint rankings flip with the ASR family ([2607.08256](https://arxiv.org/abs/2607.08256)) [P]. Paraformer, SenseVoice and Fun-ASR are one family (FunAudioLLM); Qwen3-ASR, FireRedASR2 and Whisper are others [P]. Seed-TTS-eval's own zh scoring is character-level with no pinyin [P] |
| **Identity** | **Cast confusion matrix**: classify each output among the 22 centroids (built from real training clips) | Gate: cast-average accuracy ≥ (real held-out accuracy − 10 points) and **no new systematic off-diagonal pair**. Track per-clip margin (own minus best other). Per-character rates at ~20 probes are ±20 points, so gate on the cast average [I]. Precedent: nearest-centroid ID in a Japanese voice-actor paper ([2607.15694](https://arxiv.org/abs/2607.15694)), which also found that no single global threshold works [P] |
| **Similarity** | Centred cosine to the character's real held-out centroid, with an **independent** encoder | Run a bake-off on real clips: WavLM-SV (Seed-TTS-eval), CAM++ 200k-zh, ERes2NetV2 200k-zh (`iic/…_sv_zh-cn_16k-common`), litagin `anime_speaker_embedding` (char), KyaraEmbed. Pick the one that best separates the real cast [I]. VoxCeleb ECAPA reaches 13.8% EER on same-actor anime speech (KyaraBench) [P]. Don't judge with Qwen's encoder: it is the conditioning signal, so it is a biased judge [I] |
| **Stability** | Runaways (cap hits, or > 5× and > 6.5 s over the expected duration), **speaking rate** vs the character's real rate | Qwen #179 reports speech getting faster each epoch [P]. Kurisu's runaway definition [L] |
| **Short lines** | Identity and similarity on probes under 1.5 s | #343 [S] |
| **Emotion** (diagnostic) | emotion2vec+ large top label on output vs on the real take of the same held-out line | The CV3-Eval protocol [P]. Not validated on Mandarin acting [U]. Never use emotion-embedding cosine: it rewards acoustic mimicry ("The False Resonance") [P] |
| **Naturalness** (alarm only) | UTMOS/DNSMOS as regression alarms. For ranking: SpeechJudge-GRM pairwise, after a check against our own listening | On SpeechJudge's expressive subset (Genshin character voices included), UTMOS agrees with human preference 53.5% and DNSMOS 55.8% (chance 50%); audio LLMs 66–68% [P] |

**Selection rule (draft; freeze it before E1 is scored) [I]:**
1. Eligibility: runaway = 0 on the select set.
2. Primary: lowest cast-mean toneless pinyin error (ASR family 1).
3. Identity gate: as in the table.
4. Tie-break (within the E1 seed-to-seed spread): highest centred similarity.
5. Guards:
   - speaking rate within ±10% of real, per band;
   - ASR family 2 must agree with the ranking, otherwise the result is reported as a tie.
6. Validation loss is logged, not used.

**Listening test** [I, with power figures from `notes/research_C_eval.md`]:
- **Blind A/B between finalists**, same text and character: ~50 trials detect a 70/30 preference, ~90 a 65/35.
- **4-option "which character is this?"**, with the worst real-data confusions as distractors and real clips as controls: ~26 trials show above-chance identification if true accuracy is 50%.
- Include 文绉/朱特.
- With one rater (the Kurisu setup), treat results as indicative. More native raters who know the cast make it credible.

---

## 11. Experiment plan

Each run changes one thing relative to its parent.

| ID | Where | What | Decision |
|---|---|---|---|
| **E0** | local, no training | **Bars and tool calibration.** Per character, on ~20–30 held-out lines: (a) Qwen Base + locked mean x-vector, `language="Chinese"`; (b) Qwen Base ICL from the best `reference_clips.csv` clip; (c) CosyVoice3-0.5B zero-shot; (d) IndexTTS-2.5 zero-shot. Score real held-out clips with every tool; run the speaker-model and ASR bake-offs. **CosyVoice3 smoke test:** on the stock model, `add_zero_shot_spk` + `save_spkinfo`, then generate from the stored `spk_id` alone via `inference_sft`. Its `frontend_sft` feeds the flow only an embedding, where the flow normally also gets prompt tokens and mel [P code] | Fixes the select-set metrics and each character's real-clip ceiling. Shows how much a fine-tune must add |
| **E1** | H100 | Joint full FT, plain pooling, 2 training seeds | The default model and the noise floor |
| **E2** | H100 | E1 + √hours weighting as row repetition (≤ 3 copies per clip, mains never downsampled, more steps) | Adopt if small/tiny bands improve beyond the E1 spread with the mains within noise |
| **E3** | local eval | Per character: best of E1/E2 vs best E0 zero-shot | Roster: a row ships if it beats zero-shot on its own lines (pinyin error, identity, similarity, blind A/B for the small band) |
| **E4** | H100 | **CosyVoice3-0.5B** multi-speaker SFT (llm + flow), same data and splits, `spk2info` per character, `inference_sft` | Runs only as specified if the E0 smoke test produces normal speech from a stored embedding. **If it fails**, CosyVoice3 has no reference-free path. E4 then becomes "Qwen custom voice vs CosyVoice3 fine-tune driven by a stored prompt clip per character" (`zero_shot_spk_id`): still useful, but a different product, reported as such. Either way, switch only if it beats the Qwen pick on the same rule by more than the E1 spread, and report emotion-instruct retention separately |
| E5 | H100, contingent | Learnable zero-init residual on the confused rows | Only if E1/E2 show a systematic confusion pair |
| E6 | local, optional | LoRA r32 on the 5060 Ti (fits at micro-batch 12, 13.1 GiB [L]) vs E1 | Tells whether future retrains can run at home |
| E7 | H100, optional | + bit-bank lines as per-clip x-vector rows | Regulariser; adopt only if within noise or better |
| **Final** | — | Adopted changes, 2 seeds, rule on a confirm suite with fresh sampling seeds, blind listening | Ship |

**Order:**
1. E0 (local).
2. E1, E2 (one pod session).
3. E3.
4. E4.
5. Contingent runs.

E0 alone may change the plan: if CosyVoice3 or IndexTTS-2.5 zero-shot is already close to the real-clip ceilings for the mains, the case for E4 grows.

---

## 12. Cost and hardware

- **Local, RTX 5060 Ti 16 GB** [L]:
  - Qwen3-TTS Base, the codec tokenizer, Qwen3-ASR 0.6B/1.7B, the forced aligner and ECAPA/WavLM-SV are all cached; the `.venv-tts` env works (torch 2.13 + cu132, `qwen-tts` 0.1.1).
  - Covers data prep, codec extraction, E0 generation and all scoring.
  - Load models from the snapshot path: a transformers tokenizer check calls the Hub even when `HF_HUB_OFFLINE=1` is set.
- **Rented:** full FT peaked at 37 GiB on an H100 [L].
  - v2 ran 240 steps in 216 s. At ~640 steps per data-epoch, that is ~10 min per epoch [I].
  - E1 (2 seeds) + E2 + E4 is a few GPU-hours including evaluation. Evaluation, not training, is the bottleneck, as on Kurisu.

---

## 13. Risks and open questions

1. **Timbre bleed between similar voices.** Measured separable at the centroid level, but nothing published shows whether a codec LM keeps 22 close voices apart [U]. The identity gate and E5 exist for this.
2. **Short lines.** 14% of clips are under 1 s, and #343 reports wrong timbre on short utterances. Mitigations: the cap in §8 and dedicated probes.
3. **The tail.** At 3–8 min, a fine-tuned row may not beat zero-shot. That is a roster outcome, not a failure.
4. **No emotion control.** A fine-tuned Qwen Base delivers emotion the way the script implies (punctuation, wording, the character's habits) but takes no instruction (#25; a fine-tuned Base ignores instruct, Discussion #121) [P].
   - If explicit emotion control is a requirement, that weighs toward CosyVoice3 (E4) or IndexTTS-2.5 zero-shot.
5. **Possible shared voice actor (文绉/朱特)** [U]. The two rows may sound alike by nature.
6. **Rights.** The dataset README restricts use to personal use. The voices belong to the performers and the studio.
   - Publishing models, as was done for Kurisu, is a separate decision with the same consent and impersonation concerns.
   - The Kurisu model card's "Rights and responsible use" section is the template if it ever comes up.

**Questions for you:**
- **Roster:** train rows for all 22 and ship only those that beat zero-shot (recommended), or restrict to the 14 characters with ≥ 12 min?
- **Emotion:** do you need explicit emotion or style control, or is script-driven delivery enough? This decides how much weight E4 gets.
- **Destination:** a local model, a Space, or a public release? This changes the documentation work, not the training.

---

## 14. Decisions of 2026-10-29 and what they change

**Your decisions:**
- **Roster: the 14 characters with at least 12 min.** These are the 6 mains plus 朱特, 黄丽丽, 墨叔, 丁老师, 琳姐, 文绉, 徐阿姨 and 刺儿头.
  - The 4–8 min band is out, so E3 shrinks to a sanity check of each row against zero-shot.
  - 文绉/朱特 stay in and stay flagged.
- **Explicit emotion control is required.** "Explicit" means the control overrides what the text implies. The test is neutral text with a non-neutral control.
- **Public release:** HF model repo, HF Space, GitHub code repo, **and the audio dataset**.
  - You chose to publish the audio despite the dataset README's no-redistribution note.
  - Each card gets a rights and responsible-use section modelled on Kurisu's.
- **GPU budget:** up to ~$120 of rented time.
- **Emotion labelling stays local.**

**Consequences [I]:**
1. **The shipped voices must need no reference audio.** Cloning from stored clips would put game audio inside the Space as a hidden dependency. That conflicts with the release design even though the dataset is published separately, and it makes quality depend on which clip is shipped.
   - So IndexTTS-2.5 zero-shot stays a quality bar (E0), not a product.
   - CosyVoice3's reference-free path (`inference_sft`) becomes a gating question for E4.
2. **E4 (CosyVoice3-0.5B) becomes a required arm.** It already has instruct, so training would *reinforce* an existing capability. Qwen Base has none, so we would *create* one. No published result covers creating it on Qwen Base [U].
   - Both are trained, and emotion controllability enters the selection rule.
3. **Emotion labels come from the audio, not the text.** A token trained on labels inferred from the words learns nothing the text doesn't already say, so it cannot override the text.
   - **Signals:** emotion2vec+ large; SenseVoiceSmall's emotion tag (a separate model); and, for TRI, the sprite face code each line was spoken with.
   - An LLM reading the script is used at most as a tie-breaker, never as a required agreement, because requiring it would tie token to text.
4. **New selection constraints:**
   - The identity gate must hold on emotion-tagged prompts, not only untagged ones. Emotion control and timbre can trade off: IndexTTS2 needed a gradient-reversal layer to keep emotion timbre-free [P], and Kurisu's tag upsampling cost similarity [L].
   - The untagged default must stay script-driven and clean, checked the same way as Kurisu's device leak (A3′).

### 14.1 Data build B01 (done) [L]
`tricolour-tts/build/b01_build.py` produced **21,851 clips, 23.0 h** (24 kHz, 0.12/0.25 s silent pads, −23 LUFS per (series, character) group).

| Speaker | Minutes |
|---|---:|
| 墨小菊 | 386 |
| 文芷 | 261 |
| 骆衍 | 190 |
| 迟菓 | 139 |
| 迟耀 | 125 |
| 叶心 | 119 |
| 朱特 | 29 |
| 黄丽丽 | 28 |
| 墨叔 | 28 |
| 丁老师 | 18 |
| 琳姐 | 18 |
| 文绉 | 16 |
| 徐阿姨 | 13 |
| 刺儿头 | 13 |

Minutes include the pads.

| Drops and adjustments | Clips |
|---|---:|
| Tier C | 1,925 |
| Interjection-only clips under 1 s | 939 |
| `name_conflict` (persona-swap scenes) | 304 |
| Foreign or sung lines | 32 |
| Joint lines | 19 |
| Too long | 14 |
| Lexical clips under 1 s kept (the 5% cap) | 528 |
| Digit lines rewritten as spoken hanzi | 118 |

### 14.2 TRI sprite face codes (done) [L]
`b03_faces.py` replays each KAG script to the clip's line and records the speaker's active face.

- **Coverage:** 12,500 of 16,899 TRI clips (74%) have one; the rest are hidden or off-screen.
- **Code format:** f⟨brow⟩⟨eyes⟩⟨mouth⟩[h blush][t tears/sweat], with 482 distinct codes.
- **Brow usage:**

  | Brow | Lines |
  |---|---:|
  | 4 | 5,453 |
  | 1 | 3,746 |
  | 3 | 1,854 |
  | 2 | 1,429 |

- **Other flags:** blush 442, tears 307, sprite shake 466, trembling (ガクガク) 341.
- **Brow 4 is the most common**, so it is likely the cast's resting brow. Codes are named from emotion2vec agreement, not by eye.
- **SLS/HM have no expression data** (Utage `expression` column empty). 叶心, 黄丽丽 and 徐阿姨 therefore rely on the two audio models alone.

### 14.3 Transcript verification B02/B04 (done) [L]
Every kept clip was transcribed by two unrelated ASR families: Qwen3-ASR-1.7B (forced Chinese, no context prompt) and Paraformer-zh. Each transcript was compared with the script line as toneless pinyin, with interjections removed.

- **79% of clips** (17,353) match their line exactly on at least one system.
- **Median raw CER is 0.03–0.06**, which is homophone noise that pinyin scoring removes.
- **186 clips are dropped**, in two groups:
  - **109** are no closer to their line than the 1st percentile of a control: the same speaker's audio scored against a different line (T = 0.833).
  - **77** have a best error ≥ 0.4 on lines of ≥ 4 syllables.
- **What the drops are:** mostly laughs, coughs and chewing, where the two systems disagree. Also some truncated or substituted takes, and lines whose text includes unspoken narration (`小菊姐姐笑着坐回去：『…`).

### 14.4 emotion2vec+ on this cast is biased (B05) [L]
emotion2vec+ large top-1 over 21,851 clips:

| Label | Clips | Share |
|---|---:|---:|
| surprised | 6,645 | 30% |
| angry | 3,969 | |
| happy | 3,756 | |
| sad | 3,038 | |
| neutral | 2,358 | 11% |
| fearful | 1,042 | |

Two problems follow:
- **"surprised" is badly over-called and "neutral" under-called.** Expressive acting reads as surprise.
- **Labels track voice type.** The lower male voices come out "angry": 骆衍 has 1,174 angry against 126 neutral; 朱特 181 against 113; 墨叔 250 against 22. If these labels trained an emotion control, the control would partly become a speaker cue: the trap research D warned about.

It does agree with the artists where it should, on TRI face codes:

| Face code | emotion2vec agrees |
|---|---|
| brow 2 (frown) | angry 50% |
| tears | sad 48% |
| brow 3 (raised) | surprised 46% (base rate 30%) |

**Consequences:**
- emotion2vec labels are not used as training targets until D4 measures their accuracy.
- Any use must be **relative to each character's own baseline**, for example per-speaker normalised scores, not raw top-1.
- SenseVoice's emotion head is not an independent check, since it was trained on emotion2vec+ labels.

**D4 human calibration** is live at https://claude.ai/artifact/4vBgm54XyFpnF93w5zZgv4. It is blind (no model output shown, script line hidden until asked for) and covers 310 clips: 50 each for the core emotion classes, 30 fearful, 30 other, across all 14 characters. The answer key stays local in `tts_work/d4/key.json`.

### 14.5 Final manifest and splits (B06) [L]
`tts_work/manifest.jsonl` holds **21,665 clips, 22.8 h**.

| Split | Clips | How it is chosen |
|---|---:|---|
| train | 20,210 | the rest |
| val | 842 | the dataset's scene-held-out split, capped at max(12, 4%) per character so side characters keep their data |
| test | 613 | 3% of 40-line script blocks, at least 8 per character |

Whole-scene test selection was rejected. A TRI "scene" is a whole KAG file, and holding one out took up to 41% of 文绉's lines.

### 14.6 D4: human calibration of emotion2vec (done, 310 clips, one rater) [L]
The rater labelled each clip blind to the model's label, with the script line hidden unless asked for.

**Human labels:**

| Label | Clips |
|---|---:|
| neutral | 178 |
| other | 32 |
| angry | 26 |
| happy | 24 |
| fearful | 23 |
| surprised | 19 |
| sad | 8 |

Whisper was marked on 34 clips and strong on 50.

**emotion2vec precision** (share of its top-1 calls the human agreed with; each row is a stratum of 50, or 30 for fearful and other):

| emotion2vec label | Precision |
|---|---:|
| neutral | **0.94** |
| angry | 0.40 |
| fearful | 0.30 |
| other | 0.27 |
| happy | 0.26 |
| sad | **0.10** |
| surprised | **0.08** |

**Reading:**
- **emotion2vec over-reads emotion on acted VN speech.** Most of its non-neutral calls are neutral to a human: 35 of 50 "surprised", 28 of 50 "sad", 25 of 50 "happy".
- **Its "angry" is voice-type biased:** 7 of 21 correct on male voices, 13 of 29 on female.
- **Its "neutral" is trustworthy.**
- **TRI face codes are weak emotion evidence.** Brow 4 is 82% neutral, so it is the resting face. Brow 2 (frown) is angry in 9 of 26.

**Consequences:**
- **No emotion2vec label is a training target as is.** This holds for emotion tokens, per-emotion rows and D3 offsets.
- **Usable signals:**
  - its neutral call, as a clean-default filter;
  - its angry call, gated by face brow 2 or 5 on TRI;
  - human labels.
- **At this precision, emotion control cannot be bootstrapped from automatic labels.** It needs either a model that already follows emotion instructions (CustomVoice instruct, D2) or many more human labels.
- **D3 is retired as written.** Its per-emotion x-vector cells come from emotion2vec top-1, which would make them mostly neutral clips. Only 8 human "sad" labels exist, too few to fill them.
- **D2's emotion criterion (P(target) ≥ 0.40 by emotion2vec) is weaker than written,** given 0.08–0.40 precision on non-neutral classes. Amendment, made before D2 is scored: emotion following will also be judged by a blind human A/B on a subset; emotion2vec numbers are reported, not gated.

### 14.7 Channel-effect detection, first pass [L]
The user reported echo, phone/"digital" processing and noise in many clips while labelling.

**Phone / band-limited:** detected by each clip's 1/3-octave LTAS deviation from its own character's median. Both edges (≥ 4 kHz and ≤ 250 Hz) more than 10 dB below the norm flags 170 clips; at 15 dB, 89.
- The top hits are scripted phone calls: "喂？听得见吗？", "我就先挂了哦，长途漫游还是有点贵的", "这就给他联系".
- Raw band-energy thresholds failed: low male voices look band-limited.

**AudioSet tagging (CED-base) is useless for this.**
- Telephone, Echo and Reverberation scores stay near 0 on all clips. Phone candidates' Telephone median is 0.028.
- "Speech synthesizer" fires on 945 clips, but that tracks calm, flat voices (文绉 56%, 朱特 34%, 迟耀 23%), not an effect.

**Echo/reverb:** not yet detected. It needs a room-acoustics estimator; see research notes E/F.

### 14.8 D2 result: CustomVoice + our rows, untrained [L]
Full write-up: `D2_results.md`.

**Verdict as registered: FAIL**, on rule 2 (identity under instruct) and rule 4 (default neutral).

| Condition | Identity accuracy |
|---|---:|
| Base + vector | 0.983 |
| cv_none (no instruct) | **0.946**, above the real-clip ceiling of 0.919 |
| cv_happy | 0.800 |
| cv_angry | **0.713**: 文芷/墨小菊/叶心 drift to 琳姐, 朱特 to 墨叔 |
| cv_sad | 0.838 |

- **Emotion instructions shift emotion2vec's labels strongly** (+0.27 to +0.45 over cv_none), but that is report-only. The human A/B is still needed.
- **The judge is not confusing anger with 琳姐.** Real angry clips score 0.938 with no 琳姐 confusions.
- **Rule 4 was mis-calibrated** (emotion2vec's neutral recall is low; even Base scores only 0.575). The verdict stands as registered.
- **Implication:** the next experiment is a CustomVoice fine-tune with the 14 locked rows plus instruct replay. It tests whether trained rows hold identity under emotion instructions.

## 15. Production effects (echo, phone, reverb, noise): best practice and our plan

Full sources:
- `notes/research_E_detect_restore.md`: detection and restoration, with `fx_pilot.py` and its features;
- `notes/research_F_effects_training.md`: training-side handling, with `effects_probe.py` and its results.

The user heard echo, "digital"/phone processing and some noise while labelling. These are production effects baked into the voice tracks, and the script rarely marks them.

### 15.1 How much there is [L]
Flag counts below come from signal tests and have not been confirmed by ear.

**Phone / band-limited.** Two independent detectors agree:
- the speaker-relative LTAS test (§14.7): 89 clips at −15 dB, 170 at −10 dB;
- the agent's speech-only bandwidth test: 322 clips, which contains 167 of the 170.
- So the phone set is **about 90–320 clips (0.4–1.5%)**.
- It is mostly 骆衍, 迟耀, 黄丽丽, 墨小菊 and 叶心, in a few long call scenes. One TRI scene alone has 56 flagged lines.
- The telephone band is real: flagged clips sit 33.5 dB down below 300 Hz.

**Discrete echo.**
- 28 manifest clips in 8 scenes, mainly HM `yx_act4:2400` and SLS `Act5:5280` (徐阿姨, a constant 66 ms echo).
- Effects cluster by scene: after a flagged line, the next one is also flagged 82% of the time for echo and 58% for phone. Review whole scenes, never single clips.

**Reverb.**
- Not measurable on the exported clips. The export trimmed at −45 dBFS, cutting a median 0.51 s after speech, so it removed the tails.
- Needs a room estimator (Brouhaha C50, SIGMOS reverb) run on the **source** files, scored against each character's own distribution.

**Noise.**
- No working measure yet. The band-energy floor proxy confounds quiet speech with noise.
- Brouhaha SNR on the source files is the planned measure.

**Locked speaker vectors are contaminated.**
- 黄丽丽: 10 of her 60 reference clips are flagged (HM phone lines).
- 骆衍 4/60, 徐阿姨 2/60 (echo), and 0–1 for the rest.

### 15.2 What the evidence says [P unless tagged]
- **Large TTS pipelines treat these effects as defects to filter or clean:**
  - Emilia: DNSMOS ≥ 3.0;
  - FireRedTTS: DNSMOS > 3.3 and 99.5% roll-off > 7 kHz;
  - HiFiTTS-2: bandwidth ≥ 11–13 kHz.

  None keeps them as a labelled style. Moderate filtering beats aggressive filtering.
- **Effects in a speaker's data become that speaker's identity.** Hsu et al. 2019: with a plain speaker lookup, noisy speakers' output came out at 11.3 dB SNR against 18.2 dB for clean speakers. Augmentation alone recovered 17.0 dB, and an adversarial term 18.4 dB.
- **Filter or condition: the evidence is split.** Audiobox Aesthetics found prompting beat filtering; COT-TTS found filtering beat estimated-quality conditioning. Reading: exact labels work for intentional effects; drop unintentional noise.
- **Local probe on Qwen's own encoder and codec** (`effects_probe.json`; generic DSP effects, untrained models) [L]:

  | Finding | Measurement |
  |---|---|
  | An effect moves a character's vector further than the gap to its nearest neighbour | 1.2–2.7× the gap; phone EQ makes 10 of 14 vectors land nearer another character |
  | Contamination is linear | 25% effect clips move the vector 0.25 of the full shift; Base output moves 0.15–0.20 |
  | The speaker slot alone carries large-room reverb, with no training | 0.70 |
  | It does not carry a discrete echo | 0.06; the 12 Hz codec smears it |
  | For a hard telephone band, the codec is the ceiling | learned 0.75–0.79 vs DSP after generation 0.99 |
  | CustomVoice instruct produces no channel or room effect | ≈ 0 |
  | Effect deltas are shared across characters | cosine 0.84–0.96 |

- **Restoration.**
  - Sidon (MIT, 48 kHz) is the only open multilingual restorer that fits 16 GB, and it regenerates everything above 8 kHz.
  - Miipher-1 dropped speaker similarity to 0.585, against 0.945 for a discriminative cleaner.
  - Restoration has only been shown to help on *noisy found data*. No study covers acted speech or intentional effects.
  - At 0.4–1.5% per effect, there is nothing to gain.
- **Detectors.**
  - AudioSet taggers are useless here, confirmed locally (§14.7). Audio-LLM captioners are near chance at naming the effect (MRMAD: Qwen3-Omni 35–37% on 4 choices).
  - No pretrained telephone or codec classifier exists. Use speech-only bandwidth tests.
  - Discrete echo: a cepstral peak.
  - Reverb: Brouhaha C50 and SIGMOS.
- **Effects as user features.**
  - No published perceptual comparison of DSP vs learned effects in TTS exists.
  - Products expose them as tags or presets (MiniMax `lofi_telephone` / `spacious_echo`; Irodori 📞 / 📢 tokens).

### 15.3 Plan [I]
1. **Clean default voices.**
   - No clip flagged as phone, echo, reverb or noise trains a default row or enters its locked vector.
   - Flags propagate scene by scene: if ≥ 30% of one character's lines in a scene are flagged, review the whole scene.
   - Restore nothing.
2. **Rebuild the locked vectors before any training run.** The current `tts_work/xvec_rows.pt` must not become the training table. Use screened references (60 clean clips each), adding a check of each reference's projection on the shared effect deltas (≤ 0.05 per effect, gate G0). 黄丽丽's vector must change; 骆衍's and 徐阿姨's probably will.
3. **Effects on request, via DSP presets fitted to the games' own takes**, so all 14 characters get them, including those who never had a phone scene. Ranked by readiness:
   - **phone (ready):** EQ curve from the pooled phone takes (the Kurisu method). The flagged set exists and two detectors agree.
   - **echo (ready):** delay and gain from the cepstral peak, on the 8 flagged scenes.
   - **room (not ready):** convolution reverb from a fitted IR. There is no reverb detection yet: it needs Brouhaha/SIGMOS on the `_extracted/` source files, because both the export and B01 cut the tails. The IR-fitting tools (BUDDy, VINP, Rec-RIR) are 16 kHz English research code. Until then, any room preset is a generic IR, not fitted to the games.

   **Why DSP-first when Kurisu learned rows:**
   - For a hard telephone band, the codec caps a learned row at 0.75–0.79 against 0.99 for DSP after generation.
   - The x-vector does not carry a discrete echo at all (0.06).
   - Effect deltas are shared across characters (cosine 0.84–0.96), so DSP gives all 14 characters the effect, while only about 5 have any phone takes.
4. **Optional learned phone row** (character vector plus the shared phone delta, trained on ~70–100 DSP-augmented clean clips per character, ~4–6% of rows).
   - Ships only if it beats the DSP preset in a blind A/B, with the lower 95% bound above 50%.
   - The probe suggests a hard telephone band cannot beat DSP, so this is low priority.
5. **Gates, added to the pre-registration:**
   - **G0:** reference hygiene, as in step 2.
   - **G1:** default-voice leak ≤ 0.10 on both the spectral (A3′) and x-vector projections, plus the echo and bandwidth detectors on untagged output.
   - **G2:** preset fidelity ≥ 0.5 of the real takes' shift.
   - **G3:** the learned-vs-DSP blind A/B (only if step 4 runs).
   - **G4:** default CER and similarity unchanged.
6. **Validation before thresholds are fixed:**
   - A blind effect-labelling pass of 100 clips: 50 flagged, 20 neighbours of flagged clips, 30 random, with the script text hidden.
   - It gives each detector's precision and recall, after which the thresholds are refit.
   - Then run Brouhaha and SIGMOS on the source files for reverb and noise.

### 15.4 Human validation of the effect detectors (100 clips, one rater, source audio) [L]
**Human labels:**
- **87 of 100 clean.**
- Effects heard:

  | Effect | Clips |
  |---|---:|
  | reverb | 9 |
  | phone | 7 |
  | echo | 3 |
  | noise | 2 |

**Detector agreement:**

| Detector | Flagged in sample | Agreed with the rater | Verdict |
|---|---:|---|---|
| Band-limit | 15 | 5 phone, 7 any effect | usable as a screen, not a label |
| Echo (cepstral) | 10 | 2 echo, 3 any effect | only the strongest score (134) is unambiguous |
| Reverb tail | 12 | 0 | useless |
| Bass cut | 13 | 1 | useless |
| My stricter LTAS phone test | 4 (none at −15 dB in this sample) | 1 | — |

**Corpus estimate, weighted by stratum: about 4% of clips have an audible effect.**

| Effect | Share |
|---|---:|
| phone | ~1.2% |
| echo | ~0.2% |
| reverb | ~3.4% |

The reverb figure rests on a single hit among 30 random clips, so its 95% interval runs from well under 1% to over 15%.

**Consequences:**
- **Effects are a small minority.** The plan in §15.3 stands: clean defaults, DSP presets.
- **Locked-vector references:** exclude every clip any detector flags, plus whole flagged scenes. False positives cost nothing here, because each character has hundreds of clean candidates.
- **Training data:** drop the band-limit and echo flags (~430 clips, ~2%), knowing about half are clean. Most reverb stays undetected, so gate G1 (leak into untagged output, measured against Base with the same vector) is the real safeguard.
- **Reverb detection** (Brouhaha C50 on source files) can now be validated against the 9 human positives. It is optional unless G1 shows a reverb leak.

---

## 16. Could not verify

- Any SIM/CER curve against minutes of data below 30 min for an LLM/codec TTS.
- Whether a Qwen3-TTS joint fine-tune keeps similar voices distinct. No multi-speaker Qwen fine-tune publishes separation numbers.
- The origin of the third-party Qwen3-TTS Seed-TTS-eval numbers (1.22 / 77.0; hard 6.76 / 74.8).
- Whether CosyVoice3's `inference_sft` runs end-to-end from a stored embedding after a multi-speaker SFT, and whether instruct survives it. Also whether PR #1700 (the #1731 fix) is merged.
- Accuracy on Mandarin anime or galgame speech for any ASR, speaker-verification, emotion or MOS model.
- Whether 文绉 and 朱特 share a voice actor.
- Whether community IndexTTS-2 trainers (instavar, SrQingChen) work.

---

## Sources (selected; full lists in `notes/`)

- **Qwen3-TTS:**
  - https://github.com/QwenLM/Qwen3-TTS (issues #14, #25, #179, #213, #343; PRs #278, #384)
  - Fine-tuning README: https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
  - Report: https://arxiv.org/abs/2601.15621
  - TaoLive: https://huggingface.co/TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice
  - FelysNeko: https://huggingface.co/FelysNeko/Qwen3-TTS-12Hz-1.7B-Delta-me13
  - vspeech fork: https://github.com/vspeech/Qwen3-TTS-Train
  - Baseten FT vs zero-shot: https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/
- **CosyVoice:**
  - Repo: https://github.com/FunAudioLLM/CosyVoice (issues #1703, #1731)
  - CosyVoice 2: https://arxiv.org/abs/2412.10117
  - CosyVoice 3: https://arxiv.org/abs/2505.17589
  - CV3-Eval: https://github.com/FunAudioLLM/CV3-Eval
- **Other models:**
  - IndexTTS: https://github.com/index-tts/index-tts (#518)
  - IndexTTS-2.5: https://arxiv.org/abs/2601.03888
  - dots.tts: https://github.com/rednote-hilab/dots.tts
  - FireRedTTS3: https://github.com/FireRedTeam/FireRedTTS3
  - LongCat: https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B
  - VoxCPM: https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
- **Multi-speaker and data:**
  - Luong 2019: https://arxiv.org/abs/1904.00771
  - Seed-TTS: https://arxiv.org/abs/2406.02430
  - NeuTTS LoRA: https://arxiv.org/abs/2603.10904
  - XTTS-v2 data-size study: https://www.isca-archive.org/interspeech_2025/szekely25_interspeech.pdf
  - Cooldown sampling: https://arxiv.org/abs/2410.04579
  - CSP-FT: https://arxiv.org/abs/2501.14273
- **Evaluation:**
  - Seed-TTS-eval: https://github.com/BytedanceSpeech/seed-tts-eval
  - ASR-family confound: https://arxiv.org/abs/2607.08256
  - KyaraBench: https://arxiv.org/abs/2610.06013
  - 3D-Speaker: https://github.com/modelscope/3D-Speaker
  - SpeechJudge: https://arxiv.org/abs/2511.07931
  - The False Resonance: https://arxiv.org/abs/2604.26347
  - Listening-test practice: https://arxiv.org/abs/2503.03250
  - Voice-actor nearest-centroid: https://arxiv.org/abs/2607.15694
- **Local:**
  - `E:\STEINS;GATEVoice\kurisu_v2\research\{sota_models,finetune_methods,final_report,e0_results,v3_results}*.md`
  - `kurisu_v2/pod/train_sft.py`, `kurisu_v2/v3/replay_notes.md`
  - `F:\Tricolour\voice_dataset_game\{README.md,metadata.jsonl,speakers.csv}`
