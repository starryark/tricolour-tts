# D2: Qwen3-TTS-CustomVoice with our speaker vectors (pre-registered 2026-10-30, before any D2 generation)

**Question.** Qwen3-TTS-12Hz-1.7B-CustomVoice already follows emotion instructions. If our locked Base-encoder x-vectors are written into its spare speaker rows, does it:
- (1) keep each character's identity;
- (2) follow `instruct` emotions on emotionally neutral text?

If yes, route (a′) becomes the default Qwen arm, and training would only reinforce instruct rather than teach it from nothing. See `notes/research_D_emotion.md` §D2.

## Setup (fixed now)

**Characters.** Eight: the 6 mains (墨小菊, 文芷, 骆衍, 迟菓, 迟耀, 叶心) plus 文绉 and 朱特 (the close pair).

**Speaker vectors.**
- One per character, computed with `train_sft.speaker_embedding()`: the mean Base speaker-encoder vector, rescaled to the clips' typical norm.
- Inputs: ≤ 60 tier-A B01 clips per character, with core duration 2.5–8 s, not in the dataset's val split. Seeded selection, spread across the games.
- **CustomVoice:** the vectors are written into codec-embedding rows **3020–3033**. Rows 3010, 3061, 3065 and 3066 hold presets and are untouched.
- **Base:** the same vectors, passed as `x_vector_only` voice-clone prompts.

**Text.** The 30 novel, emotionally neutral sentences in `tricolour-tts/eval/probes_neutral_zh.txt`. None is from the games.

**Conditions** (`language="Chinese"`):

| ID | Model | Instruct |
|---|---|---|
| `base` | Base + vector | none (Base has no instruct) |
| `cv_none` | CustomVoice + row | none |
| `cv_happy` | CustomVoice + row | `用开心的语气说` |
| `cv_angry` | CustomVoice + row | `用生气的语气说` |
| `cv_sad` | CustomVoice + row | `用伤心的语气说` |

**Decoding.** Upstream defaults: temperature 0.9, top-k 50, repetition penalty 1.05, non-streaming, `max_new_tokens` 400. One sample per (character, text, condition). All 30 texts for one (character, condition) are generated in one batch, seeded with `torch.manual_seed(1000 + character index)`. The seed is shared across conditions.

*Clerical amendment, made before any generation:* the first draft said "seed per text index". Batched generation cannot do that, because a text's sampling noise depends on the rest of its batch.

## Judges (fixed now; independent of the Qwen encoder)

**Identity: CAM++ zh-cn 200k** (`iic/speech_campplus_sv_zh-cn_16k-common`).
- Centroids: 14 cast centroids, each from ≤ 200 real B01 clips of that character, excluding the val split.
- Cosine is computed after subtracting the mean of the 14 centroids.
- A clip is correct if its nearest centroid is its own character.
- **Real ceiling:** the same classification of real val-split clips (core 1.5–6 s, to match the length of generated probes).

**Emotion: emotion2vec+ large**, top-1 of its 9 classes. **Its accuracy on this cast is unknown until D4**, so every emotion number is provisional.

**Runaways.** A clip that hits the 400-frame cap, or lasts more than 5× its expected duration.

## Pass rule (a′ viable)

All of these must hold:
1. **Identity, no instruct.** Cast-mean identity accuracy of `cv_none` ≥ `base` − 10 points.
2. **Identity under instruct.** Each of `cv_happy`/`cv_angry`/`cv_sad` ≥ `base` − 15 points.
3. **Emotion follows the instruct.** The mean over the three emotion conditions of P(emotion2vec top-1 = the requested emotion) ≥ **0.40**. That is the level of published fine-tune-only results.
4. **The default stays neutral.** `cv_none` P(top-1 = neutral) ≥ 0.70.
5. **Stability.** Runaways ≤ 1% in every condition.

**Reporting.**
- Per-character numbers for 文绉/朱特 and every character are reported. The rule uses cast means.
- If (1)–(2) pass and (3) fails, (a′) still beats (a) as the identity base, and the emotion gap becomes the training target.
- If (1) fails, (a′) is out, and D3 (x-vector emotion offsets on Base) decides the Qwen route.

## Amendment 1 (2026-10-30, after the D4 calibration, before any D2 output was scored)

**Why.** D4 measured emotion2vec+ large against 310 human labels on this cast. Its precision on non-neutral classes:

| Class | Precision |
|---|---:|
| angry | 0.40 |
| happy | 0.26 |
| sad | 0.10 |
| surprised | 0.08 |

Its "neutral" is reliable (0.94). Rule 3, which reads emotion following off emotion2vec top-1, therefore cannot measure what it was meant to measure.

**Changes:**
- **Rule 3 is report-only.** P(emotion2vec top-1 = target) is still computed and reported for each instruct condition, but it no longer decides pass or fail.
- **Rule 4 stays a gate.** The default stays neutral, and it relies on emotion2vec's neutral call, which D4 found reliable.
- **Emotion following is judged by a blind human A/B instead.** Per instruct condition, 16 pairs: the `cv_<emotion>` output against the `cv_none` output of the same character and text, in randomised order. The rater picks the one that sounds more <emotion>.
  - Following counts as shown if the instruct side wins with the lower bound of the exact 95% binomial interval above 50%.
  - This is reported per emotion and is not part of the automatic pass rule.
- **Rules 1, 2, 4 and 5 are unchanged.**
- **Locked vectors (known limit, noted before scoring).** The vectors were computed before effect screening. Among the 8 D2 characters, 骆衍 has 4 of 60 reference clips flagged as band-limited, and 墨小菊, 迟耀 and 叶心 have 1 each. D2 compares conditions that share the same vector, so its question is unaffected. These vectors are rebuilt from screened references before any training run (`voice_model_approach_2026-10.md` §15.3).
