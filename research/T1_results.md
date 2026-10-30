# T1 results (2026-10-30)

Pre-registration: `T1_preregistration.md`, including Amendment 1.
Scores: `tts_work/t1_eval/scores.json`, `scores.md`.

- **Pod:** one Secure H100 80GB, $3.99/h.
- **Training:** B1 11 min 52 s, C1 12 min 46 s, C2 13 min 4 s; peak VRAM about 35 GB.
- **Evaluation:** 26,072 generated clips.
- **Checkpoints downloaded** to `E:\Tricolour_t1_ckpts`: B1 465/620/775, C1 667/834, C2 685/856.

## All candidates

| Checkpoint | CER | ID | SIM | LEAK | ID_instr | Emotion shift (instr_effect) | Runaway | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| real S1 recordings | 0.0305 | 0.967 | — | 0 | — | — | — | reference |
| B0 (untrained Base + vectors) | 0.0051 | 0.911 | 0.537 | 0 | — | — | 0 | reference |
| C0 (untrained CustomVoice + rows) | 0.0161 | 0.787 | 0.438 | 0 | 0.739 | 0.362 | 1.1% | reference |
| B1 step 310 (0.5 ep) | 0.0078 | 0.927 | 0.600 | 0 | — | — | 0 | ✓ |
| **B1 step 465 (0.75 ep)** | 0.0090 | 0.947 | **0.608** | 0 | — | — | 0 | ✓ **B1 winner** |
| B1 step 620 (1.0 ep) | 0.0101 | 0.955 | 0.601 | 0 | — | — | 0.2% | ✓ |
| B1 step 775 (1.25 ep) | 0.0107 | 0.951 | 0.604 | 0 | — | — | 0 | ✓ |
| C1 step 334 | 0.0153 | 0.909 | 0.578 | 0 | 0.931 | 0.173 | 0 | ✗ G-emo |
| C1 step 500 | 0.0111 | 0.919 | 0.583 | 0 | 0.922 | 0.193 | 0 | ✓ |
| C1 step 667 | 0.0108 | 0.920 | 0.592 | 0 | 0.935 | 0.179 | 0 | ✗ G-emo |
| **C1 step 834** | **0.0096** | 0.936 | 0.598 | 0 | **0.942** | 0.211 | 0 | ✓ **selected** |
| C2 step 342 | 0.0099 | 0.900 | 0.571 | 0 | 0.745 | 0.278 | 0 | ✓ |
| C2 step 514 | 0.0144 | 0.897 | 0.567 | 0 | 0.413 | 0.413 | 0 | ✗ G-id |
| C2 step 685 | 0.0111 | 0.928 | 0.592 | 0 | 0.532 | 0.327 | 0 | ✓ |
| C2 step 856 | 0.0100 | 0.929 | 0.598 | 0 | 0.483 | 0.406 | 0 | ✓ |

**Gates, as registered:**

| Gate | Threshold |
|---|---|
| G-cer | CER ≤ 0.0251 |
| G-id | ID ≥ 0.90 |
| G-leak | LEAK ≤ 0.01 |
| G-run | runaway ≤ 0.5% |
| G-emo | instr_effect ≥ 0.181, i.e. half of C0's |

## Selection (as registered)

1. **B1 winner: step 465.** All four CERs fall within 0.005 of the best, and 465 has the highest SIM.
2. **CV winner: C1 step 834.** It has the highest ID_instr among eligible C checkpoints (0.942), tied within 0.02 with C1 step 500 and winning that tie on lower CER.
3. **C1 step 834 is non-inferior to B1 step 465:** CER 0.0096 ≤ 0.0190, and ID 0.936 ≥ 0.927. **Decision: C1 step 834.**
4. **No emotion-following claim yet.** It still needs the blind human A/B (16 pairs per emotion).

## What the numbers say

- **Fine-tuning fixed identity.** It went from 0.911 (B0) and 0.787 (C0) to 0.93–0.955, against a real-recording ceiling of 0.967. Reading accuracy stays far better than the ASR achieves on the real recordings (CER about 0.01 vs 0.031).
- **C1 fixed D2's failure:** identity under emotion instructions went from 0.739 to 0.942, and under "angry" from 0.69 to 0.955. Characters no longer collapse into 琳姐.
- **But C1 kept only about half of the instruction effect.** emotion2vec's target rate on S3:

  | Instruction | C0 | C1 step 834 |
  |---|---:|---:|
  | happy | 0.445 | 0.417 |
  | angry | 0.467 | 0.548 |
  | sad | 0.507 | **0.157** |

  Sad is largely lost. The no-instruction S2 sad rate is about 0.005, so the real data almost never sounds "sad" to emotion2vec, and replay alone did not keep it. These are emotion2vec numbers, report-only.
- **C2 trades identity for emotion.** The character-voiced pseudo-data restored the instruction effect, including sad (0.53–0.68), but outputs under instructions drift to the same generic voices as D2: 琳姐 for female characters, 墨叔/迟耀 for male (ID_instr 0.41–0.75).
  - The P data passed a Qwen-encoder identity filter, but CAM++ hears it as off-character. The filter was too lenient for emotional takes: it let CustomVoice's generic emotional voice through, and the model learned it.
  - The selection rule correctly prefers C1.
- **Weakest characters in every arm: 黄丽丽 and 徐阿姨.**
  - 黄丽丽's identity is 0.44–0.78. She drifts to 文芷/琳姐.
  - 徐阿姨's is 0.68–0.87.
  - They are the two characters recorded in only one of the Utage games (HM and SLS respectively), with 28 and 13 min of data. Untrained CustomVoice was already worst on them (0.17 and 0.34).
- **The overlap between instructed and real emotional speech is limited.** Real angry clips score identity 0.938, real other-emotion clips 0.873 (D4 labels). C2's 0.43–0.53 is a genuine identity loss, not normal emotional variation.

## Pending
- **C1b, a second seed of C1:** seed-noise check, in particular whether C1's narrow G-emo pass (0.211 against 0.181) is robust.
- **Human A/B on C1 step 834:** does each instruction audibly change the delivery?
- **ASR-family confirmation of the finalists' CER** with Qwen3-ASR (prereg metric note).

## Human emotion A/B on C1 step 834 (one rater, blind, 2026-10-30)
Each pair is the same character and line, instructed vs not, in random order, forced choice ("which sounds more <emotion>?"). Data: `tts_work/ab/ab_results.json`. Key: `tts_work/ab/key.json`, local only.

| Emotion | Instructed chosen | One-sided p | Two-sided 95% CI lower bound | Verdict |
|---|---:|---:|---:|---|
| angry (生气) | 14/16 | 0.002 | 0.617 | **followed** |
| sad (伤心) | 16/16 | < 0.0001 | 0.794 | **followed** |
| happy (开心) | 12/16 | 0.038 | 0.476 | **borderline** |
| all | 42/48 (0.88) | — | 0.748 | — |

**How the rule was read.**
- The pre-registration says "exact 95% binomial lower bound above 50%". The A/B build script, written before any listening, made that concrete as "≥ 12 of 16", which is the one-sided reading.
- Under that reading all three emotions pass.
- Under the stricter two-sided reading, happy (12/16) does not.
- Both readings are reported. The claim that ships: angry and sad clearly follow instructions; happy follows weakly.

**Side bias:** the rater chose A 30 times and B 18; the instructed takes sat on A 28 times and B 20.

**Per character:** every character's instructed take won at least half its pairs. 文绉 was weakest at 2/4; 黄丽丽, 徐阿姨 and 朱特 were each 3/4.

**emotion2vec was wrong about "sad".**
- It scored C1's sad-instruction effect almost gone (target rate 0.157 vs 0.507 untrained).
- The listener picked the sad take in 16/16 pairs.
- This matches D4, where emotion2vec's "sad" label was right only 10% of the time.
- **Consequence for the selection rule:** the G-emo gate (instr_effect from emotion2vec shifts) is unreliable exactly where it matters. It rejected C1 steps 334 and 667 on a margin of 0.002–0.008; human ears would probably not have. Future rounds should gate emotion on a human A/B or a classifier calibrated on this cast, not on emotion2vec.

## Seed replication (C1b = C1 with seed 20261031; same data and recipe)

| Checkpoint | CER | ID | SIM | ID_instr | instr_effect | Eligible |
|---|---:|---:|---:|---:|---:|---|
| C1b step 334 | 0.0123 | 0.889 | 0.571 | 0.915 | 0.156 | ✗ |
| C1b step 500 | 0.0087 | 0.912 | 0.588 | 0.934 | 0.176 | ✗ G-emo |
| C1b step 667 | 0.0097 | 0.927 | 0.597 | 0.941 | 0.187 | ✓ |
| C1b step 834 | 0.0111 | 0.954 | 0.599 | 0.951 | 0.198 | ✓ |

**Seed spread at step 834:**

| Metric | Spread |
|---|---:|
| CER | 0.0015 |
| ID | 0.018 |
| ID_instr | 0.009 |
| instr_effect | 0.013 |

**Reading:**
- C1's result replicates.
- Which checkpoints pass G-emo flips between seeds (C1 passes 500 and 834; C1b passes 667 and 834). G-emo near its threshold is seed noise, consistent with the A/B evidence against emotion2vec.
- The selection stays C1 step 834, as registered: the second seed is a robustness check, not a new candidate.
- C1b's checkpoints were not downloaded.

## Cost
- One Secure H100 pod, 3.04 h at $3.99/h: **$12.13**.
- Local GPU work (codes, R/P filtering, all scoring): free.
