# D2 results (2026-10-30): CustomVoice + our speaker rows, no training

Pre-registration: `D2_preregistration.md`, including Amendment 1.
Generations: 8 characters × 30 novel neutral sentences × 5 conditions = 1,200 clips, 0 runaways.
Judges: CAM++ zh-cn 200k for identity (centred nearest-centroid among the 14 cast centroids); emotion2vec+ large for emotion (report-only).
Scores: `tts_work/d2/summary.json`.

## Verdict as registered: FAIL (rules 2 and 4)

| Rule | Threshold | Result | |
|---|---|---|---|
| 1. Identity, no instruct | cv_none ≥ base − 0.10 = 0.883 | 0.946 | pass |
| 2. Identity under instruct | each ≥ base − 0.15 = 0.833 | happy 0.800, angry **0.713**, sad 0.838 | **fail** |
| 4. Default stays neutral | cv_none P(e2v neutral) ≥ 0.70 | **0.412** | **fail** |
| 5. Runaways | ≤ 1% per condition | 0 | pass |
| 3. Emotion (report-only, Amendment 1) | — | mean P(e2v = target) 0.45 | — |

Reference points:
- Real held-out clips: 0.919 identity accuracy.
- Base + the same vector: 0.983.

## What the numbers say

**1. CustomVoice keeps identity without an instruction.** cv_none scores 0.946, above the real-clip ceiling. The weak cases are 朱特 (0.833, confused with 墨叔) and 叶心 (0.867, confused with 琳姐).

**2. Emotion instructions move the delivery strongly.** emotion2vec share of the requested label (report-only; D4 precision limits apply):

| Instruct | cv_none | Instructed | Shift |
|---|---:|---:|---:|
| happy | 0.21 | 0.48 | +0.27 |
| angry | 0.09 | 0.38 | +0.29 |
| sad | 0.03 | 0.48 | +0.45 |

The shifts are large against the same character and text without an instruction. They are consistent with real instruction following, but D4 showed emotion2vec misreads this cast, so the human A/B in Amendment 1 is still required before any claim.

**3. Happy and angry instructions pull some characters toward a generic voice.**
- Under "angry", 文芷, 墨小菊 and 叶心 drift to 琳姐 (13/12/12 of 30), and 朱特 drifts to 墨叔.
- Under "happy", 叶心 drifts to 琳姐 (16/30).
- The judge is not the cause. On real clips (D4 human labels), angry lines are identified correctly 0.938 of the time, with no 琳姐 confusions. Real emotional lines in general score 0.873 against 0.982 for neutral ones, so emotional output should be expected near 0.87–0.94, not 0.98.
- 0.713 (angry) is well below that.

**4. Rule 4 was mis-calibrated.** I amended it on emotion2vec's neutral *precision* (0.94), but its neutral *recall* on this cast is low: it called only 47 of the 178 human-neutral D4 clips neutral. Base with no instruction scores just 0.575 on the same measure, so the 0.70 threshold was unreachable even for the comparison anchor.
- The verdict stands as registered.
- A secondary reading, not a rescue: cv_none (0.412) is measurably more "emotional" than Base (0.575) on identical texts and vectors, so CustomVoice adds some expressiveness by default.

## What follows

- **Untrained CustomVoice with our rows is not shippable.** It meets the identity bar without instructions and fails it under happy/angry instructions.
- **This is the gap a training run has to close.** Fine-tune CustomVoice with the 14 locked rows on the cleaned data, replaying instruct examples so instruct stays alive (the CosyVoice3 §2.6 recipe). The question it answers: do trained-in rows hold identity under emotion instructions?
- **The emotion human A/B (Amendment 1)** can run on these same D2 clips.
