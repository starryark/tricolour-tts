# T1 pre-launch review (independent, 2026-10-30)

Scope: everything listed in the brief. No pod was launched and no file under review was modified.
Local checks were run from the scratchpad with the Kurisu venv (torch 2.13, transformers 4.57.3, qwen-tts 0.1.1).

**Verdict.** The trainer changes (a)–(e) are correct, and I could not break them. The problems are elsewhere:
- the run plan will very likely hit the 5 h hard stop before evaluation finishes, and a kill loses everything;
- the cost-safety code in `launch.py` is not a real hard stop;
- the local scoring has a centroid leak.

---

## BLOCKER

### B-1. The run plan will likely hit the 5 h kill, and a kill loses everything

**Files:** `pod/run_t1.sh:44-58`, `pod/launch.py:99`, prereg "Budget".

**Training is quick.** Kurisu's H100 logs (`kurisu_v2/artifacts/logs/queue*.log`) show this trainer at **36–46 clips/s** with micro 16 × 2. That is about 12 min of pure training per arm (B1 24.8k clips, C1 26.7k, C2 27.9k, at 1.25 epochs). Add four checkpoint writes of 4.3 GB each, the ~14 GB `state.pt`, the validation passes and the model load, and it comes to about 20 min per arm.

**Evaluation is the bulk of the time, and nobody has timed it.**
- `gen_eval` runs **672 `generate_custom_voice` batches** (B1 4×28, C1 and C2 4×70 each), about 17.4k clips.
- The S3 suite alone is 1,260 clips per CV checkpoint.
- Each batch runs until its longest row finishes, and every AR frame also runs 15 sub-talker steps.
- On the **local 16 GB Windows card**, d2 (`tts_work/d2`, 30-clip batches, same `KW`) took **19–163 s per batch, median about 55 s**. The H100 figure is unknown in both directions.
- At 15 s per batch on the H100 that is 2.8 h; at 30 s it is 5.6 h.
- The prereg's "≈3 h total" is therefore unsupported.

**The failure scenario:**
1. All three arms train first (`run_t1.sh:44-51`); only then does evaluation start (`:53-58`).
2. Within an arm, checkpoints are globbed in order 310 → 775, so the 0.5-epoch checkpoint is evaluated *first*. The prereg declares it the first to drop.
3. The watchdog fires mid-C1 or mid-C2.
4. The pod has `volumeInGb: 0` and the outputs are never downloaded, so the termination deletes every checkpoint and all evaluation audio. You pay for about 5 h and get nothing.

**Fixes:**
- Time one 32-clip S3 batch on the pod right after bootstrap, before training, and size the lease from that.
- Restructure to train → evaluate per arm.
- Evaluate in priority order: 1.25, 1.0, 0.75, then 0.5.
- `rsync` each arm's `eval/` home as soon as it finishes.
- If needed, cut S3 to 2 checkpoints, or the batch count.

### B-2. The watchdog is not a hard stop

**File:** `pod/launch.py:144-156`, plus `gql` at `:41-48`.

**Problems:**
- **It tries once.** `terminate()` failing with `SystemExit` prints "terminate failed" and `return`s, so the pod keeps billing.
- **Network errors crash it.** `gql` converts only `HTTPError`. A `URLError`, `TimeoutError`, `ConnectionResetError` or `JSONDecodeError` raised at the deadline crashes the watchdog with a traceback.
- **It never confirms.** Nothing checks that the pod is actually gone.

**What I tested.** A child spawned with the exact `creationflags` survived the tool call that spawned it for its full 150 s, from both the Bash and the PowerShell tool. **Not tested:** survival when the Claude Code session exits, or across Windows sleep. Sleep is a realistic failure: billing continues until the machine wakes.

**Fixes:**
- At the deadline, loop on `try: terminate() except Exception` with backoff until `pod_info()` is `None` or the status is terminated.
- Add a second, pod-side layer: in `bootstrap.sh`, `nohup sh -c 'sleep 18000; runpodctl remove pod $RUNPOD_POD_ID' &`. I believe RunPod images ship `runpodctl` with a pod-scoped key; verify this on the pod.
- A plus worth keeping: the watchdog re-reads `lease.json` every loop, so the deadline can be extended live.

### B-3. A failed create can leave an orphaned, billed pod with no watchdog

**File:** `pod/launch.py:95-112`.

**What goes wrong:**
- **The mutation can succeed even when the call fails.** `podFindAndDeployOnDemand` can create the pod server-side while the client sees a 60 s socket timeout (`URLError`/`TimeoutError`).
- **The handler misses that error.** `except SystemExit` does not catch it, so the lease stays `"creating"` with **no `pod_id`**.
- **No watchdog starts.** The watchdog is spawned only after success (`:109-112`).
- **`terminate` cannot find the pod.** It prints "no pod in the lease".
- **The "unknown" status does not block a second create.** On an `HTTPError` the status becomes `"unknown"`, which `create()` does not refuse (`:84`), so a second `create` can start a second pod.

**Fix:**
- Spawn the watchdog *before* the mutation.
- Catch `Exception`.
- On any failure, find the pod by its unique `name` (`myself { pods { id name } }`), then record and terminate it.
- Refuse `create` while the status is `"unknown"`.
- Minor: the lease is written before the `.pub` key is read (`:95-97`), so a missing key leaves a stale `"creating"` lease.

---

## SHOULD-FIX

### S-1. The price cap checks only the quote

**File:** `launch.py:90-92, 107`.

The cap is checked against `lowestPrice.uninterruptablePrice` and never against the `costPerHr` the deploy actually returned. Add a post-create check: if `pod["costPerHr"] > MAX_USD_PER_HOUR`, terminate immediately.

### S-2. The equivalence gate does not gate `run_t1.sh`, and does not touch the generation path

**Files:** `bootstrap.sh:35-36`, `run_t1.sh`.
- **No marker check.** `run_t1.sh` never checks that bootstrap passed. `bash bootstrap.sh; bash run_t1.sh`, or a manual rerun after a failure, trains anyway. Have bootstrap write `$W/GATE_PASS` and refuse to start without it.
- **Generation is never exercised before the run.** The test captures prefill embeddings only. Decoding (`speech_tokenizer.decode`, onnxruntime/sox/librosa), the sampling loop and FLAC writing are first exercised by `gen_eval`, hours in. Add one `generate_custom_voice` round trip plus a decode to bootstrap.
- **Torch differs from local.** The pod runs torch 2.8 (from the image); the local tests ran on 2.13. The bootstrap comment claiming the same versions is true only for transformers and qwen-tts.

### S-3. Checkpoints are never kept

The prereg terminates the pod after the evaluation audio is downloaded. With no volume and no checkpoint download, the selected checkpoint no longer exists when selection happens locally. Shipping would then need a retrain, which is not bit-reproducible. Decide explicitly: download about 52 GB, download only the 1.0/1.25 checkpoints, or accept retraining.

### S-4. Evaluation leakage in local scoring

This can be fixed before scoring, at no pod cost.
- **Test clips sit inside the identity centroids.** `eval/judge.py:82-100` builds the CAM++ centroids from every kept B01 clip not in the dataset's `val.csv`. That pool includes the b06 **test** split. **69 of the 191 S1 real clips are inside the centroids**, and so are 110 test clips overall and 71 effect-dropped clips.
- **The ceiling is defined differently from the prereg.** The prereg's "real test clips (ID ceiling…)" would therefore be self-included. Meanwhile, `Cast.ceiling()` (`:130-135`) actually uses *val* clips.
- **Fix:** exclude manifest test ids, and dropped ids, from the centroids, and make the ceiling definition match the prereg.
- **Three S1 lines also occur in train for the same speaker** (punctuation-insensitive): chiguo「忘掉不也是逃避吗？」, chiguo「你知道吗？」, yexin「最好的朋友？…」. `make_suites.py:24-26` has no text-overlap check. Drop these lines from S1.
- The effect is equal across arms, but it favours trained models against the untrained baselines in G-cer and G-id.

### S-5. Selection effects that favour C2 or instruction-ignoring outputs

**Files:** `rp_score.py:1-11, 83-90`; prereg Selection 2.
- **The P identity filter keeps clips that ignored the instruction.** It keeps clips whose Qwen-encoder vector stays nearest the character. Strongly emotional takes move speaker embeddings the most, so the filter preferentially keeps them out, and P teaches "row + instruction → near-neutral". Qwen's encoder and CAM++ are different models, but both are speaker-verification embedders and correlated. So the claim "cannot simply teach to the test" overstates the independence.
- **The selection rule then rewards that.** Selection 2 picks the **highest ID_instr**, which rewards ignoring instructions, while emotion is "reported only".
- **Fix:** add a guard to the selection, for example an emotion2vec target-emotion rate on S3 that must not fall below C1 or the untrained CustomVoice, or S3-vs-S2 twin divergence.
- **The same ASR filters and judges.** R/P are filtered with Paraformer (`rp_score.py:83`), the same ASR that scores CER. Keep the Qwen3-ASR confirmation step for finalists.

### S-6. R/P handling in `run_t1.sh`

**File:** `run_t1.sh:15-21`.
- **One missing file stops every arm.** A missing `R_*`/`P_*_with_codes.jsonl` aborts even `run_t1.sh B1` under `set -e`.
- **Empty files fail silently.** If they exist but are empty (for example, `select` dropped everything), C1 trains on B1 data with no error.
- **Fix:** assert `wc -l` against the expected counts (train 19,829, val 826, R about 1.5k, P about 1k), and build only the arms requested.
- Also `mkdir -p $W/logs`: if `tee` cannot open its file, `pipefail` aborts the script after a finished training run, before `TRAIN_DONE` is written, and a rerun then `rm -rf`s that arm.
- Note: `tts_work/rp/` does not exist yet, so R/P, `rp_score` and b10 on R/P are still to be run.

---

## NOTE: verified correct, or minor

- **Collate (a): verified locally with a stub processor and config.**
  - In a mixed batch, each row equals its single-row collate, with right padding only.
  - An instruction row equals the plain row shifted by `o` for input_ids, every mask, labels, codec_ids and attention.
  - The instruction and role positions carry text only and no labels.
  - The sub-talker reads `hidden[t-1]`, starting at codec_bos; labels equal the teacher-forced codec inputs and end in EOS.
  - Bucketing length equals the actual length for an explicit language and is one larger for Auto.
  - `p_auto` (about 0.27 observed) keeps the instruction.
  - The dialect guard fires.
  - These are exactly the gaps in `test_equivalence_tri.py`: positions after codec_bos, labels, masks, padding and `train=True`. Fold them into the test.
- **Matches `generate()`.** Non-streaming layout, the instruction prefix as text_projection only, positions from the `cumsum` of the attention mask (so left padding at inference equals right padding in training), and the frame inputs (the sum of 16 codebooks plus `tts_pad`). `generate_custom_voice` defaults `non_streaming_mode=True`, and `gen_eval` and `rp_generate` pass it.
- **Speaker rows (b)–(d).**
  - Training vectors equal the saved rows exactly: the bf16 round-trip happens before training, and the save casts to bf16.
  - `codec_head` is a separate tensor (`tie_word_embeddings=False`), so presets receive no output-side gradient. Zero-gradient rows get exactly-zero Adam updates.
  - AdamW decay shrinks the preset rows by about 3×10⁻⁵ by step 775–872. That is below bf16 half-ULP (about 2×10⁻³), so the saved preset rows are bit-identical to the init.
  - Rows 3000–3013 (Base) and 3020–3033 (CustomVoice) are untrained (norm 0.022) and clear of the presets (3010, 3061, 3065, 3066, 2861–2878).
  - `speaker_rows()` reserves only rows named in the config. It knows nothing of rows below 2048, or of the unnamed trained rows 2049–2477, and does not enforce `row0 >= 2048`. That is fine for 3000 and 3020.
- **Math** (simulated with the real `length_batches`):

  | Arm | Micro-batches | Steps/epoch | Total steps | Warm-up | Saves | Stop |
  |---|---|---|---|---|---|---|
  | B1 | 1,240 | 620 | 1,240 | 37 | 310, 465, 620, 775 | 775 |
  | C1 | about 1,334 | 667 | — | — | 334, 500, 667, 834 | 834 |
  | C2 | about 1,396 | 698 | — | — | 349, 524, 698, 872 | 872 |

  - The learning rate at the four saves is 4.46e-6, 3.74e-6, 2.86e-6 and 1.97e-6 (C1/C2 within 1%).
  - Accumulation is 2, and the final partial micro-batch (5 clips) gets a full micro-batch's weight. This is trivial.
  - The scorer must map step numbers to epoch fractions per arm, since they differ between arms.
- **Disk, waste and settings.**
  - Epoch 0 ends before the 1.25 stop, so `state.pt` (about 14–16 GB) and `checkpoint-epoch-0` (4.3 GB) are written even though full-mode resume is refused. That costs minutes and about 20 GB per arm.
  - Peak disk is about 80 GB of 150, so it fits.
  - `--batch_size 16` skips the VRAM probe; Kurisu peaked at 36.7 GB, so this is fine.
- **Seeds.** `gen_eval` seeds per (suite, speaker, emotion, batch), so they are identical across checkpoints and arms ✓. S2/S3 twins use different seeds, which is fine.
- **Data.**
  - `train_with_codes` has 19,829 rows and `val` 826, matching the row files, with the same text and speaker and no dropped ids.
  - The reference clips are all train split, none dropped, and the speaker table matches the current `refs/*.txt`.
  - The S2 probes appear nowhere in train.
  - The R/P text pool excludes held-out texts by exact string only; a near-duplicate differing in punctuation would pass.
  - b10 resumes by `id` only: delete `*_with_codes` if R/P are regenerated.
- **Small-sample and design caveats.**
  - S1 is 191 clips, with wenzhou 3 and huanglili 6, so per-character S1 is noisy.
  - C1 changes the init, the rows *and* the data relative to B1, so "each arm changes one thing" is not true for B1 → C1.
- **Minor.**
  - `gen_eval` writes FLAC at PCM_16, which clips any |x| > 1.
  - Shell scripts are LF ✓.
  - The trainer `.strip()`s instructions and inference does not; none of our instructions have whitespace.
