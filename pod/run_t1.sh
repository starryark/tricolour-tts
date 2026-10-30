#!/bin/bash
# T1 (research/T1_preregistration.md + Amendment 1). Order: untrained references B0/C0 first
# (they also time generation), then per arm: train -> evaluate checkpoints newest first.
# The local side copies each finished arm's audio home while the next arm trains.
#   usage: run_t1.sh [arm ...]   (default: B1 C1 C2)
set -euo pipefail
export HF_HOME=/workspace/hf TRI_POD_WORK=/workspace/tri PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
W=/workspace/tri
Q=$W/qwen
mkdir -p $W/logs $W/runs $W/eval
[ -f $W/BOOTSTRAP_OK ] || { echo "bootstrap has not passed - run bootstrap.sh"; exit 1; }
cd $W/code
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-TTS-12Hz-1.7B-Base', revision='fd4b254389122332181a7c3db7f27e918eec64e3'))")
CV=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'))")
if [ $# -gt 0 ]; then ARMS=("$@"); else ARMS=(B1 C1 C2); fi

need () { # need <file>: exists and non-empty
  [ -s "$1" ] || { echo "missing or empty: $1"; exit 1; }
}
need $Q/train_with_codes.jsonl; need $Q/val_with_codes.jsonl; need $Q/speaker_table.pt; need $Q/suites.json

COMMON=(--mode full --full_dtype fp32 --optimizer adamw --lr 5e-6 --weight_decay 0.01
        --epochs 2 --warmup_ratio 0.03 --min_lr_ratio 0.1 --batch_size 16 --effective_batch 32
        --grad_clip 1.0 --sub_talker_weight 0.3 --label_alignment fixed --p_auto_lang 0.25
        --max_frames 400 --seed 20261030 --save_at 0.5,0.75,1.0,1.25 --stop_after 1.25
        --no_epoch_checkpoints --eval_every 200 --max_val_batches 1000 --log_every 10 --keep_last 0
        --vram_fraction 0.95 --attn sdpa --tensorboard --speaker_table $Q/speaker_table.pt)

data () { # data <arm>: build that arm's train/val files
  local arm=$1 extra=()
  case $arm in
    B1) extra=() ;;
    C1) extra=(R) ;;
    C2) extra=(R P) ;;
    C1b) extra=(R) ;;
  esac
  local tr=($Q/train_with_codes.jsonl) va=($Q/val_with_codes.jsonl)
  for s in "${extra[@]}"; do
    need $Q/${s}_train_with_codes.jsonl; need $Q/${s}_val_with_codes.jsonl
    tr+=($Q/${s}_train_with_codes.jsonl); va+=($Q/${s}_val_with_codes.jsonl)
  done
  mkdir -p $Q/arms
  cat "${tr[@]}" > $Q/arms/${arm}_train.jsonl
  cat "${va[@]}" > $Q/arms/${arm}_val.jsonl
  echo "[t1] $arm: $(wc -l < $Q/arms/${arm}_train.jsonl) train / $(wc -l < $Q/arms/${arm}_val.jsonl) val rows"
}

train () { # train <arm> <init> <row0> [extra flags]
  local arm=$1 init=$2 row0=$3; shift 3
  if [ -f "$W/runs/$arm/TRAIN_DONE" ]; then echo "[t1] $arm already trained - skipped"; return 0; fi
  rm -rf "$W/runs/$arm"
  python train_sft.py --init_model_path "$init" --spk_row0 $row0 \
      --train_jsonl $Q/arms/${arm}_train.jsonl --val_jsonl $Q/arms/${arm}_val.jsonl \
      --name $arm --out $W/runs/$arm "${COMMON[@]}" "$@" 2>&1 | tee $W/logs/train_$arm.log
  ls $W/runs/$arm/checkpoint-step-*/config.json > /dev/null
  touch $W/runs/$arm/TRAIN_DONE
}

evalgen () { # evalgen <arm>: newest checkpoint first (prereg drop order: 0.5 epoch last)
  local arm=$1 args=()
  # EVAL_CKPTS=n keeps only the n newest checkpoints (prereg Amendment 1 drop order)
  for ck in $(ls -d $W/runs/$arm/checkpoint-step-* | sort -t- -k3 -nr | head -n ${EVAL_CKPTS:-99}); do args+=(--ckpt "$ck"); done
  python gen_eval.py --suites $Q/suites.json "${args[@]}" --out $W/eval 2>&1 | tee -a $W/logs/eval_$arm.log
  touch $W/eval/$arm/EVAL_DONE
  echo "EVALGEN $arm"
}

# untrained references, same save and generation path as the trained checkpoints
[ -f $W/runs/B0/checkpoint-step-0/config.json ] || python make_step0.py --init "$BASE" --row0 3000 \
    --table $Q/speaker_table.pt --out $W/runs/B0/checkpoint-step-0
[ -f $W/runs/C0/checkpoint-step-0/config.json ] || python make_step0.py --init "$CV" --row0 3020 \
    --table $Q/speaker_table.pt --out $W/runs/C0/checkpoint-step-0
evalgen B0
evalgen C0

for arm in "${ARMS[@]}"; do
  data $arm
  case $arm in
    B1) train B1 "$BASE" 3000 ;;
    C1) train C1 "$CV" 3020 --preset_speakers vivian,serena,uncle_fu ;;
    C2) train C2 "$CV" 3020 --preset_speakers vivian,serena,uncle_fu ;;
    C1b) train C1b "$CV" 3020 --preset_speakers vivian,serena,uncle_fu --seed 20261031 ;;   # 2nd seed of the selected arm
  esac
  echo "TRAINED $arm"
  evalgen $arm
done
echo T1_DONE
