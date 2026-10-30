#!/bin/bash
# R/P instruction-data generation on the pod (eval/rp_generate.py --from_plan), after bootstrap.
set -euo pipefail
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
W=/workspace/tri
until [ -f $W/BOOTSTRAP_OK ]; do sleep 20; done
cd $W
CV=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'))")
python code/rp_generate.py --from_plan rp/plan.json --cv "$CV" --table qwen/speaker_table.pt --out rp --batch 96
echo RP_DONE
