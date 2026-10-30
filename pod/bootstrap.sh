#!/bin/bash
# Fresh pod -> environment, pinned models, then the equivalence gate on the pod's own
# library versions. Expects /workspace/tri/{code,qwen} uploaded from the local machine.
set -euo pipefail
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 TRI_POD_WORK=/workspace/tri PYTHONUNBUFFERED=1
W=/workspace/tri
mkdir -p $W/logs
cd $W

# 0. Fail fast on a bad GPU.
python -c "import torch; x=torch.randn(4096,4096,device='cuda'); print('gpu ok', torch.cuda.get_device_name(0), float((x@x).abs().mean()))"

# 1. Environment: the local versions the trainer and the equivalence test were checked with.
python -m pip install -q --upgrade pip
pip install -q "transformers==4.57.3" "accelerate==1.12.0" hf_transfer einops librosa soundfile \
    onnxruntime sox tensorboard
pip install -q --no-deps "qwen-tts==0.1.1"
python -c "import torch, transformers, qwen_tts; print('env ok', torch.__version__, transformers.__version__)"
echo STAGE_ENV_DONE

# 2. Models, pinned to the revisions verified locally.
python - <<'EOF'
from huggingface_hub import snapshot_download as s
for repo, rev in (("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
                  ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe")):
    print("model ok", s(repo, revision=rev))
EOF
echo STAGE_MODELS_DONE

# 3. Gate: trainer rows == generate_custom_voice inputs on this pod's torch/transformers.
export HF_HUB_OFFLINE=1
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-TTS-12Hz-1.7B-Base', revision='fd4b254389122332181a7c3db7f27e918eec64e3'))")
CV=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'))")
cd $W/code
python test_equivalence_tri.py --base "$BASE" --cv "$CV" --tmp /workspace/eqv 2>&1 | tee $W/logs/equivalence.log
grep -q "^PASS" $W/logs/equivalence.log || { echo "EQUIVALENCE_FAILED"; exit 1; }

# 4. Outer safety bound: the pod removes itself after 8 h (the local watchdog's 6 h deadline is the
#    primary, editable one), if this image gives the pod a key for runpodctl.
# SSH sessions do not inherit the container's RunPod variables; read the two needed from PID 1.
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do
  if [ -z "${!v:-}" ]; then
    val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -n 1)
    export "$v=$val"
  fi
done
if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
  nohup bash -c "sleep ${SELF_DESTRUCT_S:-28800}; runpodctl remove pod $RUNPOD_POD_ID" > $W/logs/self_destruct.log 2>&1 &
  echo "self-destruct armed in ${SELF_DESTRUCT_S:-28800}s"
else
  echo "self-destruct NOT available (runpodctl/RUNPOD_POD_ID/RUNPOD_API_KEY missing) - local watchdog only"
fi
touch $W/BOOTSTRAP_OK
echo BOOTSTRAP_DONE
