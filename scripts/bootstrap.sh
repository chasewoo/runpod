#!/usr/bin/env bash
# Run inside a runpod/pytorch pod. Idempotent: safe to re-run.
set -euo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${MODELS_DIR:=/workspace/models}"
: "${OUTPUTS_DIR:=/workspace/outputs}"
: "${WORKFLOWS_DIR:=/workspace/workflows}"
: "${REPO_DIR:=/workspace/runsulphur}"
: "${COMFY_HOST:=127.0.0.1}"
: "${COMFY_PORT:=8188}"
: "${API_PORT:=8000}"
: "${SKIP_MODEL_DOWNLOAD:=0}"

echo "[bootstrap] $(date) starting"

apt-get update -qq && apt-get install -y -qq --no-install-recommends \
  git ffmpeg libgl1 libglib2.0-0 curl ca-certificates >/dev/null

mkdir -p "$MODELS_DIR"/{checkpoints,loras,text_encoders,vae,latent_upscale_models} \
         "$OUTPUTS_DIR" "$WORKFLOWS_DIR"

# ComfyUI
if [ ! -d "$COMFYUI_DIR/.git" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi
cd "$COMFYUI_DIR"
pip install -q -r requirements.txt

# Custom nodes
if [ ! -d "$COMFYUI_DIR/custom_nodes/ComfyUI-LTXVideo" ]; then
  git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git "$COMFYUI_DIR/custom_nodes/ComfyUI-LTXVideo"
  pip install -q -r "$COMFYUI_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt" || true
fi
if [ ! -d "$COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite" ]; then
  git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite"
  pip install -q -r "$COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt" || true
fi

# Symlink models dir
for sub in checkpoints loras text_encoders vae latent_upscale_models; do
  rm -rf "$COMFYUI_DIR/models/$sub"
  ln -s "$MODELS_DIR/$sub" "$COMFYUI_DIR/models/$sub"
done
rm -rf "$COMFYUI_DIR/output"
ln -s "$OUTPUTS_DIR" "$COMFYUI_DIR/output"

# API deps + dep pin: ComfyUI works with transformers 4.x; image ships 5.8 which
# wants huggingface_hub>=1.5. Downgrade transformers to keep our HF pin.
pip install -q "transformers>=4.45,<5" \
              fastapi==0.115.6 'uvicorn[standard]==0.32.1' httpx==0.27.2 \
              websockets==13.1 python-multipart==0.0.12 pydantic==2.9.2 \
              huggingface_hub==0.26.2 hf_transfer==0.1.8 pillow==11.0.0

export HF_HUB_ENABLE_HF_TRANSFER=1

# Models
if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
  python "$REPO_DIR/scripts/download_models.py"
fi

# Auto-tag workflows (if download succeeded)
if compgen -G "$WORKFLOWS_DIR/*.json" >/dev/null; then
  python "$REPO_DIR/scripts/autotag_workflows.py" "$WORKFLOWS_DIR" || true
fi

# Start ComfyUI in background, redirecting logs
# --disable-smart-memory: skip the pinned-memory pre-allocation that hangs
#   on MooseFS network filesystem (locked the pod for >5 min at startup)
# --gpu-only: avoid CPU<->GPU weight offload; A100 80GB fits Sulphur-2 bf16 easily
nohup python "$COMFYUI_DIR/main.py" --listen "$COMFY_HOST" --port "$COMFY_PORT" \
  --disable-auto-launch --disable-smart-memory --gpu-only \
  > /workspace/comfyui.log 2>&1 &
echo "[bootstrap] comfyui pid=$!"

# Wait for comfy
for i in $(seq 1 180); do
  if curl -sf "http://$COMFY_HOST:$COMFY_PORT/system_stats" >/dev/null; then
    echo "[bootstrap] comfy ready"
    break
  fi
  sleep 2
done

# Start API in foreground (so the pod stays alive)
cd "$REPO_DIR"
exec uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT"
