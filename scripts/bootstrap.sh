#!/usr/bin/env bash
# Run on a fresh RunPod GPU pod (or any cloud GPU box) that has the persistent
# Sulphur volume mounted at /workspace. Idempotent: each step skips if already done.
#
# Required env (passed via RunPod template or pre-set in /workspace/.env):
#   HF_TOKEN   — needed for the gated google/gemma-3-12b repo
#
# Optional:
#   SKIP_MODEL_DOWNLOAD=1    skip the (slow) HF fetch step
#   API_PORT=8000            FastAPI port
#   COMFY_PORT=8188          ComfyUI port

set -euo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${MODELS_DIR:=/workspace/models}"
: "${OUTPUTS_DIR:=/workspace/outputs}"
: "${WORKFLOWS_DIR:=/workspace/workflows}"
: "${REPO_DIR:=/workspace/runsulphur}"
: "${COMFY_HOST:=0.0.0.0}"
: "${COMFY_PORT:=8188}"
: "${API_PORT:=8000}"
: "${SKIP_MODEL_DOWNLOAD:=0}"

# Source persisted env (HF_TOKEN, etc.) without echoing it
if [ -f /workspace/.env ]; then
  set -a
  # shellcheck disable=SC1091
  . /workspace/.env
  set +a
fi

echo "[bootstrap] $(date) starting"

mkdir -p "$MODELS_DIR"/{checkpoints,loras,text_encoders,vae,latent_upscale_models} \
         "$OUTPUTS_DIR" "$WORKFLOWS_DIR" /workspace/logs

# ── System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
  curl ca-certificates tmux >/dev/null

# ── ComfyUI core
if [ ! -d "$COMFYUI_DIR/.git" ]; then
  echo "[bootstrap] cloning ComfyUI"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi
pip install -q -r "$COMFYUI_DIR/requirements.txt"

# ── Custom-node packs (everything the Sulphur workflows depend on)
declare -a NODE_REPOS=(
  "Lightricks/ComfyUI-LTXVideo"
  "Kosinkadink/ComfyUI-VideoHelperSuite"
  "kijai/ComfyUI-KJNodes"
  "cubiq/ComfyUI_essentials"
  "sipherxyz/comfyui-art-venture"
  "aria1th/ComfyUI-LogicUtils"
  "WASasquatch/was-node-suite-comfyui"
  "rgthree/rgthree-comfy"
  "chrisgoringe/cg-use-everywhere"
  "pythongosssss/ComfyUI-Custom-Scripts"
)
for repo in "${NODE_REPOS[@]}"; do
  name=$(basename "$repo")
  dest="$COMFYUI_DIR/custom_nodes/$name"
  if [ ! -d "$dest/.git" ]; then
    echo "[bootstrap] installing custom node: $name"
    git clone --depth 1 "https://github.com/$repo" "$dest"
  fi
  if [ -f "$dest/requirements.txt" ]; then
    pip install -q -r "$dest/requirements.txt" || true
  fi
done

# Link ComfyUI's model dirs into our persistent volume layout
for sub in checkpoints loras text_encoders vae latent_upscale_models; do
  rm -rf "$COMFYUI_DIR/models/$sub"
  ln -s "$MODELS_DIR/$sub" "$COMFYUI_DIR/models/$sub"
done
rm -rf "$COMFYUI_DIR/output"
ln -s "$OUTPUTS_DIR" "$COMFYUI_DIR/output"

# ── Python deps for the API + downloads
# transformers 5.x bundled in the runpod/pytorch image wants huggingface_hub>=1.5;
# pin both back to a compatible pair for ComfyUI.
pip install -q "transformers>=4.45,<5" \
              fastapi==0.115.6 'uvicorn[standard]==0.32.1' httpx==0.27.2 \
              websockets==13.1 python-multipart==0.0.12 pydantic==2.9.2 \
              huggingface_hub==0.26.2 hf_transfer==0.1.8 pillow==11.0.0 \
              opencv-python-headless imageio-ffmpeg sageattention

# ── Models + workflows
if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
  echo "[bootstrap] running download_models.py"
  python "$REPO_DIR/scripts/download_models.py"
fi
# Patch shipped workflow JSONs (filename rewrites, sage_attention defaults)
echo "[bootstrap] running setup_workflows.py"
python "$REPO_DIR/scripts/setup_workflows.py" || true

# ── Launch ComfyUI in tmux (survives SSH disconnect)
COMFY_LOG=/workspace/logs/comfyui.log
echo "[bootstrap] starting ComfyUI"
tmux kill-session -t comfy 2>/dev/null || true
tmux new-session -d -s comfy "exec python $COMFYUI_DIR/main.py \
  --listen $COMFY_HOST --port $COMFY_PORT \
  --disable-auto-launch --disable-smart-memory --gpu-only \
  >$COMFY_LOG 2>&1"

# Wait for ComfyUI to be ready
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null; then
    echo "[bootstrap] ComfyUI ready after ${i}x2s"
    break
  fi
  sleep 2
done

# UI→API workflow conversion (needs ComfyUI's /object_info schema)
echo "[bootstrap] converting UI workflows to API format"
python "$REPO_DIR/scripts/ui_to_api.py" "$WORKFLOWS_DIR" \
  --comfy-url "http://127.0.0.1:$COMFY_PORT" --inplace || true
# Re-run the rename+sage patch on the freshly-converted JSONs, then autotag
python "$REPO_DIR/scripts/setup_workflows.py" || true

# ── Start FastAPI in foreground so the container stays alive
cd "$REPO_DIR"
exec uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT"
