#!/usr/bin/env bash
set -euo pipefail

: "${COMFYUI_DIR:=/opt/ComfyUI}"
: "${MODELS_DIR:=/workspace/models}"
: "${OUTPUTS_DIR:=/workspace/outputs}"
: "${COMFY_HOST:=127.0.0.1}"
: "${COMFY_PORT:=8188}"
: "${API_PORT:=8000}"
: "${SKIP_MODEL_DOWNLOAD:=0}"

mkdir -p "$MODELS_DIR"/{checkpoints,loras,text_encoders,vae,latent_upscale_models} "$OUTPUTS_DIR"

# Symlink workspace model dirs into ComfyUI's expected layout
for sub in checkpoints loras text_encoders vae latent_upscale_models; do
  rm -rf "$COMFYUI_DIR/models/$sub"
  ln -s "$MODELS_DIR/$sub" "$COMFYUI_DIR/models/$sub"
done

# Outputs symlink
rm -rf "$COMFYUI_DIR/output"
ln -s "$OUTPUTS_DIR" "$COMFYUI_DIR/output"

if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
  python /app/scripts/download_models.py
fi

# Start ComfyUI in background
cd "$COMFYUI_DIR"
python main.py --listen "$COMFY_HOST" --port "$COMFY_PORT" --disable-auto-launch &
COMFY_PID=$!

echo "[entrypoint] ComfyUI pid=$COMFY_PID, waiting for it to become ready..."
for i in $(seq 1 120); do
  if curl -sf "http://$COMFY_HOST:$COMFY_PORT/system_stats" >/dev/null; then
    echo "[entrypoint] ComfyUI ready."
    break
  fi
  sleep 2
done

cd /app
exec uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT"
