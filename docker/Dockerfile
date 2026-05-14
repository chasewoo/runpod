FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    COMFYUI_DIR=/opt/ComfyUI \
    MODELS_DIR=/workspace/models \
    OUTPUTS_DIR=/workspace/outputs

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates ffmpeg libgl1 libglib2.0-0 \
      python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

# ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_DIR
WORKDIR $COMFYUI_DIR
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install -r requirements.txt

# LTX-Video custom nodes
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git $COMFYUI_DIR/custom_nodes/ComfyUI-LTXVideo \
    && pip install -r $COMFYUI_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt || true

# Common helper nodes (VideoHelperSuite for mp4 saving, ComfyUI-Manager for diagnostics)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git $COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite \
    && pip install -r $COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt || true

# FastAPI server deps
RUN pip install \
      fastapi==0.115.6 uvicorn[standard]==0.32.1 \
      httpx==0.27.2 websockets==13.1 \
      python-multipart==0.0.12 pydantic==2.9.2 \
      huggingface_hub==0.26.2 hf_transfer==0.1.8 \
      pillow==11.0.0

# App
WORKDIR /app
COPY api /app/api
COPY scripts /app/scripts
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8188

CMD ["/app/entrypoint.sh"]
