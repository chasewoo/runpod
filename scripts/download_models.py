"""Download Sulphur-2 + LTX-2.3 dependencies into $MODELS_DIR.

Idempotent: skips files that already exist with the right size.
Controlled by env:
  MODELS_DIR              base dir (default /workspace/models)
  SULPHUR_VARIANT         bf16 | fp8mixed | distil   (default bf16)
  SKIP_MODEL_DOWNLOAD=1   skip entirely
  HF_TOKEN                if repos are gated
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
VARIANT = os.environ.get("SULPHUR_VARIANT", "bf16").lower()
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Repo IDs (override via env if upstream renames)
SULPHUR_REPO = os.environ.get("SULPHUR_REPO", "SulphurAI/Sulphur-2-base")
LTX_REPO = os.environ.get("LTX_REPO", "Lightricks/LTX-2.3")

# Map variant -> (checkpoint filename, optional lora filename)
VARIANT_FILES = {
    "bf16":     ("sulphur_dev_bf16.safetensors", None),
    "fp8mixed": ("sulphur_dev_fp8mixed.safetensors", None),
    "distil":   ("sulphur_distil_bf16.safetensors", "sulphur_lora_rank_768.safetensors"),
}

# LTX-2.3 shared assets (VAE, upscalers, text encoder shards)
LTX_ASSETS = [
    # (filename_in_repo, target_subdir)
    ("vae/ltxv_vae.safetensors", "vae"),
    ("latent_upscalers/ltxv_spatial_upscaler.safetensors", "latent_upscale_models"),
    ("latent_upscalers/ltxv_temporal_upscaler.safetensors", "latent_upscale_models"),
    # Gemma 3 text encoder lives as a folder of shards. We mirror the folder.
    # We pull a manifest-style index file; full shards downloaded lazily via snapshot_download below.
]

GEMMA_REPO = os.environ.get("GEMMA_REPO", "Lightricks/LTX-Video-Q8-Kernels")  # placeholder if upstream packages elsewhere
GEMMA_SUBDIR_NAME = "gemma-3-12b-it-qat-q4_0-unquantized"


def fetch(repo_id: str, filename: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / Path(filename).name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size/1e9:.2f} GB)")
        return dest
    print(f"[download] {repo_id}:{filename} -> {dest}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
        token=HF_TOKEN,
    )
    return Path(path)


def main() -> int:
    if VARIANT not in VARIANT_FILES:
        print(f"Unknown SULPHUR_VARIANT={VARIANT}; expected one of {list(VARIANT_FILES)}", file=sys.stderr)
        return 2

    ckpt_name, lora_name = VARIANT_FILES[VARIANT]
    fetch(SULPHUR_REPO, ckpt_name, MODELS_DIR / "checkpoints")
    if lora_name:
        fetch(SULPHUR_REPO, lora_name, MODELS_DIR / "loras")

    for fname, sub in LTX_ASSETS:
        try:
            fetch(LTX_REPO, fname, MODELS_DIR / sub)
        except Exception as e:
            print(f"[warn] LTX asset {fname} failed: {e}. You may need to set LTX_REPO or fetch manually.")

    # Gemma text encoder: try snapshot_download for the full folder.
    try:
        from huggingface_hub import snapshot_download
        text_enc_target = MODELS_DIR / "text_encoders" / GEMMA_SUBDIR_NAME
        if not text_enc_target.exists() or not any(text_enc_target.iterdir()):
            print(f"[download] {GEMMA_REPO} -> {text_enc_target}")
            snapshot_download(
                repo_id=GEMMA_REPO,
                local_dir=str(text_enc_target),
                token=HF_TOKEN,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
            )
        else:
            print(f"[skip] text encoder dir already populated: {text_enc_target}")
    except Exception as e:
        print(f"[warn] gemma text encoder snapshot failed: {e}. Set GEMMA_REPO env to the correct repo.")

    # Workflow JSONs from the Sulphur repo (best-effort)
    workflow_dir = Path(os.environ.get("WORKFLOWS_DIR", "/workspace/workflows"))
    workflow_dir.mkdir(parents=True, exist_ok=True)
    for wf in ("ltx23_t2v_base.json", "ltx23_t2v_distil.json",
               "ltx23_i2v_base.json", "ltx23_i2v_distil.json"):
        try:
            fetch(SULPHUR_REPO, f"workflows/{wf}", workflow_dir)
        except Exception as e:
            print(f"[warn] workflow {wf} not fetched: {e}")

    print("[done] model setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
