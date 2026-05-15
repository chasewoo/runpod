"""Download every model file Sulphur-2's shipped workflows reference.

Idempotent: anything with non-zero size on disk is skipped, including the
sulphur_final.safetensors alias and the moved-out distill LoRA.

Env vars:
  MODELS_DIR        target dir (default /workspace/models)
  WORKFLOWS_DIR     workflow JSON output dir (default /workspace/workflows)
  HF_TOKEN          required for the gated google/gemma-3-12b repo
  HF_HUB_ENABLE_HF_TRANSFER  set to 1 for accelerated downloads (we default off
                             because it hangs on the HF Xet CDN under some loads)
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
WORKFLOWS_DIR = Path(os.environ.get("WORKFLOWS_DIR", "/workspace/workflows"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Each entry: (repo_id, file_in_repo, target_dir, optional rename)
SINGLE_FILES = [
    # Sulphur 2 base ckpt (46 GB) — drop the all-in-one in checkpoints/
    ("SulphurAI/Sulphur-2-base", "sulphur_dev_bf16.safetensors", MODELS_DIR / "checkpoints", None),
    # Sulphur LoRA (10 GB) — workflow references "sulphur_final.safetensors"; we also
    # symlink that name later in setup_workflows.py.
    ("SulphurAI/Sulphur-2-base", "sulphur_lora_rank_768.safetensors", MODELS_DIR / "loras", None),
    # Sulphur distill LoRA — workflows reference it at flat path, so flatten the subdir.
    ("SulphurAI/Sulphur-2-base",
     "distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
     MODELS_DIR / "loras",
     "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"),
    # LTX 2.3 base in fp8 (22 GB) — the workflow's CheckpointLoaderSimple / TextEncoderLoader / AudioVAELoader all point here.
    ("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-dev-fp8.safetensors", MODELS_DIR / "checkpoints", None),
    # LTX distill LoRAs
    ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors", MODELS_DIR / "loras", None),
    # Upscalers — workflows ask for x2-1.0; we also keep 1.1 since some variants use it.
    ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", MODELS_DIR / "latent_upscale_models", None),
    ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", MODELS_DIR / "latent_upscale_models", None),
    ("Lightricks/LTX-2.3", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors", MODELS_DIR / "latent_upscale_models", None),
    # Gemma fp8 single-file text encoder (Pavpif's repack)
    ("Pavpif/ltx2-gemma3-text-encoder", "model_gemma_3_12B_it_fp8_e4m3fn.safetensors", MODELS_DIR / "text_encoders", None),
    # Workflow JSONs (filenames have a literal space in the source repo)
    ("SulphurAI/Sulphur-2-base", "workflows/ltx23_t2v base.json",      WORKFLOWS_DIR, "ltx23_t2v_base.json"),
    ("SulphurAI/Sulphur-2-base", "workflows/ltx23_t2v distilled.json", WORKFLOWS_DIR, "ltx23_t2v_distil.json"),
    ("SulphurAI/Sulphur-2-base", "workflows/ltx23_i2v base.json",      WORKFLOWS_DIR, "ltx23_i2v_base.json"),
    ("SulphurAI/Sulphur-2-base", "workflows/ltx23_i2v distilled.json", WORKFLOWS_DIR, "ltx23_i2v_distil.json"),
]

# Multi-shard repo (the unquantized Gemma weights). Required by some advanced workflows
# that read shards via the index. The fp8 single-file above is what the default
# t2v/i2v workflows use, so this is optional — included for completeness.
GEMMA_SHARDS_TARGET = MODELS_DIR / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_SHARDS_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


def fetch(repo: str, filename: str, target: Path, rename: str | None = None) -> bool:
    target.mkdir(parents=True, exist_ok=True)
    dest = target / (rename or Path(filename).name)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} ({dest.stat().st_size/1e9:.2f} GB)")
        return True
    t0 = time.time()
    print(f"[dl  ] {repo}:{filename}")
    try:
        downloaded = hf_hub_download(repo_id=repo, filename=filename,
                                     local_dir=str(target), token=HF_TOKEN)
        downloaded = Path(downloaded)
        # hf_hub_download preserves the in-repo path; if filename has a subdir or
        # if we want a rename, move into place.
        if downloaded != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(dest))
            # clean up empty subdirs (e.g. workflows/, distill_loras/)
            for parent in [downloaded.parent, downloaded.parent.parent]:
                try:
                    parent.rmdir()
                except OSError:
                    pass
        sz = dest.stat().st_size
        print(f"[ok  ] {dest.name} {sz/1e9:.2f} GB in {time.time()-t0:.0f}s")
        return True
    except Exception as e:
        print(f"[fail] {filename}: {e}")
        return False


def main() -> int:
    if not HF_TOKEN:
        print("warning: HF_TOKEN unset; gated repos (Gemma) will 401.")
    ok = fail = 0
    for repo, fn, tgt, rename in SINGLE_FILES:
        if fetch(repo, fn, tgt, rename):
            ok += 1
        else:
            fail += 1

    # Symlink the workflow-expected alias name
    lora_dir = MODELS_DIR / "loras"
    real = lora_dir / "sulphur_lora_rank_768.safetensors"
    alias = lora_dir / "sulphur_final.safetensors"
    if real.exists() and not alias.exists():
        alias.symlink_to(real.name)
        print(f"[link] {alias.name} -> {real.name}")

    # Optional Gemma shards snapshot (skip if dir already populated)
    if not GEMMA_SHARDS_TARGET.exists() or not any(GEMMA_SHARDS_TARGET.iterdir()):
        try:
            print(f"[dl  ] {GEMMA_SHARDS_REPO} shards -> {GEMMA_SHARDS_TARGET}")
            GEMMA_SHARDS_TARGET.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=GEMMA_SHARDS_REPO,
                local_dir=str(GEMMA_SHARDS_TARGET),
                token=HF_TOKEN,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
            )
            sz = sum(f.stat().st_size for f in GEMMA_SHARDS_TARGET.rglob("*") if f.is_file())
            print(f"[ok  ] gemma shards total {sz/1e9:.2f} GB")
            ok += 1
        except Exception as e:
            print(f"[fail] gemma shards (gated? accept license at https://huggingface.co/{GEMMA_SHARDS_REPO}): {e}")
            fail += 1
    else:
        print(f"[skip] gemma shards dir already populated")

    print(f"\n[done] downloads ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
