"""Derive `ltx23_*_full.json` workflows that use the baked sulphur_dev_bf16 weights
standalone (no LoRA stack).

Per Sulphur's README: "just use the lora or use the full models, don't use both
at the same time." The shipped workflows take the LoRA path; this script clones
them and switches to the full-model path.

Run after `ui_to_api.py` (workflows must already be in API format).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WF = Path(os.environ.get("WORKFLOWS_DIR", "/workspace/workflows"))
NEW_CKPT = "sulphur_dev_bf16.safetensors"
SRC_DST = [
    ("ltx23_i2v_distil.json", "ltx23_i2v_full.json"),
    ("ltx23_t2v_distil.json", "ltx23_t2v_full.json"),
]


def main() -> int:
    for src, dst in SRC_DST:
        src_path = WF / src
        if not src_path.exists():
            print(f"[skip] source {src} not found")
            continue
        with src_path.open() as fh:
            wf = json.load(fh)

        ckpt_changes = lora_bypassed = 0
        for nid, n in wf.items():
            ct = n.get("class_type", "")
            ins = n.setdefault("inputs", {})
            if ct in ("CheckpointLoaderSimple", "LTXAVTextEncoderLoader", "LTXVAudioVAELoader"):
                if isinstance(ins.get("ckpt_name"), str):
                    ins["ckpt_name"] = NEW_CKPT
                    ckpt_changes += 1
            if ct == "LoraLoaderModelOnly":
                for k in ("strength_model", "strength_clip"):
                    if k in ins:
                        ins[k] = 0
                lora_bypassed += 1

        with (WF / dst).open("w") as fh:
            json.dump(wf, fh, indent=2)
        print(f"[ok] {dst}: ckpt_refs={ckpt_changes} loras_bypassed={lora_bypassed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
