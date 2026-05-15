"""Patch the shipped Sulphur workflow JSONs so ComfyUI can run them as-is.

Three patches:
1. Rewrite the placeholder Gemma filename (`gemma_3_12B_it_fp4_mixed.safetensors`)
   to the file we actually downloaded (`model_gemma_3_12B_it_fp8_e4m3fn.safetensors`).
2. Default `PathchSageAttentionKJ.sage_attention` to "auto" when missing.
3. (Optional) Run the existing autotag_workflows pass so the FastAPI builder can
   inject prompt/dims/sampler by node title.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WF_DIR = Path(os.environ.get("WORKFLOWS_DIR", "/workspace/workflows"))
REPO_DIR = Path(os.environ.get("REPO_DIR", "/workspace/runsulphur"))

FILENAME_REPLACEMENTS = {
    "gemma_3_12B_it_fp4_mixed.safetensors": "model_gemma_3_12B_it_fp8_e4m3fn.safetensors",
}


def _walk_replace(o, table):
    """Mutate strings in-place inside dict/list trees according to table. Returns count."""
    count = 0
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and v in table:
                o[k] = table[v]
                count += 1
            else:
                count += _walk_replace(v, table)
    elif isinstance(o, list):
        for v in o:
            count += _walk_replace(v, table)
    return count


def main() -> int:
    files = sorted(WF_DIR.glob("*.json"))
    if not files:
        print(f"[setup_workflows] no JSONs in {WF_DIR}")
        return 1

    for f in files:
        with f.open() as fh:
            wf = json.load(fh)

        # UI-format conversion is handled by ui_to_api.py separately (needs ComfyUI running).
        if isinstance(wf, dict) and "nodes" in wf:
            print(f"[skip] {f.name}: UI-format (convert via ui_to_api.py first)")
            continue

        # Filename replacement
        renames = _walk_replace(wf, FILENAME_REPLACEMENTS)

        # sage_attention default
        sage_defaults = 0
        for n in wf.values():
            if isinstance(n, dict) and n.get("class_type") == "PathchSageAttentionKJ":
                ins = n.setdefault("inputs", {})
                if "sage_attention" not in ins:
                    ins["sage_attention"] = "auto"
                    sage_defaults += 1

        with f.open("w") as fh:
            json.dump(wf, fh, indent=2)
        print(f"[ok  ] {f.name}: renames={renames} sage_defaults={sage_defaults}")

    # Run autotag (it now handles list-typed text inputs cleanly)
    autotag = REPO_DIR / "scripts" / "autotag_workflows.py"
    if autotag.exists():
        print("[setup_workflows] running autotag")
        subprocess.run([sys.executable, str(autotag), str(WF_DIR)], check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
