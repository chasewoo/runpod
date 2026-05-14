"""Auto-tag ComfyUI workflow JSONs with API_* titles so workflow_builder can inject params.

Heuristics on API-format JSON ({node_id: {class_type, inputs, _meta}}):
  - CLIPTextEncode whose `text` input looks "positive-ish" / longer / has no "low quality" -> API_PROMPT
    The other CLIPTextEncode -> API_NEGATIVE
  - LoadImage -> API_IMAGE
  - EmptyLatentVideo / LTXVEmptyLatent / EmptyLTXVLatent / EmptyHunyuanLatentVideo -> API_DIMS
  - KSampler / KSamplerAdvanced / SamplerCustomAdvanced -> API_SAMPLER
    (if multiple, picks the one with highest steps; SamplerCustomAdvanced gets API_SAMPLER too)
  - VHS_VideoCombine / SaveAnimatedWEBP / SaveImage (video) -> API_SAVE

Usage:
    python scripts/autotag_workflows.py /workspace/workflows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


NEG_HINTS = ("low quality", "blurry", "distorted", "ugly", "bad", "worst", "nsfw", "watermark")

DIM_CLASSES = {"EmptyLatentVideo", "LTXVEmptyLatent", "EmptyLTXVLatent",
               "EmptyLTXVLatentVideo", "EmptyHunyuanLatentVideo",
               "LTXVImgToVideo", "LTXVConditioning"}
SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced",
                   "LTXVScheduler", "LTXVKSampler"}
SAVE_CLASSES = {"VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG"}
IMAGE_LOAD_CLASSES = {"LoadImage"}


def set_title(node: dict, title: str) -> None:
    node.setdefault("_meta", {})["title"] = title


def _score_negative(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for h in NEG_HINTS if h in t)


def tag(workflow: dict) -> dict[str, list[str]]:
    """Mutate workflow in place. Return {tag: [node_ids]} report."""
    report: dict[str, list[str]] = {}

    clip_nodes = []
    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        if ct == "CLIPTextEncode":
            clip_nodes.append((nid, node))
        elif ct in IMAGE_LOAD_CLASSES:
            set_title(node, "API_IMAGE")
            report.setdefault("API_IMAGE", []).append(nid)
        elif ct in SAVE_CLASSES:
            set_title(node, "API_SAVE")
            report.setdefault("API_SAVE", []).append(nid)

    # CLIPTextEncode: classify by negative-hint score
    if clip_nodes:
        scored = [(nid, n, _score_negative((n.get("inputs") or {}).get("text", ""))) for nid, n in clip_nodes]
        # negative = highest score; if tie, pick the shorter text
        scored.sort(key=lambda x: (-x[2], len((x[1].get("inputs") or {}).get("text", ""))))
        neg_nid, neg_node, _ = scored[0]
        set_title(neg_node, "API_NEGATIVE")
        report.setdefault("API_NEGATIVE", []).append(neg_nid)
        for nid, node, _ in scored[1:]:
            set_title(node, "API_PROMPT")
            report.setdefault("API_PROMPT", []).append(nid)
            break  # only tag one positive

    # DIMS: prefer EmptyLatentVideo-family, fallback to any with width+height+length inputs
    dims_node = None
    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        ins = node.get("inputs") or {}
        if ct in DIM_CLASSES and ("width" in ins or "length" in ins or "num_frames" in ins):
            dims_node = (nid, node)
            break
    if dims_node is None:
        for nid, node in workflow.items():
            ins = node.get("inputs") or {}
            if "width" in ins and "height" in ins and ("length" in ins or "num_frames" in ins):
                dims_node = (nid, node)
                break
    if dims_node:
        nid, node = dims_node
        set_title(node, "API_DIMS")
        report.setdefault("API_DIMS", []).append(nid)

    # SAMPLER: prefer KSampler-style with `steps` input
    sampler_candidates = []
    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        ins = node.get("inputs") or {}
        if ct in SAMPLER_CLASSES or "steps" in ins or "noise_seed" in ins:
            sampler_candidates.append((nid, node, int(ins.get("steps", 0) or 0)))
    if sampler_candidates:
        sampler_candidates.sort(key=lambda x: -x[2])
        nid, node, _ = sampler_candidates[0]
        set_title(node, "API_SAMPLER")
        report.setdefault("API_SAMPLER", []).append(nid)

    return report


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: autotag_workflows.py <dir-or-file>")
        return 2
    target = Path(argv[1])
    files = [target] if target.is_file() else sorted(target.glob("*.json"))
    if not files:
        print(f"no JSON files at {target}")
        return 1
    for f in files:
        with f.open() as fh:
            wf = json.load(fh)
        # The file might be UI-format (with `nodes` array) instead of API-format.
        # We only tag API-format.
        if not isinstance(wf, dict) or "nodes" in wf:
            print(f"[skip] {f.name}: not API-format (no node_id->node map). Re-export as API format.")
            continue
        rpt = tag(wf)
        with f.open("w") as fh:
            json.dump(wf, fh, indent=2)
        print(f"[tag]  {f.name}: { {k: v for k, v in rpt.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
