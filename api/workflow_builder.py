"""Inject API params into a ComfyUI workflow JSON by node title.

Convention: in the source workflow (edited in ComfyUI), set node titles to one of:
    API_PROMPT          CLIPTextEncode positive
    API_NEGATIVE        CLIPTextEncode negative
    API_IMAGE           LoadImage  (i2v only)
    API_DIMS            EmptyLatentVideo / LTXVEmptyLatent (width, height, length)
    API_SAMPLER         KSampler / SamplerCustomAdvanced (steps, cfg, seed)
    API_SAVE            VHS_VideoCombine / SaveAnimatedWEBP  (filename_prefix)

If the workflow doesn't have one of these tags, that param is silently skipped.
Unknown advanced overrides are applied verbatim.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _by_title(workflow: dict) -> dict[str, dict]:
    """Return mapping title -> node-dict. ComfyUI API-format workflows: {node_id: {class_type, inputs, _meta}}."""
    out = {}
    for nid, node in workflow.items():
        title = (node.get("_meta") or {}).get("title")
        if title:
            out[title] = node
    return out


def _set_input(node: dict, key: str, value: Any) -> None:
    node.setdefault("inputs", {})[key] = value


def build(
    template: dict,
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    steps: int,
    cfg: float,
    seed: int | None,
    image_filename: str | None = None,
    save_prefix: str = "sulphur",
    advanced: dict | None = None,
) -> dict:
    wf = copy.deepcopy(template)
    nodes = _by_title(wf)

    if "API_PROMPT" in nodes:
        _set_input(nodes["API_PROMPT"], "text", prompt)
    if "API_NEGATIVE" in nodes:
        _set_input(nodes["API_NEGATIVE"], "text", negative)
    if "API_IMAGE" in nodes and image_filename:
        _set_input(nodes["API_IMAGE"], "image", image_filename)
    if "API_DIMS" in nodes:
        n = nodes["API_DIMS"]
        for k, v in (("width", width), ("height", height),
                     ("length", num_frames), ("num_frames", num_frames),
                     ("frame_rate", fps), ("fps", fps)):
            if k in (n.get("inputs") or {}):
                _set_input(n, k, v)
    if "API_SAMPLER" in nodes:
        n = nodes["API_SAMPLER"]
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        for k, v in (("steps", steps), ("cfg", cfg), ("seed", seed), ("noise_seed", seed)):
            if k in (n.get("inputs") or {}):
                _set_input(n, k, v)
    if "API_SAVE" in nodes:
        n = nodes["API_SAVE"]
        for k in ("filename_prefix", "filename"):
            if k in (n.get("inputs") or {}):
                _set_input(n, k, save_prefix)
        if "frame_rate" in (n.get("inputs") or {}):
            _set_input(n, "frame_rate", fps)

    # advanced overrides: {title: {input: value}}
    if advanced:
        for title, kvs in advanced.items():
            if title in nodes and isinstance(kvs, dict):
                for k, v in kvs.items():
                    _set_input(nodes[title], k, v)

    return wf
