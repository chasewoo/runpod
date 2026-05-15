"""Convert ComfyUI UI-format workflow JSON to API-format.

UI format: {nodes: [{id, type, widgets_values, inputs:[{name,link?,...}], outputs}, ...],
            links: [[link_id, src_node, src_slot, dst_node, dst_slot, type], ...]}

API format: {<node_id>: {class_type, inputs: {<name>: value | [src_node, src_slot]}, _meta: {title}}, ...}

Requires running against a ComfyUI instance to fetch node schemas via /object_info,
so that widget values can be mapped to named input keys.

Usage:
    python scripts/ui_to_api.py <ui_workflow.json> [--comfy-url http://127.0.0.1:8188] [-o out.json]
    python scripts/ui_to_api.py <dir>             # batch convert all *.json in dir, write *_api.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def fetch_object_info(comfy_url: str) -> dict:
    r = httpx.get(f"{comfy_url}/object_info", timeout=30)
    r.raise_for_status()
    return r.json()


def convert(ui: dict, object_info: dict) -> dict:
    """Return API-format dict {str(node_id): {class_type, inputs, _meta}}."""
    api: dict[str, dict] = {}

    # links: link_id -> (src_node, src_slot)
    links_by_id: dict[int, tuple[int, int]] = {}
    for link in ui.get("links", []):
        # link: [link_id, src_node, src_slot, dst_node, dst_slot, link_type]
        if isinstance(link, list) and len(link) >= 6:
            link_id, src_n, src_s = link[0], link[1], link[2]
            links_by_id[link_id] = (src_n, src_s)

    for node in ui.get("nodes", []):
        nid = node["id"]
        class_type = node.get("type")
        if class_type in (None, "Note", "PrimitiveNode", "Reroute", "MarkdownNote"):
            # Pass-through utility nodes; we still emit them so links resolve, but reroutes need flattening.
            if class_type == "Reroute":
                # Reroutes get flattened — they shouldn't appear in API format.
                continue
            if class_type in ("Note", "MarkdownNote"):
                continue

        node_info = object_info.get(class_type)
        if node_info is None:
            # Unknown class. Pass widget values positionally with synthetic names.
            inputs: dict[str, Any] = {}
            for i, val in enumerate(node.get("widgets_values") or []):
                inputs[f"arg_{i}"] = val
            for inp in node.get("inputs") or []:
                if inp.get("link") is not None:
                    src = links_by_id.get(inp["link"])
                    if src:
                        inputs[inp["name"]] = [str(src[0]), src[1]]
            api[str(nid)] = {
                "class_type": class_type or "Unknown",
                "inputs": inputs,
                "_meta": {"title": node.get("title") or class_type},
            }
            continue

        # Schema: object_info[class_type]["input"]["required"|"optional"]
        # ordered dict: each entry is {name: [type, options]}
        schema = node_info.get("input") or {}
        ordered_input_names: list[str] = []
        for group in ("required", "optional"):
            entries = schema.get(group) or {}
            # Preserve insertion order (Python 3.7+ dicts do)
            for name in entries.keys():
                ordered_input_names.append(name)

        # Connection inputs come from node["inputs"], widget values from node["widgets_values"]
        connection_input_names = {ent["name"] for ent in (node.get("inputs") or []) if isinstance(ent, dict)}

        inputs: dict[str, Any] = {}

        # Map widget values to widget input names (those that are NOT connection inputs)
        widget_values = node.get("widgets_values") or []
        # Some widget types (e.g. seed with control-after-generate) expand to extra widget values.
        # We do best-effort positional mapping over ordered_input_names that are not connections.
        widget_input_names = [n for n in ordered_input_names if n not in connection_input_names]
        for name, val in zip(widget_input_names, widget_values):
            inputs[name] = val

        # Map connection inputs
        for inp in node.get("inputs") or []:
            name = inp.get("name")
            link_id = inp.get("link")
            if link_id is not None:
                src = links_by_id.get(link_id)
                if src:
                    inputs[name] = [str(src[0]), src[1]]

        api[str(nid)] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": node.get("title") or class_type},
        }

    # Flatten Reroute nodes: any input whose source is a Reroute should be replaced with the Reroute's source.
    # Build a map of Reroute id -> upstream source.
    reroute_map: dict[str, list] = {}
    for node in ui.get("nodes", []):
        if node.get("type") == "Reroute":
            for inp in node.get("inputs") or []:
                if inp.get("link") is not None:
                    src = links_by_id.get(inp["link"])
                    if src:
                        reroute_map[str(node["id"])] = [str(src[0]), src[1]]
    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        for nid, data in api.items():
            for k, v in list(data["inputs"].items()):
                if isinstance(v, list) and len(v) == 2 and v[0] in reroute_map:
                    data["inputs"][k] = reroute_map[v[0]]
                    changed = True
        iterations += 1

    return api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="UI workflow JSON file OR directory")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("-o", "--output", help="output file (single-file mode)")
    ap.add_argument("--inplace", action="store_true", help="overwrite input file with API format")
    args = ap.parse_args()

    print(f"[ui_to_api] fetching node schemas from {args.comfy_url}")
    obj_info = fetch_object_info(args.comfy_url)
    print(f"[ui_to_api] loaded {len(obj_info)} node types")

    target = Path(args.target)
    if target.is_dir():
        files = sorted(target.glob("*.json"))
    else:
        files = [target]

    for f in files:
        with f.open() as fh:
            wf = json.load(fh)
        if not (isinstance(wf, dict) and "nodes" in wf):
            print(f"[skip] {f.name}: already API format")
            continue
        api_wf = convert(wf, obj_info)
        if args.inplace or target.is_dir():
            outp = f
        elif args.output:
            outp = Path(args.output)
        else:
            outp = f.with_name(f.stem + "_api.json")
        with outp.open("w") as fh:
            json.dump(api_wf, fh, indent=2)
        print(f"[ok] {f.name} -> {outp.name}  (nodes={len(api_wf)})")


if __name__ == "__main__":
    main()
