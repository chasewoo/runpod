"""Thin ComfyUI client: queue prompts, poll history, fetch outputs."""
from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import websockets


COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
WS_BASE = f"ws://{COMFY_HOST}:{COMFY_PORT}"


class ComfyClient:
    def __init__(self, client_id: str | None = None):
        self.client_id = client_id or uuid.uuid4().hex
        self._http = httpx.AsyncClient(base_url=BASE, timeout=60.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def upload_image(self, data: bytes, filename: str) -> str:
        files = {"image": (filename, data, "application/octet-stream")}
        r = await self._http.post("/upload/image", files=files, data={"overwrite": "true"})
        r.raise_for_status()
        return r.json()["name"]

    async def queue_prompt(self, workflow: dict) -> str:
        r = await self._http.post("/prompt", json={"prompt": workflow, "client_id": self.client_id})
        r.raise_for_status()
        return r.json()["prompt_id"]

    async def history(self, prompt_id: str) -> dict | None:
        r = await self._http.get(f"/history/{prompt_id}")
        r.raise_for_status()
        data = r.json()
        return data.get(prompt_id)

    async def fetch_file(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        r = await self._http.get("/view", params={"filename": filename, "subfolder": subfolder, "type": type_})
        r.raise_for_status()
        return r.content

    async def stream_events(self, prompt_id: str) -> AsyncIterator[dict]:
        """Yield {type, progress, done, error, outputs} dicts. Closes when execution completes for prompt_id."""
        url = f"{WS_BASE}/ws?clientId={self.client_id}"
        async with websockets.connect(url, max_size=2**24) as ws:
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                mtype = msg.get("type")
                data = msg.get("data", {})
                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue
                if mtype == "progress":
                    yield {"type": "progress", "value": data.get("value", 0), "max": data.get("max", 1)}
                elif mtype == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        yield {"type": "done"}
                        return
                elif mtype == "execution_error":
                    yield {"type": "error", "error": data.get("exception_message") or json.dumps(data)}
                    return


def find_video_outputs(history_entry: dict) -> list[tuple[str, str]]:
    """Return list of (filename, subfolder) for video outputs in a history entry."""
    out = []
    for node_outputs in (history_entry.get("outputs") or {}).values():
        for key in ("gifs", "videos", "images"):
            for item in node_outputs.get(key, []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                if key == "images" and not fn.lower().endswith((".mp4", ".webm", ".gif", ".webp")):
                    continue
                out.append((fn, item.get("subfolder", "")))
    return out
