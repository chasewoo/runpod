"""FastAPI server. Wraps ComfyUI behind a simple async job API."""
from __future__ import annotations

import asyncio
import base64
import os
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .comfy_client import ComfyClient, find_video_outputs
from .schemas import GenerateRequest, JobState
from . import workflow_builder as wb


WORKFLOWS_DIR = Path(os.environ.get("WORKFLOWS_DIR", "/workspace/workflows"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/workspace/outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# LTX latent stride: frame counts must satisfy (N-1) % 8 == 0 (LTX convention).
LTX_FRAME_MULTIPLE = 8


def _round_frames(seconds: float, fps: int) -> int:
    raw = max(1, round(seconds * fps))
    # round to (k*8 + 1)
    n = ((raw - 1) // LTX_FRAME_MULTIPLE) * LTX_FRAME_MULTIPLE + 1
    return max(9, n)


def _round_dim(x: int) -> int:
    return max(256, (x // 32) * 32)


def _workflow_path(mode: str, variant: str) -> Path:
    name = f"ltx23_{mode}_{variant}.json"
    p = WORKFLOWS_DIR / name
    if not p.exists():
        # also accept underscored variants written by hand
        alt = WORKFLOWS_DIR / f"{mode}_{variant}.json"
        if alt.exists():
            return alt
        raise FileNotFoundError(f"workflow not found: {p}. Place a tagged ComfyUI API-format JSON there.")
    return p


app = FastAPI(title="Sulphur-2 API", version="0.1.0")

# job_id -> JobState
JOBS: dict[str, JobState] = {}
# job_id -> Path to mp4
OUTPUT_FILES: dict[str, Path] = {}


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"http://{os.environ.get('COMFY_HOST','127.0.0.1')}:{os.environ.get('COMFY_PORT','8188')}/system_stats")
            comfy_ok = r.status_code == 200
        except Exception:
            comfy_ok = False
    return {"api": "ok", "comfy": "ok" if comfy_ok else "down"}


@app.post("/generate", response_model=JobState)
async def generate(req: GenerateRequest):
    try:
        wf_path = _workflow_path(req.mode, req.variant)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    template = wb.load(wf_path)

    width = _round_dim(req.width)
    height = _round_dim(req.height)
    num_frames = _round_frames(req.duration_seconds, req.fps)

    client = ComfyClient()
    image_filename: Optional[str] = None
    if req.mode == "i2v":
        img_bytes: bytes | None = None
        if req.image_b64:
            img_bytes = base64.b64decode(req.image_b64)
        elif req.image_url:
            async with httpx.AsyncClient(timeout=60.0) as c:
                rr = await c.get(req.image_url)
                rr.raise_for_status()
                img_bytes = rr.content
        if not img_bytes:
            raise HTTPException(status_code=400, detail="i2v mode requires image_b64 or image_url")
        image_filename = await client.upload_image(img_bytes, f"input_{uuid.uuid4().hex}.png")

    job_id = uuid.uuid4().hex
    save_prefix = f"sulphur/{job_id}"

    workflow = wb.build(
        template,
        prompt=req.prompt,
        negative=req.negative_prompt,
        width=width, height=height,
        num_frames=num_frames, fps=req.fps,
        steps=req.steps, cfg=req.cfg, seed=req.seed,
        image_filename=image_filename,
        save_prefix=save_prefix,
        advanced=req.advanced,
    )

    try:
        prompt_id = await client.queue_prompt(workflow)
    except httpx.HTTPStatusError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"ComfyUI rejected workflow: {e.response.text}")

    state = JobState(job_id=job_id, state="queued")
    JOBS[job_id] = state

    asyncio.create_task(_run_job(client, job_id, prompt_id))
    return state


async def _run_job(client: ComfyClient, job_id: str, prompt_id: str) -> None:
    state = JOBS[job_id]
    state.state = "running"
    try:
        async for evt in client.stream_events(prompt_id):
            if evt["type"] == "progress":
                state.progress = evt["value"] / max(1, evt["max"])
            elif evt["type"] == "error":
                state.state = "error"
                state.error = evt["error"]
                return
            elif evt["type"] == "done":
                break

        # Fetch output(s)
        entry = await client.history(prompt_id)
        if not entry:
            state.state = "error"
            state.error = "no history entry returned"
            return
        outputs = find_video_outputs(entry)
        if not outputs:
            state.state = "error"
            state.error = "ComfyUI returned no video outputs"
            return
        # Save first output to OUTPUTS_DIR/job_id.<ext>
        filename, subfolder = outputs[0]
        data = await client.fetch_file(filename, subfolder=subfolder, type_="output")
        ext = Path(filename).suffix or ".mp4"
        local = OUTPUTS_DIR / f"{job_id}{ext}"
        local.write_bytes(data)
        OUTPUT_FILES[job_id] = local
        state.output_url = f"/download/{job_id}"
        state.progress = 1.0
        state.state = "completed"
    except Exception as e:
        state.state = "error"
        state.error = repr(e)
    finally:
        await client.aclose()


@app.get("/status/{job_id}", response_model=JobState)
async def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return JOBS[job_id]


@app.get("/download/{job_id}")
async def download(job_id: str):
    path = OUTPUT_FILES.get(job_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="output not ready")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/workflows")
async def list_workflows():
    if not WORKFLOWS_DIR.exists():
        return JSONResponse({"workflows": []})
    return {"workflows": sorted(p.name for p in WORKFLOWS_DIR.glob("*.json"))}
