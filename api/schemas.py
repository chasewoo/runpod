from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Mode = Literal["t2v", "i2v"]
# base    LTX 2.3 + Sulphur LoRA + LTX distill LoRAs, 30 sampling steps
# distil  same model stack, 8 sampling steps (uses the distill LoRAs)
# full    sulphur_dev_bf16 standalone, no LoRA stack — purest Sulphur, 8 steps
Variant = Literal["base", "distil", "full"]


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted"
    mode: Mode = "t2v"
    variant: Variant = "base"

    # Image input (i2v). Either inline base64 or URL. Ignored for t2v.
    image_b64: Optional[str] = None
    image_url: Optional[str] = None

    # Clarity / dims. LTX likes multiples of 32.
    width: int = Field(default=768, ge=256, le=1920)
    height: int = Field(default=512, ge=256, le=1920)

    # Duration. fps * seconds + 1, rounded to nearest LTX-valid frame count.
    duration_seconds: float = Field(default=4.0, gt=0.1, le=20.0)
    fps: int = Field(default=24, ge=8, le=60)

    # Sampling
    steps: int = Field(default=30, ge=1, le=100)
    cfg: float = Field(default=3.0, ge=0.0, le=20.0)
    seed: Optional[int] = None

    # Free-form advanced overrides. Map of node title -> {input_name: value}.
    advanced: Optional[dict] = None


class JobState(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "error"]
    progress: float = 0.0
    error: Optional[str] = None
    output_url: Optional[str] = None
