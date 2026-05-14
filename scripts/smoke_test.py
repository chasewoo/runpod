"""Smoke test against a live deployed pod. Submits a t2v job, polls, downloads mp4.

Usage:
    BASE=https://<pod>-8000.proxy.runpod.net python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

BASE = os.environ.get("BASE")
if not BASE:
    print("set BASE=https://<podid>-8000.proxy.runpod.net", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    payload = {
        "prompt": os.environ.get("PROMPT",
            "a serene mountain lake at sunrise, mist rolling over the water, slow cinematic camera move"),
        "mode": "t2v",
        "variant": os.environ.get("VARIANT", "base"),
        "width": int(os.environ.get("WIDTH", "704")),
        "height": int(os.environ.get("HEIGHT", "480")),
        "duration_seconds": float(os.environ.get("DUR", "3")),
        "fps": int(os.environ.get("FPS", "24")),
        "steps": int(os.environ.get("STEPS", "25")),
        "cfg": float(os.environ.get("CFG", "3.0")),
        "seed": 42,
    }
    with httpx.Client(base_url=BASE, timeout=600.0) as c:
        h = c.get("/health").json()
        print("health:", h)
        if h.get("comfy") != "ok":
            print("ComfyUI not ready yet")
            return 1

        r = c.post("/generate", json=payload)
        r.raise_for_status()
        job = r.json()
        jid = job["job_id"]
        print("job:", jid)

        t0 = time.time()
        while True:
            s = c.get(f"/status/{jid}").json()
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.1f}s] state={s['state']} progress={s['progress']:.2f}")
            if s["state"] == "completed":
                break
            if s["state"] == "error":
                print("error:", s["error"])
                return 2
            time.sleep(5)

        out = Path(f"smoke_{jid}.mp4")
        with c.stream("GET", f"/download/{jid}") as resp:
            resp.raise_for_status()
            with out.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        print(f"saved: {out} ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
