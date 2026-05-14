"""Minimal client. Submit a job, poll status, download mp4.

Usage:
    BASE=https://<podid>-8000.proxy.runpod.net python tests/client_example.py t2v "a neon cat"
    BASE=...                                         python tests/client_example.py i2v "the cat starts running" path/to/img.png
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    prompt = sys.argv[2]
    payload = {
        "prompt": prompt,
        "mode": mode,
        "variant": os.environ.get("VARIANT", "base"),
        "width": int(os.environ.get("WIDTH", "768")),
        "height": int(os.environ.get("HEIGHT", "512")),
        "duration_seconds": float(os.environ.get("DUR", "4")),
        "fps": int(os.environ.get("FPS", "24")),
        "steps": int(os.environ.get("STEPS", "30")),
        "cfg": float(os.environ.get("CFG", "3.0")),
    }
    if mode == "i2v":
        if len(sys.argv) < 4:
            print("i2v needs image path")
            sys.exit(1)
        img = Path(sys.argv[3]).read_bytes()
        payload["image_b64"] = base64.b64encode(img).decode()

    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        r = c.post("/generate", json=payload)
        r.raise_for_status()
        job = r.json()
        jid = job["job_id"]
        print("job:", jid)

        while True:
            r = c.get(f"/status/{jid}")
            r.raise_for_status()
            s = r.json()
            print(f"  state={s['state']} progress={s['progress']:.2f}", end="\r")
            if s["state"] == "completed":
                break
            if s["state"] == "error":
                print("\nerror:", s["error"])
                sys.exit(2)
            time.sleep(2)

        print("\ndownloading...")
        with c.stream("GET", f"/download/{jid}") as resp:
            resp.raise_for_status()
            out = Path(f"{jid}.mp4")
            with out.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
            print("saved:", out)


if __name__ == "__main__":
    main()
