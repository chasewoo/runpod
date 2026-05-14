# runsulphur

Run **Sulphur-2** (LTX-2.3 fine-tune) on RunPod as an HTTP API. No ComfyUI knowledge needed once it's running.

Architecture:
- **ComfyUI** loads Sulphur-2 weights and runs the LTX-2.3 workflow.
- **FastAPI** wraps ComfyUI behind 3 endpoints: `POST /generate` → `GET /status/{id}` → `GET /download/{id}`.
- Models live on a **RunPod Network Volume** mounted at `/workspace` (download once, reuse forever).

---

## 1. One-time RunPod setup

1. **Create a Network Volume**: RunPod console → Storage → New Network Volume. ≥120 GB, in a region that has A100 80GB. Note the region.
2. **Build the image** (locally, or use the GitHub Container Registry):
   ```bash
   docker build -t <yourname>/runsulphur:latest -f docker/Dockerfile .
   docker push <yourname>/runsulphur:latest
   ```
3. **Create a RunPod template**:
   - Container image: `<yourname>/runsulphur:latest`
   - Container disk: 20 GB
   - Volume mount path: `/workspace`
   - Expose HTTP port: `8000`
   - (optional, for debugging ComfyUI UI) Expose HTTP port `8188`
   - Env vars (defaults shown):
     - `SULPHUR_VARIANT=bf16` (`bf16` | `fp8mixed` | `distil`)
     - `SULPHUR_REPO=SulphurAI/Sulphur-2-base`
     - `LTX_REPO=Lightricks/LTX-2.3`
     - `HF_TOKEN=` (only if the repos are gated)
     - `SKIP_MODEL_DOWNLOAD=0`
4. **Deploy a Pod** from that template on **A100 80GB SXM** + your Network Volume. Wait for the first boot to download ~50 GB. Subsequent boots skip downloads (files already on the volume).
5. Open the **HTTP 8000 proxy URL** RunPod shows you. It looks like `https://<podid>-8000.proxy.runpod.net`.

---

## 2. Calling the API

### Quick check
```bash
curl https://<podid>-8000.proxy.runpod.net/health
# {"api":"ok","comfy":"ok"}
```

### Text-to-video
```bash
curl -X POST https://<podid>-8000.proxy.runpod.net/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cinematic shot of a lone astronaut on a red dune at sunset, slow dolly forward",
    "mode": "t2v",
    "variant": "base",
    "width": 768, "height": 512,
    "duration_seconds": 4, "fps": 24,
    "steps": 30, "cfg": 3.0
  }'
# -> {"job_id":"abc...","state":"queued",...}
```

### Image-to-video
```bash
B64=$(base64 -i hero.png)
curl -X POST https://<podid>-8000.proxy.runpod.net/generate \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"the character starts running through rain\",\"mode\":\"i2v\",\"image_b64\":\"$B64\"}"
```

### Poll + download
```bash
JID=abc...
curl https://<podid>-8000.proxy.runpod.net/status/$JID
# when state == "completed":
curl -o out.mp4 https://<podid>-8000.proxy.runpod.net/download/$JID
```

Or use the bundled Python client:
```bash
BASE=https://<podid>-8000.proxy.runpod.net python tests/client_example.py t2v "your prompt here"
```

---

## 3. Request body fields

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | required |
| `negative_prompt` | `"low quality, blurry, distorted"` | |
| `mode` | `t2v` | `t2v` or `i2v` |
| `variant` | `base` | `base` (quality) / `distil` (fast, needs LoRA) |
| `image_b64` / `image_url` | — | required for `i2v` |
| `width` / `height` | 768 / 512 | rounded down to multiples of 32 |
| `duration_seconds` | 4 | converted to LTX-valid frame count `8k+1` |
| `fps` | 24 | |
| `steps` | 30 | 25–40 sweet spot for base; 8–12 for distil |
| `cfg` | 3.0 | LTX likes low CFG (2–4) |
| `seed` | random | int |
| `advanced` | `{}` | per-node overrides, see below |

### Advanced overrides

The server injects user params into the workflow by **node title**. In the source ComfyUI workflow, name the nodes:

- `API_PROMPT` — positive CLIPTextEncode
- `API_NEGATIVE` — negative CLIPTextEncode
- `API_IMAGE` — LoadImage (i2v only)
- `API_DIMS` — empty-latent / EmptyLatentVideo (width, height, length, frame_rate)
- `API_SAMPLER` — sampler node (steps, cfg, seed)
- `API_SAVE` — VHS_VideoCombine (filename_prefix, frame_rate)

If your workflow uses different node titles, either rename them in ComfyUI and re-export "API format", or use the `advanced` field:

```json
{"advanced": {"My Custom Node": {"some_input": 0.42}}}
```

---

## 4. Workflows

On first boot, `scripts/download_models.py` tries to fetch the four official Sulphur workflows into `/workspace/workflows/`:

- `ltx23_t2v_base.json`
- `ltx23_t2v_distil.json`
- `ltx23_i2v_base.json`
- `ltx23_i2v_distil.json`

**You must edit each one in ComfyUI (port 8188 proxy) to tag the nodes with the `API_*` titles above, then export as "API format" and replace the file.** Without tags, the server will queue the workflow with its built-in default values and ignore your request params.

(Once tagged, restart the API — it reloads templates per request, so no rebuild is needed.)

---

## 5. Troubleshooting

- `/health` returns `"comfy":"down"` for ~60 s on cold boot while ComfyUI loads. Wait.
- First `/generate` after boot is slow (model loads into VRAM). Subsequent calls are fast.
- If `ComfyUI rejected workflow` — open ComfyUI UI on port 8188, load the JSON, hit Queue. The Comfy error message is more detailed.
- `bf16` weights need ~50 GB VRAM during inference at high res. Drop dims to 640x384 or switch to `fp8mixed` if OOM on smaller cards.
- HF gated downloads: set `HF_TOKEN` in pod env vars.
