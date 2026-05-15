# runsulphur

Run **Sulphur-2** (LTX-2.3 fine-tune by SulphurAI) on RunPod as an HTTP API. No ComfyUI knowledge needed once it's up.

## What you get

- `POST /generate` → returns `{job_id, state: "queued"}`
- `GET /status/{id}` → `{state, progress, output_url, error}`
- `GET /download/{id}` → mp4 (1408×960 by default after the LTX 2-stage upscale)
- ComfyUI Web UI also exposed on port `8188` if you want to inspect or run things manually

The first deploy downloads ~130 GB of models onto a RunPod Network Volume. Subsequent pod starts reuse the volume and skip downloads — back online in ~5 minutes.

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌──────────────┐
│  Your code  │────▶│ FastAPI (8000) │────▶│ ComfyUI (8188)│──▶ GPU
│ /generate   │     │ api/main.py    │     │ + custom nodes│
│ /status     │◀────│ in-memory jobs │◀────│ + LTX 2.3 fp8 │
│ /download   │     │ ws progress    │     │ + Sulphur LoRA│
└─────────────┘     └────────────────┘     └──────────────┘
                            ▲                      ▲
                            └────── /workspace ────┘  (RunPod Network Volume)
                                    ├── ComfyUI/             cloned
                                    ├── models/              ~130 GB
                                    ├── workflows/           4 API-format JSONs
                                    └── runsulphur/          this repo
```

The Sulphur workflows ship as UI-format JSON with a stack of LTX 2.3 base + distill LoRAs + Sulphur LoRA. `scripts/bootstrap.sh` handles all of: cloning ComfyUI, installing the 10+ custom-node packs the workflows reference, downloading the model files from HuggingFace (Sulphur, Lightricks LTX-2.3-fp8, Pavpif Gemma encoder), patching the workflows to match local filenames, and starting both services.

---

## 1. One-time setup

### 1.1 RunPod account

1. Create an API key at https://www.runpod.io/console/user/settings → API Keys (read/write).
2. Add your SSH public key (`ssh-keygen -t ed25519 -f .ssh/runpod -N ""`) at https://www.runpod.io/console/user/settings → SSH Keys, **and** pass it as the `PUBLIC_KEY` env var when deploying — the deploy script does both for you.
3. Create a Network Volume in a region that stocks A100 SXM 80GB:
   ```
   Console → Storage → Network Volumes → New
   Datacenter:  US-MO-1            ← A100 SXM stock is reliable here
   Size:        150 GB              ← fits ~130 GB models + buffer
   Name:        sulphur2-vol
   ```
   Copy the Volume ID it shows you.

### 1.2 HuggingFace

1. Generate a Read token at https://huggingface.co/settings/tokens.
2. Open https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized and click **Agree and access repository** (Google's Gemma license; one-time per HF account).

### 1.3 Local `.env`

This file is gitignored. Create it at repo root:

```dotenv
RUNPOD_API_KEY=...
RUNPOD_VOLUME_ID=...      # the ID from step 1.1.3
HF_TOKEN=hf_...
```

### 1.4 Python local helpers

```bash
python -m venv .venv && .venv/bin/pip install httpx
```

---

## 2. Deploy

```bash
set -a; source .env; set +a

# Deploy an A100 SXM 80GB pod, mount the volume, run bootstrap.sh
.venv/bin/python scripts/deploy_runpod.py deploy \
  --repo-url https://github.com/chasewoo/runpod.git \
  --volume-id "$RUNPOD_VOLUME_ID" \
  --gpu "A100 SXM" \
  --hf-token "$HF_TOKEN" \
  --pubkey .ssh/runpod.pub
```

The script prints the pod id. Watch it come up:

```bash
.venv/bin/python scripts/deploy_runpod.py wait <pod_id>
```

When `wait` reports `API healthy`, you're ready. URLs:

- API: `https://<pod_id>-8000.proxy.runpod.net`
- ComfyUI Web: `https://<pod_id>-8188.proxy.runpod.net`

**First-ever boot**: ~30–60 min while it downloads ~130 GB of models (limited by HuggingFace per-token bandwidth).
**Subsequent boots**: ~5 minutes — bootstrap detects existing files and only installs pip deps.

---

## 3. Calling the API

Full reference with all endpoints, params, advanced overrides, error table, and a Python client: **[docs/API.md](docs/API.md)**.

The short version:

### Health
```bash
curl https://<pod_id>-8000.proxy.runpod.net/health
# {"api":"ok","comfy":"ok"}
```

### Image-to-video (works today)
```bash
B64=$(base64 -i input.png | tr -d '\n')
curl -X POST https://<pod_id>-8000.proxy.runpod.net/generate \
  -H 'Content-Type: application/json' \
  -d "{
    \"prompt\":\"the character starts walking forward, cinematic lighting\",
    \"mode\":\"i2v\",
    \"variant\":\"distil\",
    \"width\":704, \"height\":480,
    \"duration_seconds\":3, \"fps\":24,
    \"steps\":8, \"cfg\":2.5,
    \"image_b64\":\"$B64\"
  }"
# -> {"job_id":"abc...","state":"queued"}
```

### Poll + download
```bash
JID=abc...
curl https://<pod_id>-8000.proxy.runpod.net/status/$JID
# when state == "completed":
curl -o out.mp4 https://<pod_id>-8000.proxy.runpod.net/download/$JID
```

Or the bundled Python client:
```bash
BASE=https://<pod_id>-8000.proxy.runpod.net python tests/client_example.py i2v "your prompt" input.png
```

### Request body fields

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | required |
| `negative_prompt` | `"low quality, blurry, distorted"` | |
| `mode` | `i2v` | `t2v` workflows still need a manual `LoadImage` patch — see Known issues |
| `variant` | `distil` | `base` (quality, 30 steps) / `distil` (fast, 8 steps + LoRA stack) |
| `image_b64` / `image_url` | — | required for i2v |
| `width` / `height` | 704 / 480 | upscaler turns it into 1408×960 |
| `duration_seconds` | 3 | LTX rounds to `8k+1` frames |
| `fps` | 24 | |
| `steps` | 8 (distil) / 30 (base) | |
| `cfg` | 2.5 | LTX likes low CFG (2–4) |
| `seed` | random | |
| `advanced` | `{}` | per-node overrides, see workflow_builder.py |

---

## 4. Day-2 operations

### Stop the pod (keep the volume)
```bash
.venv/bin/python scripts/deploy_runpod.py terminate <pod_id>
```
Pod is gone; volume keeps the 130 GB of models. Storage cost ≈ $0.35/day for 150 GB.

### Bring it back later
Re-run the `deploy` command from §2. Bootstrap sees the existing volume contents and skips downloads.

### Inspect on the pod
```bash
ssh -i .ssh/runpod -p <ssh_port> root@<ssh_ip>
# /workspace/logs/   bootstrap + comfyui logs
# /workspace/outputs/ generated mp4s
# /workspace/models/  the model files
```
The pod's SSH host/port shows in `deploy_runpod.py status <pod_id>` output.

---

## 5. Known issues

**t2v workflow needs a manual patch.** Sulphur's shipped `ltx23_t2v_*.json` reuses i2v nodes (`LTXVImgToVideoInplace`) wired into the output graph; ComfyUI rejects them as t2v because the `image` input is unconnected. To make t2v work, open the workflow in the ComfyUI Web UI, drop in an `EmptyImage` node (width/height/batch=1), connect it to the `image` input of nodes 14 and 22, save (API Format) back to `/workspace/workflows/ltx23_t2v_distil.json`. The i2v workflows work without patching.

**ComfyUI startup is slow on MooseFS.** The RunPod network volume is MooseFS-backed; first cold load of the 22 GB LTX-2.3 fp8 ckpt takes ~3 minutes. Subsequent jobs reuse the in-VRAM model.

**Audio output is on.** Sulphur 2 / LTX 2.3 always render audio as well; expect the output mp4 to have an AAC stream alongside H.264.

---

## 6. Project layout

```
api/
  main.py              FastAPI app
  comfy_client.py      ComfyUI client (queue + ws progress + /view)
  workflow_builder.py  Inject prompt/dims/sampler into workflow by API_* node titles
  schemas.py           Pydantic request/response models
scripts/
  bootstrap.sh         One-shot pod boot (apt, ComfyUI, custom nodes, models, services)
  download_models.py   Idempotent HuggingFace pulls
  setup_workflows.py   Patch shipped JSONs (filename rewrites, sage_attention defaults)
  ui_to_api.py         Convert ComfyUI UI-format JSON to API format
  autotag_workflows.py Tag nodes with API_PROMPT / API_DIMS / etc.
  deploy_runpod.py     RunPod GraphQL deploy/status/stop/terminate
tests/
  client_example.py    Minimal client (submit + poll + download)
```

---

## 7. Costs (May 2026 sticker)

| Item | Rate |
|---|---|
| RunPod A100 SXM 80GB | $1.39/h on-demand |
| RunPod Network Volume | $0.07/GB/month → ~$0.35/day for 150 GB |
| HuggingFace bandwidth | free (rate-limited per token) |

Typical workflow: deploy → run jobs (`$1.39/h` while pod up) → terminate → volume keeps models cheaply for next time.
