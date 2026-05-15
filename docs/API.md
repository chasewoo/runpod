# API Reference

`BASE = https://<pod_id>-8000.proxy.runpod.net` (or `http://127.0.0.1:8000` from inside the pod).

All endpoints are unauthenticated (the proxy URL is the secret — only people with the pod id can hit it).

---

## Endpoints

### `GET /health`
Cheap liveness check; also tells you whether the GPU backend is up.

```bash
curl $BASE/health
```
```json
{"api": "ok", "comfy": "ok"}        // happy path
{"api": "ok", "comfy": "down"}      // FastAPI is up but ComfyUI isn't ready yet
```

`comfy: "down"` for ~1 min after pod start is normal; wait then retry.

---

### `GET /workflows`
List the workflow JSONs the server has loaded.

```bash
curl $BASE/workflows
```
```json
{"workflows": ["ltx23_i2v_base.json", "ltx23_i2v_distil.json",
               "ltx23_t2v_base.json", "ltx23_t2v_distil.json"]}
```

---

### `POST /generate`
Submit one video generation job. Returns immediately with a `job_id` and `state=queued`.

**Request body** (JSON):

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | **required** |
| `negative_prompt` | string | `"low quality, blurry, distorted"` | |
| `mode` | `"t2v"` \| `"i2v"` | `"t2v"` | i2v works today; t2v needs the README §5 workflow patch |
| `variant` | `"base"` \| `"distil"` | `"base"` | `distil` is ~4× faster, 8 steps instead of 30 |
| `image_b64` | string (base64 PNG/JPG) | — | required when `mode="i2v"` |
| `image_url` | string | — | alternative to `image_b64` (server fetches) |
| `width` | int (256–1920) | 768 | rounded **down** to multiple of 32 |
| `height` | int (256–1920) | 512 | same |
| `duration_seconds` | float (0.1–20) | 4 | converted to LTX-valid frame count `8k+1` |
| `fps` | int (8–60) | 24 | |
| `steps` | int (1–100) | 30 | `25–40` for base, `8–12` for distil |
| `cfg` | float (0–20) | 3.0 | LTX likes low cfg (2–4) |
| `seed` | int | random | |
| `advanced` | object | `{}` | per-node overrides, see below |

The upscaler in the workflow doubles dims on output, so a `704×480` request produces a `1408×960` mp4.

**Response**:
```json
{
  "job_id": "c29feae8a36b4bd99b45e53b5a11d676",
  "state": "queued",
  "progress": 0.0,
  "error": null,
  "output_url": null
}
```

**Errors**:
- `400 Bad Request` — invalid mode, missing image for i2v, unknown workflow file
- `502 Bad Gateway` — ComfyUI rejected the workflow (model file missing, validation error). The body contains the ComfyUI error message.
- RunPod proxy itself may return 502 for ~30 s right after pod start while it warms up. Retry.

---

### `GET /status/{job_id}`
Poll the job state. Cheap, call as often as you like.

```bash
curl $BASE/status/$JID
```
```json
{
  "job_id": "c29feae8a36b4bd99b45e53b5a11d676",
  "state": "running",         // queued | running | completed | error
  "progress": 0.625,          // 0.0 .. 1.0, comes from ComfyUI ws events
  "error": null,              // string when state=="error"
  "output_url": null          // "/download/{job_id}" once state=="completed"
}
```

Timing (A100 SXM, default params):
- `queued → running`: ~1 s
- model load on cold pod: 60–180 s (one-time)
- sampling: ~5 s per step × `steps`
- post (VAE decode, upscale, video encode): 30–60 s

---

### `GET /download/{job_id}`
Stream the output mp4. Only valid when `state == "completed"`.

```bash
curl -o out.mp4 $BASE/download/$JID
```

The file is also persisted at `/workspace/outputs/<job_id>.mp4` on the pod. After pod termination, those mp4s remain on the volume until you delete them.

Returns `404` if the job is unknown or not yet complete.

---

## Examples

### curl — image-to-video round trip

```bash
BASE=https://<pod_id>-8000.proxy.runpod.net
B64=$(base64 -i input.png | tr -d '\n')

JID=$(curl -s -X POST $BASE/generate \
  -H 'Content-Type: application/json' \
  -d "{
    \"prompt\":\"the character starts running through rain, cinematic\",
    \"mode\":\"i2v\",
    \"variant\":\"distil\",
    \"width\":704, \"height\":480,
    \"duration_seconds\":3, \"fps\":24,
    \"steps\":8, \"cfg\":2.5,
    \"image_b64\":\"$B64\"
  }" | jq -r .job_id)

echo "job: $JID"

# poll
while :; do
  S=$(curl -s $BASE/status/$JID)
  echo "$S"
  STATE=$(echo "$S" | jq -r .state)
  [ "$STATE" = "completed" ] && break
  [ "$STATE" = "error" ]    && exit 1
  sleep 5
done

curl -o out.mp4 $BASE/download/$JID
```

### Python client (sync, no deps beyond `httpx`)

```python
import base64, time, httpx, pathlib

BASE = "https://<pod_id>-8000.proxy.runpod.net"

def generate(prompt, image_path=None, **opts):
    payload = {"prompt": prompt, **opts}
    if image_path:
        payload["mode"] = "i2v"
        payload["image_b64"] = base64.b64encode(pathlib.Path(image_path).read_bytes()).decode()
    else:
        payload["mode"] = "t2v"

    with httpx.Client(base_url=BASE, timeout=600) as c:
        jid = c.post("/generate", json=payload).raise_for_status().json()["job_id"]
        while True:
            s = c.get(f"/status/{jid}").raise_for_status().json()
            print(f"{s['state']:9s}  {s['progress']*100:5.1f}%")
            if s["state"] == "completed":
                break
            if s["state"] == "error":
                raise RuntimeError(s["error"])
            time.sleep(5)

        out = pathlib.Path(f"{jid}.mp4")
        with c.stream("GET", f"/download/{jid}") as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return out

# usage
generate(
    "a quiet bookstore on a rainy afternoon, slow dolly forward",
    image_path="bookstore.png",
    variant="distil", width=704, height=480,
    duration_seconds=3, steps=8, cfg=2.5, seed=42,
)
```

The repo ships `tests/client_example.py` — same idea, runnable directly:

```bash
BASE=$BASE python tests/client_example.py i2v "your prompt" path/to/image.png
```

### Async / parallel jobs

`/generate` returns immediately; submit many, poll each `/status/{id}` independently. Server processes them serially because ComfyUI has one GPU queue.

```python
jids = []
for p in prompts:
    jids.append(c.post("/generate", json={"prompt": p, ...}).json()["job_id"])

# later
mp4s = [download(j) for j in jids]
```

---

## `advanced`: per-node overrides

The server fills the workflow by **node title**. Each shipped workflow has these tagged nodes:

| Tag | What it controls |
|---|---|
| `API_PROMPT` | positive `CLIPTextEncode.text` |
| `API_NEGATIVE` | negative `CLIPTextEncode.text` |
| `API_DIMS` | `EmptyLTXVLatentVideo.width/height/length/frame_rate` |
| `API_SAMPLER` | `KSampler.steps/cfg/seed` (or `noise_seed` for advanced samplers) |
| `API_IMAGE` | `LoadImage.image` (i2v only) |
| `API_SAVE` | `VHS_VideoCombine.filename_prefix/frame_rate` |

If you need to override an input the API doesn't expose, send it via `advanced`:

```json
{
  "prompt": "...",
  "advanced": {
    "API_SAMPLER": {"sampler_name": "euler_ancestral"},
    "PathchSageAttentionKJ": {"sage_attention": "sageattn_qk_int8_pv_fp8_cuda"}
  }
}
```

Keys are node `_meta.title` strings; values are `{input_name: value}` maps. Values can be either literals or `[src_node_id, src_slot]` for connection refs.

---

## Failure modes

| You see | What it means | Fix |
|---|---|---|
| `502` on first call after deploy | RunPod proxy hasn't registered the pod yet | wait 30 s, retry |
| `{"comfy":"down"}` on `/health` | ComfyUI still loading models from MooseFS | wait, the first cold load is ~3 min |
| `400` "i2v mode requires image_b64 or image_url" | you sent mode i2v without an image | provide one or switch to mode t2v |
| `502` with body `ComfyUI rejected workflow: ...missing_node_type` | a custom node pack isn't installed | bootstrap.sh installs all known ones; ssh in and `git pull` plus restart |
| `state="error"` with `ComfyUI returned no video outputs` | the sampler ran but `VHS_VideoCombine` failed silently | check `/workspace/logs/comfyui.log` for the trace |
| `state="error"` with CUDA OOM | sampling went past 80 GB VRAM | drop dims (`width`/`height`) or `duration_seconds` |
