"""Deploy or manage a Sulphur-2 pod on RunPod via GraphQL.

Usage:
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py deploy   --repo-url https://github.com/<you>/runsulphur
                                                                   [--volume-id <id>]
                                                                   [--gpu 'NVIDIA A100 80GB PCIe']
                                                                   [--hf-token <token>]
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py wait     <pod_id>
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py status   <pod_id>
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py logs     <pod_id>          # not supported via GQL; prints SSH hints
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py stop     <pod_id>
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py terminate <pod_id>
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py list-gpus
    RUNPOD_API_KEY=...    python scripts/deploy_runpod.py list-volumes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx


GQL_URL = "https://api.runpod.io/graphql"

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"


def gql(query: str, variables: dict | None = None) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("error: set RUNPOD_API_KEY", file=sys.stderr)
        sys.exit(2)
    r = httpx.post(
        f"{GQL_URL}?api_key={key}",
        json={"query": query, "variables": variables or {}},
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]


def list_gpus() -> None:
    q = """query { gpuTypes { id displayName memoryInGb secureCloud communityCloud } }"""
    d = gql(q)
    for g in d["gpuTypes"]:
        print(f"{g['id']:40s}  {g['displayName']:40s}  {g['memoryInGb']}GB  secure={g['secureCloud']} comm={g['communityCloud']}")


def list_volumes() -> None:
    q = """query { myself { networkVolumes { id name size dataCenterId } } }"""
    d = gql(q)
    for v in d["myself"]["networkVolumes"]:
        print(f"{v['id']}  name={v['name']!r}  size={v['size']}GB  dc={v['dataCenterId']}")


def find_gpu_id(display_name: str) -> str:
    q = """query { gpuTypes { id displayName } }"""
    d = gql(q)
    matches = [g for g in d["gpuTypes"] if g["displayName"] == display_name]
    if not matches:
        # loose match
        matches = [g for g in d["gpuTypes"] if display_name.lower() in g["displayName"].lower()]
    if not matches:
        raise SystemExit(f"no gpu type matches {display_name!r}. Run `list-gpus` to inspect.")
    return matches[0]["id"]


def deploy(args: argparse.Namespace) -> None:
    gpu_id = find_gpu_id(args.gpu)
    print(f"[deploy] gpu_type_id={gpu_id}")

    env_pairs = [
        {"key": "SULPHUR_VARIANT", "value": args.variant},
        {"key": "REPO_DIR", "value": "/workspace/runsulphur"},
        {"key": "REPO_URL", "value": args.repo_url},
        {"key": "REPO_BRANCH", "value": args.branch},
    ]
    if args.hf_token:
        env_pairs.append({"key": "HF_TOKEN", "value": args.hf_token})
    if args.pubkey:
        pubkey = Path(args.pubkey).read_text().strip()
        env_pairs.append({"key": "PUBLIC_KEY", "value": pubkey})

    variables: dict[str, Any] = {
        "input": {
            "name": args.name,
            "imageName": args.image,
            "gpuCount": 1,
            "gpuTypeId": gpu_id,
            "containerDiskInGb": 40,
            "volumeInGb": 0 if args.volume_id else args.local_volume_gb,
            "volumeMountPath": "/workspace",
            "ports": "8000/http,8188/http,22/tcp",
            "env": env_pairs,
            "cloudType": "SECURE",
        }
    }
    if args.volume_id:
        variables["input"]["networkVolumeId"] = args.volume_id
        # When using a network volume, RunPod requires no local volume.
        variables["input"]["volumeInGb"] = 0

    q = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) {
        id imageName machineId desiredStatus
      }
    }
    """
    d = gql(q, variables)
    pod = d["podFindAndDeployOnDemand"]
    if not pod:
        raise SystemExit("deployment returned null — out of stock for that GPU+region. Try a different --gpu or volume region.")
    print(f"[deploy] pod_id={pod['id']} status={pod['desiredStatus']}")
    print(f"[hint] watch: python scripts/deploy_runpod.py wait {pod['id']}")


def get_pod(pod_id: str) -> dict:
    q = """
    query Pod($id: String!) {
      pod(input: {podId: $id}) {
        id name desiredStatus
        runtime { uptimeInSeconds ports { ip privatePort publicPort isIpPublic type } }
        machine { podHostId }
      }
    }
    """
    return gql(q, {"id": pod_id})["pod"]


def _proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def status(pod_id: str) -> None:
    p = get_pod(pod_id)
    print(json.dumps(p, indent=2))
    print(f"api  : {_proxy_url(pod_id, 8000)}")
    print(f"comfy: {_proxy_url(pod_id, 8188)}")


def wait(pod_id: str, *, timeout_s: int = 1500) -> None:
    api_url = _proxy_url(pod_id, 8000)
    print(f"[wait] api will be at {api_url}")
    deadline = time.time() + timeout_s
    last_state = ""
    while time.time() < deadline:
        try:
            p = get_pod(pod_id)
            state = p.get("desiredStatus", "?")
            uptime = (p.get("runtime") or {}).get("uptimeInSeconds")
            if state != last_state:
                print(f"[wait] pod state={state} uptime={uptime}")
                last_state = state
            if state == "RUNNING":
                try:
                    r = httpx.get(f"{api_url}/health", timeout=10.0)
                    if r.status_code == 200 and r.json().get("api") == "ok":
                        print(f"[wait] API healthy: {r.json()}")
                        return
                except Exception:
                    pass
        except Exception as e:
            print(f"[wait] pod query failed: {e}")
        time.sleep(15)
    raise SystemExit("timed out waiting for pod")


def stop(pod_id: str) -> None:
    q = """mutation Stop($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }"""
    print(gql(q, {"id": pod_id}))


def terminate(pod_id: str) -> None:
    q = """mutation Term($id: String!) { podTerminate(input: {podId: $id}) }"""
    print(gql(q, {"id": pod_id}))


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("deploy")
    pd.add_argument("--repo-url", required=True)
    pd.add_argument("--branch", default="main")
    pd.add_argument("--volume-id", default=None, help="RunPod network volume id (recommended).")
    pd.add_argument("--local-volume-gb", type=int, default=80, help="Used only when --volume-id is omitted")
    pd.add_argument("--gpu", default=DEFAULT_GPU)
    pd.add_argument("--image", default=DEFAULT_IMAGE)
    pd.add_argument("--variant", default="bf16", choices=["bf16", "fp8mixed", "distil"])
    pd.add_argument("--name", default="sulphur2-api")
    pd.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    pd.add_argument("--pubkey", default=".ssh/runpod.pub", help="Path to SSH public key for root access")

    for name, fn in (("wait", wait), ("status", status), ("stop", stop), ("terminate", terminate)):
        s = sub.add_parser(name)
        s.add_argument("pod_id")

    sub.add_parser("list-gpus")
    sub.add_parser("list-volumes")

    args = p.parse_args()
    if args.cmd == "deploy":
        deploy(args)
    elif args.cmd == "wait":
        wait(args.pod_id)
    elif args.cmd == "status":
        status(args.pod_id)
    elif args.cmd == "stop":
        stop(args.pod_id)
    elif args.cmd == "terminate":
        terminate(args.pod_id)
    elif args.cmd == "list-gpus":
        list_gpus()
    elif args.cmd == "list-volumes":
        list_volumes()
    return 0


if __name__ == "__main__":
    sys.exit(main())
