"""Deploy the GLM-OCR worker as a RunPod serverless endpoint.

Usage:
    python scripts/deploy.py --image DOCKERHUB_USER/glm-ocr-runpod:latest

After it succeeds, the endpoint ID is printed and appended to .env as
RUNPOD_ENDPOINT_ID=... so scripts/test_endpoint.py can pick it up.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_BASE = "https://rest.runpod.io/v1"
API_KEY = os.getenv("RUNPOD_API_KEY")
if not API_KEY:
    sys.exit("RUNPOD_API_KEY not set. Put it in .env")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def ensure_registry_auth(name: str, username: str, password: str) -> str:
    """Create (or reuse) a RunPod container registry auth record."""
    r = requests.get(f"{API_BASE}/containerregistryauth", headers=HEADERS, timeout=30)
    r.raise_for_status()
    for item in (r.json() or []):
        if item.get("name") == name:
            print(f"[ok] Reusing registry auth: {item.get('id')}  (name={name})")
            return item["id"]
    body = {"name": name, "username": username, "password": password}
    r = requests.post(f"{API_BASE}/containerregistryauth", headers=HEADERS, json=body, timeout=30)
    if not r.ok:
        sys.exit(f"Registry auth create failed: {r.status_code} {r.text}")
    auth_id = r.json().get("id")
    print(f"[ok] Registry auth created: {auth_id}  (name={name})")
    return auth_id


def create_template(image_name: str, name: str, registry_auth_id: str | None = None) -> str:
    body = {
        "name": name,
        "imageName": image_name,
        "isServerless": True,
        "containerDiskInGb": 25,
        "volumeInGb": 0,
        "env": {
            "MODEL_PATH": "zai-org/GLM-OCR",
        },
        "readme": "GLM-OCR (zai-org/GLM-OCR) serverless worker. Accepts PDFs or images.",
    }
    if registry_auth_id:
        body["containerRegistryAuthId"] = registry_auth_id
    r = requests.post(f"{API_BASE}/templates", headers=HEADERS, json=body, timeout=30)
    if not r.ok:
        sys.exit(f"Template create failed: {r.status_code} {r.text}")
    tpl = r.json()
    tpl_id = tpl.get("id") or tpl.get("templateId")
    print(f"[ok] Template created: {tpl_id}  (name={name})")
    return tpl_id


def create_endpoint(template_id: str, name: str, gpu_types: list[str]) -> str:
    body = {
        "name": name,
        "templateId": template_id,
        "gpuTypeIds": gpu_types,
        "workersMin": 0,
        "workersMax": 1,
        "idleTimeout": 5,
        "flashboot": True,
        "executionTimeoutMs": 600_000,  # 10 min
        "computeType": "GPU",
        # Restrict to a few common DCs so scheduling is fast
        "dataCenterIds": ["US-CA-2", "US-IL-1", "CA-MTL-1", "EU-RO-1"],
    }
    r = requests.post(f"{API_BASE}/endpoints", headers=HEADERS, json=body, timeout=30)
    if not r.ok:
        sys.exit(f"Endpoint create failed: {r.status_code} {r.text}")
    ep = r.json()
    ep_id = ep.get("id") or ep.get("endpointId")
    print(f"[ok] Endpoint created: {ep_id}  (name={name})")
    return ep_id


def persist_endpoint_id(ep_id: str) -> None:
    env_path = ROOT / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    lines = [ln for ln in lines if not ln.startswith("RUNPOD_ENDPOINT_ID=")]
    lines.append(f"RUNPOD_ENDPOINT_ID={ep_id}")
    env_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] Wrote RUNPOD_ENDPOINT_ID to {env_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="ghcr.io/arbourfr/glm-ocr-runpod:latest", help="Docker image")
    ap.add_argument("--name", default="glm-ocr", help="Template + endpoint name prefix")
    ap.add_argument("--ghcr-user", default=None, help="If set, creates RunPod registry auth for ghcr.io with this username (requires GHCR_TOKEN env)")
    ap.add_argument(
        "--gpu",
        nargs="+",
        default=[
            "NVIDIA RTX A4000",
            "NVIDIA RTX 2000 Ada Generation",
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 3090",
        ],
        help="Ordered list of acceptable GPU type IDs (cheapest first).",
    )
    args = ap.parse_args()

    print(f"[deploy] image={args.image}")
    auth_id = None
    if args.ghcr_user:
        token = os.getenv("GHCR_TOKEN")
        if not token:
            sys.exit("GHCR_TOKEN env var required when using --ghcr-user")
        auth_id = ensure_registry_auth(
            name=f"ghcr-{args.ghcr_user}",
            username=args.ghcr_user,
            password=token,
        )
    tpl_id = create_template(args.image, name=f"{args.name}-template", registry_auth_id=auth_id)
    ep_id = create_endpoint(tpl_id, name=args.name, gpu_types=args.gpu)
    persist_endpoint_id(ep_id)
    print()
    print("Next:  python scripts/test_endpoint.py")


if __name__ == "__main__":
    main()
