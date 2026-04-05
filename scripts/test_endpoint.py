"""Send a PDF to the deployed GLM-OCR endpoint and save the OCR result.

Usage:
    python scripts/test_endpoint.py                         # uses samples/acte_10175367.pdf
    python scripts/test_endpoint.py path/to/file.pdf
    python scripts/test_endpoint.py path/to/file.pdf --pages 1 2

Reads RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID from .env.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
if not API_KEY or not ENDPOINT_ID:
    sys.exit("Need RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env (run scripts/deploy.py first)")

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def submit_job(pdf_path: Path, pages: list[int] | None, dpi: int, max_new_tokens: int) -> str:
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()
    payload: dict = {"input": {"pdf_base64": pdf_b64, "dpi": dpi, "max_new_tokens": max_new_tokens}}
    if pages:
        payload["input"]["pages"] = pages

    r = requests.post(RUN_URL, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"[submit] job_id={job_id}")
    return job_id


def poll(job_id: str, timeout_s: int = 900) -> dict:
    url = f"{STATUS_URL}/{job_id}"
    start = time.time()
    last_status = None
    while True:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status != last_status:
            print(f"[poll] status={status}  ({time.time() - start:.1f}s)")
            last_status = status
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return data
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(ROOT / "samples/acte_10175367.pdf"))
    ap.add_argument("--pages", nargs="+", type=int, help="1-indexed page filter")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Not found: {pdf_path}")

    print(f"[test] pdf={pdf_path}  size={pdf_path.stat().st_size} bytes")
    job_id = submit_job(pdf_path, args.pages, args.dpi, args.max_new_tokens)
    result = poll(job_id)

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / f"{pdf_path.stem}.json"
    raw_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[out] raw → {raw_path}")

    if result.get("status") != "COMPLETED":
        sys.exit(f"Job failed: {json.dumps(result, indent=2)[:1000]}")

    output = result.get("output", {})
    if "error" in output:
        sys.exit(f"Worker error: {output['error']}")

    md_path = out_dir / f"{pdf_path.stem}.md"
    with md_path.open("w") as f:
        f.write(f"# OCR — {pdf_path.name}\n\n")
        f.write(f"Model: {output.get('model')}  |  Pages: {output.get('num_pages')}  |  Elapsed: {output.get('elapsed_s')}s\n\n")
        for p in output.get("pages", []):
            f.write(f"\n---\n\n## Page {p['page']}\n\n{p['text']}\n")
    print(f"[out] markdown → {md_path}")

    # Preview
    for p in output.get("pages", [])[:1]:
        print("\n" + "=" * 60)
        print(f"PAGE {p['page']} PREVIEW (first 500 chars):")
        print("=" * 60)
        print(p["text"][:500])


if __name__ == "__main__":
    main()
