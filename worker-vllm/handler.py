"""RunPod serverless handler that proxies to a local vLLM server running GLM-OCR.

vLLM handles continuous batching internally: we submit all pages concurrently
and vLLM schedules them for maximum throughput.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from typing import Any

import fitz  # PyMuPDF
import httpx
import runpod
from PIL import Image

VLLM_URL = f"http://127.0.0.1:{os.getenv('VLLM_PORT', '8000')}/v1/chat/completions"
VLLM_READY = os.getenv("VLLM_READY", "0") == "1"
MODEL_NAME = "glm-ocr"  # matches --served-model-name in start.sh
DEFAULT_PROMPT = "Text Recognition:"
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "32"))


def _fetch_bytes(url: str) -> bytes:
    with httpx.Client(timeout=60) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def _pdf_to_images(pdf_bytes: bytes, dpi: int, page_filter: list[int] | None) -> list[tuple[int, Image.Image]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    out: list[tuple[int, Image.Image]] = []
    for i, page in enumerate(doc, start=1):
        if page_filter and i not in page_filter:
            continue
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        out.append((i, img))
    doc.close()
    return out


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


async def _ocr_one(client: httpx.AsyncClient, page_num: int, img: Image.Image, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_url(img)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.time()
    r = await client.post(VLLM_URL, json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"] or ""
    return {"page": page_num, "text": text.strip(), "elapsed_s": round(time.time() - t0, 2)}


async def _ocr_all(pages: list[tuple[int, Image.Image]], prompt: str, max_tokens: int) -> list[dict]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _guarded(client: httpx.AsyncClient, pn: int, im: Image.Image) -> dict:
        async with sem:
            return await _ocr_one(client, pn, im, prompt, max_tokens)

    limits = httpx.Limits(max_connections=MAX_CONCURRENCY * 2, max_keepalive_connections=MAX_CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [_guarded(client, pn, im) for pn, im in pages]
        return await asyncio.gather(*tasks)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    payload = event.get("input", {}) or {}

    # Diagnostic short-circuit: if caller sends {"diagnostic": true}, just report status
    if payload.get("diagnostic"):
        try:
            r = httpx.get("http://127.0.0.1:8000/v1/models", timeout=5)
            vllm_status = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            vllm_status = f"unreachable: {e!r}"
        import subprocess
        smi = subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total","--format=csv,noheader"], capture_output=True, text=True)
        return {
            "vllm_ready_env": VLLM_READY,
            "vllm_http_status": vllm_status,
            "gpu": smi.stdout.strip(),
            "env_model_path": os.getenv("MODEL_PATH"),
        }

    if not VLLM_READY:
        # Try once more in case it came up after handler boot
        try:
            r = httpx.get("http://127.0.0.1:8000/v1/models", timeout=5)
            if r.status_code != 200:
                return {"error": f"vLLM not ready: HTTP {r.status_code}: {r.text[:500]}"}
        except Exception as e:
            return {"error": f"vLLM unreachable: {e!r}"}

    prompt = payload.get("prompt", DEFAULT_PROMPT)
    max_new_tokens = int(payload.get("max_new_tokens", 1024))
    dpi = int(payload.get("dpi", 200))
    page_filter = payload.get("pages")

    pdf_bytes: bytes | None = None
    single_image: Image.Image | None = None

    if payload.get("pdf_base64"):
        pdf_bytes = base64.b64decode(payload["pdf_base64"])
    elif payload.get("pdf_url"):
        pdf_bytes = _fetch_bytes(payload["pdf_url"])
    elif payload.get("image_base64"):
        single_image = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"]))).convert("RGB")
    elif payload.get("image_url"):
        single_image = Image.open(io.BytesIO(_fetch_bytes(payload["image_url"]))).convert("RGB")
    else:
        return {"error": "Provide one of: pdf_base64, pdf_url, image_base64, image_url"}

    if single_image is not None:
        pages = [(1, single_image)]
    else:
        assert pdf_bytes is not None
        pages = _pdf_to_images(pdf_bytes, dpi=dpi, page_filter=page_filter)

    results = asyncio.run(_ocr_all(pages, prompt, max_new_tokens))
    # Sort by page
    results.sort(key=lambda r: r["page"])

    return {
        "pages": [{"page": r["page"], "text": r["text"]} for r in results],
        "num_pages": len(results),
        "model": "zai-org/GLM-OCR (vLLM)",
        "elapsed_s": round(time.time() - t0, 2),
        "per_page_s": [r["elapsed_s"] for r in results],
        "concurrency": MAX_CONCURRENCY,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
