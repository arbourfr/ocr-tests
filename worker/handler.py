"""RunPod serverless handler for GLM-OCR.

Input schema:
    {
      "input": {
        "pdf_base64": "<base64-encoded PDF bytes>",   # OR
        "image_base64": "<base64-encoded image>",     # OR
        "pdf_url": "https://...",                     # OR
        "image_url": "https://...",
        "prompt": "Text Recognition:",                # optional, default shown
        "max_new_tokens": 8192,                       # optional
        "dpi": 200,                                   # optional, PDF render DPI
        "pages": [1, 2]                               # optional, 1-indexed, default all
      }
    }

Output:
    {
      "pages": [
        {"page": 1, "text": "..."},
        ...
      ],
      "num_pages": N,
      "model": "zai-org/GLM-OCR",
      "elapsed_s": 12.3
    }
"""

import base64
import io
import os
import time
from typing import Any

import fitz  # PyMuPDF
import requests
import runpod
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = os.getenv("MODEL_PATH", "zai-org/GLM-OCR")
DEFAULT_PROMPT = "Text Recognition:"

print(f"[boot] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
DEVICE_INFO = {}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    DEVICE_INFO = {"gpu": props.name, "vram_gb": round(props.total_memory / 1e9, 1), "cc": f"{props.major}.{props.minor}"}
    print(f"[boot] GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f}GB  CC: {props.major}.{props.minor}")

print(f"[boot] Loading model {MODEL_PATH}…")
_t0 = time.time()
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
MODEL = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",  # uses FlashAttention-2 internally on Ampere+
)
MODEL.eval()
print(f"[boot] Model loaded in {time.time() - _t0:.1f}s on device={MODEL.device}  dtype={MODEL.dtype}  attn=sdpa")

# torch.compile — mode=default gives ~20-40% speedup on steady-state decoding.
# We compile just the language model forward (vision encoder has variable shapes).
try:
    if hasattr(MODEL, "language_model"):
        MODEL.language_model = torch.compile(MODEL.language_model, mode="default", dynamic=True)
        print("[boot] torch.compile enabled on language_model")
    else:
        MODEL = torch.compile(MODEL, mode="default", dynamic=True)
        print("[boot] torch.compile enabled on model")
except Exception as _e:
    print(f"[boot] torch.compile skipped: {_e}")

# Warm up with a tiny dummy forward pass so the first real request is already hot
try:
    from PIL import Image as _PILImage
    _dummy = _PILImage.new("RGB", (224, 224), (255, 255, 255))
    _msg = [{"role": "user", "content": [{"type": "image", "image": _dummy}, {"type": "text", "text": "Text Recognition:"}]}]
    _inp = PROCESSOR.apply_chat_template(_msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(MODEL.device)
    _inp.pop("token_type_ids", None)
    with torch.inference_mode():
        MODEL.generate(**_inp, max_new_tokens=4, do_sample=False)
    print(f"[boot] warmup complete")
except Exception as _e:
    print(f"[boot] warmup skipped: {_e}")


def _fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
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


def _run_ocr(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = PROCESSOR.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(MODEL.device)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        generated = MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = PROCESSOR.decode(
        generated[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text.strip()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    payload = event.get("input", {}) or {}

    prompt = payload.get("prompt", DEFAULT_PROMPT)
    max_new_tokens = int(payload.get("max_new_tokens", 1024))
    dpi = int(payload.get("dpi", 200))
    page_filter = payload.get("pages")

    # Resolve input source
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

    # Run OCR
    pages_out: list[dict[str, Any]] = []
    if single_image is not None:
        text = _run_ocr(single_image, prompt, max_new_tokens)
        pages_out.append({"page": 1, "text": text})
    else:
        assert pdf_bytes is not None
        imgs = _pdf_to_images(pdf_bytes, dpi=dpi, page_filter=page_filter)
        for page_num, img in imgs:
            text = _run_ocr(img, prompt, max_new_tokens)
            pages_out.append({"page": page_num, "text": text})

    return {
        "pages": pages_out,
        "num_pages": len(pages_out),
        "model": MODEL_PATH,
        "elapsed_s": round(time.time() - t0, 2),
        "device": DEVICE_INFO,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
