"""Run GLM-OCR locally on this Mac (MPS or CPU) — no cloud needed.

Uses transformers + PyMuPDF. First run downloads ~2GB weights from HuggingFace
(cached in ~/.cache/huggingface, reused on subsequent runs).

GLM-OCR is 0.9B params → fits comfortably on M-series Macs via MPS.
On 16GB M1/M2, expect ~5-15s per page. On CPU, ~30-60s per page.

Usage:
    # default: runs first 3 pages of samples/acte_10175367.pdf (quick test)
    python scripts/run_glm_local.py

    # all pages
    python scripts/run_glm_local.py --all

    # specific pages
    python scripts/run_glm_local.py --pages 1 5 10

    # different PDF
    python scripts/run_glm_local.py path/to/file.pdf --all

    # force CPU (if MPS breaks)
    python scripts/run_glm_local.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Let unsupported MPS ops transparently fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "zai-org/GLM-OCR"
DEFAULT_PROMPT = "Text Recognition:"


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pdf_to_images(pdf_path: Path, dpi: int, pages: list[int] | None) -> list[tuple[int, Image.Image]]:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    out: list[tuple[int, Image.Image]] = []
    for i, page in enumerate(doc, start=1):
        if pages and i not in pages:
            continue
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        out.append((i, img))
    doc.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(ROOT / "samples/acte_10175367.pdf"))
    ap.add_argument("--pages", nargs="+", type=int, help="1-indexed page list")
    ap.add_argument("--all", action="store_true", help="Process every page (overrides --pages)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-side", type=int, default=1600, help="Cap longest image side (px) to control MPS memory")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Not found: {pdf_path}")

    # Page selection: default to first 3 pages for a quick test
    page_filter: list[int] | None
    if args.all:
        page_filter = None
    elif args.pages:
        page_filter = args.pages
    else:
        page_filter = [1, 2, 3]
        print("[info] defaulting to pages 1-3 (use --all for every page, or --pages N M ...)")

    device = pick_device(args.device)
    print(f"[device] {device}")

    # Load model
    t0 = time.time()
    print(f"[load] {MODEL_PATH} (first run downloads ~2GB)…")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # fp16 on MPS: halves activation memory, needed to stay under the MPS
    # 2**32-byte single-NDArray ceiling. fp32 on MPS blows past it on vision models.
    if args.dtype == "auto":
        torch_dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    print(f"[load] model ready in {time.time() - t0:.1f}s  (dtype={torch_dtype})")

    # Render pages
    print(f"[pdf] rendering at {args.dpi} DPI…")
    images = pdf_to_images(pdf_path, dpi=args.dpi, pages=page_filter)
    # Cap image size — large inputs blow the MPS 2**32 NDArray limit
    resized: list[tuple[int, Image.Image]] = []
    for page_num, img in images:
        longest = max(img.size)
        if longest > args.max_side:
            scale = args.max_side / longest
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        resized.append((page_num, img))
    images = resized
    w, h = images[0][1].size if images else (0, 0)
    print(f"[pdf] {len(images)} page(s) to OCR (first page: {w}x{h})")

    # Run OCR
    results: list[tuple[int, str, float]] = []
    for page_num, img in images:
        t_page = time.time()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        dt = time.time() - t_page
        results.append((page_num, text, dt))
        print(f"[ocr] page {page_num}: {len(text):,} chars in {dt:.1f}s")

    # Write output
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / f"{pdf_path.stem}.glm.md"
    total_time = sum(dt for _, _, dt in results)
    with md_path.open("w") as f:
        f.write(f"# OCR — {pdf_path.name} (GLM-OCR local, {device})\n\n")
        f.write(f"Model: {MODEL_PATH}  |  Pages: {len(results)}  |  Total OCR time: {total_time:.1f}s  |  Avg: {total_time/max(1,len(results)):.1f}s/page\n\n---\n")
        for page_num, text, dt in results:
            f.write(f"\n## Page {page_num}\n\n*({dt:.1f}s)*\n\n{text}\n")
    print(f"\n[out] {md_path}")
    print(f"[done] {len(results)} pages in {total_time:.1f}s total ({total_time/max(1,len(results)):.1f}s/page)")


if __name__ == "__main__":
    main()
