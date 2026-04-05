"""Run OCR on a PDF using Google Gemini (default: gemini-3-flash-preview).

Sends the PDF inline as a base64-encoded part. Gemini natively parses PDFs
(layout, tables, images, text) and returns markdown.

Usage:
    python scripts/run_gemini_ocr.py                              # samples/acte_10175367.pdf
    python scripts/run_gemini_ocr.py path/to/file.pdf
    python scripts/run_gemini_ocr.py --model gemini-2.5-flash
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

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY not set in .env")

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROMPT = (
    "Transcribe this PDF into clean Markdown. Preserve structure: headings, "
    "paragraphs, tables, lists, footnotes. Do NOT summarize — give the full text "
    "verbatim. For each page, start with a `## Page N` heading."
)


def run_ocr(pdf_path: Path, model: str, prompt: str) -> dict:
    pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    body = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 65536,
        },
    }
    t0 = time.time()
    r = requests.post(url, json=body, timeout=600)
    elapsed = time.time() - t0
    if not r.ok:
        sys.exit(f"Gemini error {r.status_code}: {r.text[:1000]}")
    data = r.json()

    # Extract text
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        sys.exit(f"Unexpected response shape: {e}\n{json.dumps(data, indent=2)[:1000]}")

    usage = data.get("usageMetadata", {})
    return {
        "model": model,
        "text": text,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(ROOT / "samples/acte_10175367.pdf"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Not found: {pdf_path}")

    print(f"[gemini] model={args.model}  pdf={pdf_path.name}  size={pdf_path.stat().st_size} bytes")
    result = run_ocr(pdf_path, args.model, args.prompt)
    print(f"[gemini] done in {result['elapsed_s']}s  tokens: in={result['prompt_tokens']} out={result['output_tokens']}")

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / f"{pdf_path.stem}.gemini.md"
    with md_path.open("w") as f:
        f.write(f"# OCR — {pdf_path.name} (Gemini)\n\n")
        f.write(f"Model: {result['model']}  |  Elapsed: {result['elapsed_s']}s  |  Tokens: in={result['prompt_tokens']} out={result['output_tokens']}\n\n---\n\n")
        f.write(result["text"])
    print(f"[out] {md_path}")

    print("\n" + "=" * 60)
    print(f"PREVIEW (first 500 chars):")
    print("=" * 60)
    print(result["text"][:500])


if __name__ == "__main__":
    main()
