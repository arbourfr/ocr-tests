"""Run BOTH Gemini and GLM-OCR on the same PDF, then build a side-by-side comparison.

Writes to output/:
  <name>.gemini.md        Gemini raw output
  <name>.glm.md           GLM-OCR raw output (one section per page)
  <name>.compare.md       Side-by-side: model stats + unified diff + both outputs

Usage:
    python scripts/compare.py                          # samples/acte_10175367.pdf
    python scripts/compare.py path/to/file.pdf
    python scripts/compare.py --gemini-model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
RUNPOD_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_EP = os.getenv("RUNPOD_ENDPOINT_ID")


# ─── Gemini ───────────────────────────────────────────────────────────────────

GEMINI_PROMPT = (
    "Transcribe this PDF into clean Markdown. Preserve structure: headings, "
    "paragraphs, tables, lists, footnotes. Do NOT summarize — give the full text "
    "verbatim. For each page, start with a `## Page N` heading."
)


def run_gemini(pdf_path: Path, model: str) -> dict:
    if not GEMINI_KEY:
        sys.exit("GEMINI_API_KEY not set")
    pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
            {"text": GEMINI_PROMPT},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 65536},
    }
    t0 = time.time()
    r = requests.post(url, json=body, timeout=600)
    elapsed = time.time() - t0
    if not r.ok:
        sys.exit(f"Gemini error {r.status_code}: {r.text[:500]}")
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    return {
        "model": model,
        "text": text,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
    }


# ─── GLM-OCR via RunPod ──────────────────────────────────────────────────────

def run_glm(pdf_path: Path) -> dict:
    if not RUNPOD_KEY or not RUNPOD_EP:
        sys.exit("RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID missing — run scripts/deploy.py first")
    headers = {"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"}
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()

    t0 = time.time()
    r = requests.post(
        f"https://api.runpod.ai/v2/{RUNPOD_EP}/run",
        headers=headers,
        json={"input": {"pdf_base64": pdf_b64, "dpi": 200}},
        timeout=60,
    )
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"[glm] job={job_id}  polling…")

    last_status = None
    while True:
        s = requests.get(f"https://api.runpod.ai/v2/{RUNPOD_EP}/status/{job_id}", headers=headers, timeout=30)
        s.raise_for_status()
        data = s.json()
        status = data.get("status")
        if status != last_status:
            print(f"[glm] status={status}  ({time.time() - t0:.1f}s)")
            last_status = status
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        if time.time() - t0 > 900:
            sys.exit("[glm] timeout after 900s")
        time.sleep(3)

    elapsed = time.time() - t0
    if data.get("status") != "COMPLETED":
        sys.exit(f"[glm] job failed: {json.dumps(data)[:500]}")
    output = data.get("output", {})
    if "error" in output:
        sys.exit(f"[glm] worker error: {output['error']}")

    # Flatten pages into markdown
    parts = []
    for p in output.get("pages", []):
        parts.append(f"## Page {p['page']}\n\n{p['text']}")
    text = "\n\n".join(parts)
    return {
        "model": output.get("model", "zai-org/GLM-OCR"),
        "text": text,
        "elapsed_s": round(elapsed, 2),
        "num_pages": output.get("num_pages"),
        "worker_elapsed_s": output.get("elapsed_s"),
    }


# ─── Comparison ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Aggressive normalization for comparison: lowercase, collapse whitespace, strip punctuation variance."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def char_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def build_comparison(pdf_path: Path, gemini: dict, glm: dict) -> str:
    a, b = gemini["text"], glm["text"]
    ratio = char_ratio(a, b)
    wa, wb = word_set(a), word_set(b)
    jaccard = len(wa & wb) / max(1, len(wa | wb))
    only_gemini = sorted(wa - wb)[:40]
    only_glm = sorted(wb - wa)[:40]

    # Short unified diff (first 150 lines to stay readable)
    diff = list(difflib.unified_diff(
        a.splitlines(), b.splitlines(),
        fromfile="gemini", tofile="glm", lineterm="", n=1,
    ))[:200]

    md = []
    md.append(f"# OCR Comparison — {pdf_path.name}\n")
    md.append("## Summary\n")
    md.append("| Metric | Gemini | GLM-OCR |")
    md.append("|---|---|---|")
    md.append(f"| Model | `{gemini['model']}` | `{glm['model']}` |")
    md.append(f"| Elapsed (s) | {gemini['elapsed_s']} | {glm['elapsed_s']} |")
    md.append(f"| Characters | {len(a):,} | {len(b):,} |")
    md.append(f"| Words | {len(re.findall(r'\\w+', a)):,} | {len(re.findall(r'\\w+', b)):,} |")
    if "prompt_tokens" in gemini:
        md.append(f"| Tokens (in/out) | {gemini['prompt_tokens']}/{gemini['output_tokens']} | — |")
    if "num_pages" in glm:
        md.append(f"| Pages | — | {glm['num_pages']} |")
    md.append("")
    md.append(f"**Character similarity** (SequenceMatcher, normalized): `{ratio:.3f}`  \n")
    md.append(f"**Word-set Jaccard similarity**: `{jaccard:.3f}`  \n")
    md.append("")
    md.append(f"### Words only in Gemini output (first 40)\n\n`{', '.join(only_gemini) or '—'}`\n")
    md.append(f"### Words only in GLM output (first 40)\n\n`{', '.join(only_glm) or '—'}`\n")
    md.append("")
    md.append("## Unified diff (first 200 lines)\n")
    md.append("```diff")
    md.extend(diff)
    md.append("```")
    md.append("")
    md.append("---\n")
    md.append("## Gemini output\n")
    md.append(a)
    md.append("\n---\n")
    md.append("## GLM-OCR output\n")
    md.append(b)
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(ROOT / "samples/acte_10175367.pdf"))
    ap.add_argument("--gemini-model", default="gemini-3-flash-preview")
    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--skip-glm", action="store_true")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Not found: {pdf_path}")

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    stem = pdf_path.stem

    print(f"[compare] pdf={pdf_path.name}  size={pdf_path.stat().st_size:,} bytes\n")

    gemini = glm = None

    if not args.skip_gemini:
        print(f"→ Running Gemini ({args.gemini_model})…")
        gemini = run_gemini(pdf_path, args.gemini_model)
        (out_dir / f"{stem}.gemini.md").write_text(
            f"# OCR — {pdf_path.name} (Gemini {gemini['model']})\n\n"
            f"Elapsed: {gemini['elapsed_s']}s  |  Tokens: in={gemini['prompt_tokens']} out={gemini['output_tokens']}\n\n---\n\n"
            + gemini["text"]
        )
        print(f"  done in {gemini['elapsed_s']}s, {len(gemini['text']):,} chars\n")

    if not args.skip_glm:
        print(f"→ Running GLM-OCR via RunPod…")
        glm = run_glm(pdf_path)
        (out_dir / f"{stem}.glm.md").write_text(
            f"# OCR — {pdf_path.name} (GLM-OCR)\n\n"
            f"Elapsed: {glm['elapsed_s']}s (worker: {glm.get('worker_elapsed_s')}s)  |  Pages: {glm.get('num_pages')}\n\n---\n\n"
            + glm["text"]
        )
        print(f"  done in {glm['elapsed_s']}s, {len(glm['text']):,} chars\n")

    if gemini and glm:
        comparison = build_comparison(pdf_path, gemini, glm)
        cmp_path = out_dir / f"{stem}.compare.md"
        cmp_path.write_text(comparison)
        print(f"[out] {cmp_path}")
        # Also print the summary block to stdout
        print("\n" + "\n".join(comparison.splitlines()[:25]))


if __name__ == "__main__":
    main()
