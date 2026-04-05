# OCR Tests — GLM-OCR on RunPod Serverless

Minimal setup to benchmark [GLM-OCR](https://github.com/zai-org/GLM-OCR) (`zai-org/GLM-OCR`, 0.9B params) as a RunPod serverless endpoint and run PDFs through it.

## Architecture

```
samples/*.pdf ──▶ scripts/test_endpoint.py ──▶ RunPod API ──▶ worker/handler.py (GPU)
                                                                 │
                                                                 ├─ PyMuPDF: PDF → images
                                                                 ├─ transformers: GLM-OCR
                                                                 └─ returns markdown per page
```

## Prerequisites

- Docker (with `buildx`) — to build and push the worker image
- A Docker Hub account
- RunPod API key (already in `.env`)
- Python 3.10+ locally

## Setup

```bash
pip install -r requirements-local.txt
```

## 1. Build & push the worker image

```bash
DOCKERHUB_USER=yourname ./scripts/build_push.sh
```

This builds the CUDA image (linux/amd64), pre-downloads GLM-OCR weights into the image (~2GB → faster cold starts), and pushes to `yourname/glm-ocr-runpod:latest`.

## 2. Deploy the serverless endpoint

```bash
python scripts/deploy.py --image yourname/glm-ocr-runpod:latest
```

This creates a RunPod template and serverless endpoint, then writes `RUNPOD_ENDPOINT_ID` to `.env`.

Default GPU preference (cheapest first): RTX A4000 → RTX 2000 Ada → RTX A5000 → RTX 3090. GLM-OCR only needs ~4GB VRAM so 16GB cards are overkill but plentiful.

Endpoint config:
- `workersMin=0` (scale to zero — no idle cost)
- `workersMax=1`
- `idleTimeout=5s`
- `flashboot=true`

## 3. Test with a PDF

```bash
# uses samples/acte_10175367.pdf by default
python scripts/test_endpoint.py

# or pass your own
python scripts/test_endpoint.py /path/to/doc.pdf

# OCR only pages 1-2
python scripts/test_endpoint.py --pages 1 2
```

Output lands in `output/<name>.json` (raw) and `output/<name>.md` (readable).

**First call will be slow** (cold start: pull image + start container ≈ 60–120s). Subsequent calls within `idleTimeout` are warm.

## Input schema (direct API use)

```json
{
  "input": {
    "pdf_base64": "...",           // or pdf_url, image_base64, image_url
    "prompt": "Text Recognition:", // default
    "max_new_tokens": 8192,        // default
    "dpi": 200,                    // PDF render DPI
    "pages": [1, 2]                // optional page filter
  }
}
```

## Costs (rough)

- RTX A4000 serverless: ~$0.00016/s active, $0/s idle
- 10-page PDF ≈ 30–90s compute ≈ **$0.01–$0.02 per document**
- Image storage + cold starts: free

## Files

```
worker/
  Dockerfile              CUDA 12.4 + PyTorch 2.4 + GLM-OCR prefetched
  handler.py              RunPod handler (PDF→images→OCR→markdown)
  requirements.txt
scripts/
  build_push.sh           docker buildx → Docker Hub
  deploy.py               Creates template + endpoint via RunPod REST API
  test_endpoint.py        Submits PDF, polls, saves output
samples/
  acte_10175367.pdf       Test doc
output/                   Results land here
```

## Compare with Gemini

Gemini runs locally (no Docker, no GPU — straight API call). Requires `GEMINI_API_KEY` in `.env`.

```bash
# Just Gemini
.venv/bin/python scripts/run_gemini_ocr.py                     # default: gemini-3-flash-preview
.venv/bin/python scripts/run_gemini_ocr.py --model gemini-2.5-flash

# Both models + diff + similarity metrics
.venv/bin/python scripts/compare.py                            # runs BOTH, writes output/*.compare.md
.venv/bin/python scripts/compare.py --skip-glm                 # Gemini only
.venv/bin/python scripts/compare.py --skip-gemini              # GLM only
```

`compare.py` produces:
- `output/<name>.gemini.md` — Gemini raw output
- `output/<name>.glm.md` — GLM-OCR raw output
- `output/<name>.compare.md` — side-by-side with:
  - Timing, token counts, character/word counts per model
  - Character-level similarity (SequenceMatcher, normalized)
  - Word-set Jaccard similarity
  - Words unique to each model (first 40)
  - Unified diff (first 200 lines)
  - Both full outputs appended

## Future: benchmark alternatives

Swap `MODEL_PATH` in `worker/Dockerfile` + handler to compare:
- `stepfun-ai/GOT-OCR-2.0-hf`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `nanonets/Nanonets-OCR-s`

Or create separate endpoints per model and run `test_endpoint.py` against each.
