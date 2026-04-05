#!/usr/bin/env bash
# Launch vLLM server in the background, wait until ready, then start RunPod handler.
set -euo pipefail

MODEL="${MODEL_PATH:-zai-org/GLM-OCR}"
PORT="${VLLM_PORT:-8000}"

echo "[start] Launching vLLM server for ${MODEL} on port ${PORT}…"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --allowed-local-media-path / \
    --served-model-name glm-ocr &

VLLM_PID=$!

# Wait for vLLM readiness
echo "[start] Waiting for vLLM /health…"
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; then
        echo "[start] vLLM ready after ${i} checks"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[start] vLLM process died before becoming ready"
        exit 1
    fi
    sleep 3
done

echo "[start] Starting RunPod handler…"
exec python -u handler.py
