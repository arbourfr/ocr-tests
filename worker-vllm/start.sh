#!/usr/bin/env bash
# Launch vLLM in background; ALWAYS start RunPod handler so we can surface errors.
set -u  # note: NOT -e — we want to keep going even if vLLM fails

MODEL="${MODEL_PATH:-zai-org/GLM-OCR}"
PORT="${VLLM_PORT:-8000}"
VLLM_LOG="/tmp/vllm.log"

echo "[start] ========================================"
echo "[start] pid=$$  date=$(date -u)"
echo "[start] python: $(which python) $(python --version 2>&1)"
echo "[start] vllm:   $(python -c 'import vllm; print(vllm.__version__)' 2>&1)"
echo "[start] torch:  $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1)"
echo "[start] nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "(no nvidia-smi)"
echo "[start] ========================================"

echo "[start] Launching vLLM for ${MODEL} on :${PORT}  (logs → ${VLLM_LOG})"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --served-model-name glm-ocr > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[start] vLLM pid=${VLLM_PID}"

# Wait up to 8 minutes for vLLM readiness
echo "[start] Waiting for vLLM /health…"
VLLM_READY=0
for i in $(seq 1 160); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "[start] vLLM ready after ${i} checks ($((i*3))s)"
        VLLM_READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[start] !!! vLLM process died (exit code $?) !!!"
        echo "[start] === vLLM log (last 100 lines) ==="
        tail -n 100 "${VLLM_LOG}" 2>&1
        echo "[start] === end vLLM log ==="
        break
    fi
    sleep 3
done

if [ $VLLM_READY -eq 0 ] && kill -0 $VLLM_PID 2>/dev/null; then
    echo "[start] vLLM still not healthy after 8 min"
    tail -n 60 "${VLLM_LOG}" 2>&1
fi

export VLLM_READY
echo "[start] Starting RunPod handler (VLLM_READY=${VLLM_READY})…"
exec python -u handler.py
