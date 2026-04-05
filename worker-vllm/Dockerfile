FROM vllm/vllm-openai:v0.19.0-cu130-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_LOGGING_LEVEL=WARNING

WORKDIR /app

# Add RunPod SDK + PDF handling (small delta on top of vLLM image)
RUN pip install --no-cache-dir runpod pymupdf requests httpx

COPY handler.py start.sh ./
RUN chmod +x start.sh

ENTRYPOINT ["/app/start.sh"]
