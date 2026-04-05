FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache \
    HF_HUB_DISABLE_TELEMETRY=1

# No apt-get needed — PyMuPDF and Pillow wheels are self-contained.
# (Skipped to avoid GPG signature errors under qemu linux/amd64 emulation on arm64.)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Model is lazy-loaded on first worker boot (adds ~30-60s to first cold start).
# We cannot prefetch at build time because PyTorch under qemu linux/amd64
# emulation on arm64 Macs segfaults. On RunPod's native x86_64 host it's fine.

CMD ["python", "-u", "handler.py"]
