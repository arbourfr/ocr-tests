#!/usr/bin/env bash
# Build the worker image for linux/amd64 (RunPod runs x86_64) and push to Docker Hub.
#
# Usage:
#   DOCKERHUB_USER=myuser ./scripts/build_push.sh
#   DOCKERHUB_USER=myuser TAG=v2 ./scripts/build_push.sh
set -euo pipefail

: "${DOCKERHUB_USER:?set DOCKERHUB_USER (your Docker Hub username)}"
TAG="${TAG:-latest}"
IMAGE="${DOCKERHUB_USER}/glm-ocr-runpod:${TAG}"

cd "$(dirname "$0")/../worker"

echo "[build] $IMAGE (linux/amd64)"
docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

echo "[done] pushed $IMAGE"
echo
echo "Now run:"
echo "  python scripts/deploy.py --image $IMAGE"
