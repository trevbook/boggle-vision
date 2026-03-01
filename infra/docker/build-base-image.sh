#!/usr/bin/env bash
# =============================================================================
# Build and push the Boggle Vision Python Lambda base image to ECR.
#
# Usage:
#   ./infra/docker/build-base-image.sh
#   AWS_REGION=us-west-2 ./infra/docker/build-base-image.sh   # override region
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-personal}"
REPOSITORY_NAME="boggle-vision-python-base"
IMAGE_TAG="latest"
PYTHON_VERSION="3.12"

export AWS_PROFILE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONOREPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}"

ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPOSITORY_NAME}"

echo "============================================="
echo "Boggle Vision — Python Lambda Base Image"
echo "============================================="
echo "AWS Profile:   ${AWS_PROFILE}"
echo "AWS Account:   ${ACCOUNT_ID}"
echo "AWS Region:    ${AWS_REGION}"
echo "Repository:    ${REPOSITORY_NAME}"
echo "============================================="

# ── 1. Generate requirements.txt from uv.lock ────────────────────────────────
echo ""
echo "Generating requirements.txt for cv-pipeline..."

cd "${MONOREPO_ROOT}"

uv export \
    --package=cv-pipeline \
    --output-file="${BUILD_DIR}/requirements.txt" \
    --no-emit-workspace \
    --no-dev \
    --quiet

lines=$(wc -l < "${BUILD_DIR}/requirements.txt" | tr -d ' ')
echo "  Generated requirements.txt (${lines} lines)"

# ── 2. Ensure ECR repository exists ──────────────────────────────────────────
echo ""
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories \
    --repository-names "${REPOSITORY_NAME}" \
    --region "${AWS_REGION}" \
    >/dev/null 2>&1 || \
    aws ecr create-repository \
        --repository-name "${REPOSITORY_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE

# ── 3. Authenticate Docker with ECR ──────────────────────────────────────────
echo ""
echo "Authenticating with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ── 4. Stage model weights into build context ────────────────────────────────
echo ""
echo "Staging model weights..."

MODELS_BUILD_DIR="${BUILD_DIR}/models"
mkdir -p "${MODELS_BUILD_DIR}"
cp "${MONOREPO_ROOT}/prototyping/yolov8s-seg.pt" "${MODELS_BUILD_DIR}/"
cp "${MONOREPO_ROOT}/prototyping/legacy/models/boggle_cnn_v2.onnx" "${MODELS_BUILD_DIR}/"
cp "${MONOREPO_ROOT}/prototyping/legacy/models/boggle_cnn_v2.onnx.data" "${MODELS_BUILD_DIR}/"

echo "  Staged 3 model files into ${MODELS_BUILD_DIR}"

# ── 5. Build (arm64 Graviton to match Lambda architecture) ───────────────────
echo ""
echo "Building base image..."
cd "${BUILD_DIR}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

docker build \
    --platform linux/arm64 \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    -t "${REPOSITORY_NAME}:${IMAGE_TAG}" \
    -t "${ECR_URI}:${IMAGE_TAG}" \
    -t "${ECR_URI}:${TIMESTAMP}" \
    .

# ── 6. Push to ECR (latest + timestamped backup) ─────────────────────────────
echo ""
echo "Pushing to ECR..."
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${TIMESTAMP}"

# ── 7. Cleanup ────────────────────────────────────────────────────────────────
echo ""
echo "Cleaning up..."
rm -f "${BUILD_DIR}/requirements.txt"
rm -rf "${BUILD_DIR}/models"

echo ""
echo "============================================="
echo "Done! Base image pushed to:"
echo "  ${ECR_URI}:${IMAGE_TAG}"
echo "  ${ECR_URI}:${TIMESTAMP}"
echo "============================================="
