set dotenv-load := true
set export := true

# Default: show available commands
default:
    @just --list

# Install dependencies
install:
    bun install

# Run tests
test:
    bun test

# Run tests in watch mode
test-watch:
    bun test --watch

# Lint project files
lint:
    bunx biome check .

# Lint and auto-fix
lint-fix:
    bunx biome check --write .

# Format all files
format:
    bunx biome format --write .

# Run all checks (CI)
ci: lint test

# Start the app dev server
dev:
    bun run --filter www dev

# Start SST dev mode
sst-dev:
    bunx sst dev

# Set an SST secret
sst-set-secret SECRET VALUE:
    bunx sst secret set {{SECRET}} {{VALUE}}

# Copy models into the local workspace (for sst dev / direct uv run)
copy-models:
    mkdir -p cv_pipeline/models
    cp prototyping/yolov8s-seg.pt cv_pipeline/models/
    cp prototyping/legacy/models/boggle_cnn_v2.onnx cv_pipeline/models/
    cp prototyping/legacy/models/boggle_cnn_v2.onnx.data cv_pipeline/models/

# Build and push the Python Lambda base image to ECR
build-base-image:
    ./infra/docker/build-base-image.sh

# Deploy with SST (models are uploaded to S3 as part of the deploy)
sst-deploy:
    bunx sst deploy

