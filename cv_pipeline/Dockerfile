# syntax=docker/dockerfile:1
# Boggle Vision CV Pipeline Lambda
#
# Extends the pre-built base image containing all system and Python deps.
# Only application code is added here, keeping deploys fast.
#
# To rebuild the base image (after adding/changing Python dependencies):
#   just build-base-image

FROM 918687634324.dkr.ecr.us-east-1.amazonaws.com/boggle-vision-python-base:latest

# ARM64 Lambda workaround: ONNX Runtime's cpuinfo library crashes if
# /sys/devices/system/cpu/{possible,present} are missing (Lambda strips them).
# COPY writes to the image layer (RUN fails because /sys is read-only during
# docker build). "0-1" matches 2048 MB Lambda (~2 vCPUs).
# See: https://github.com/microsoft/onnxruntime/issues/10038
COPY <<EOF /sys/devices/system/cpu/possible
0-1
EOF
COPY <<EOF /sys/devices/system/cpu/present
0-1
EOF

COPY . ${LAMBDA_TASK_ROOT}
