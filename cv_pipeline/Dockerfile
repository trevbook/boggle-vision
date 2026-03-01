# Boggle Vision CV Pipeline Lambda
#
# Extends the pre-built base image containing all system and Python deps.
# Only application code is added here, keeping deploys fast.
#
# To rebuild the base image (after adding/changing Python dependencies):
#   just build-base-image

FROM 918687634324.dkr.ecr.us-east-1.amazonaws.com/boggle-vision-python-base:latest

COPY . ${LAMBDA_TASK_ROOT}
