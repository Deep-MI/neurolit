#!/bin/bash
# Developer script to build a Singularity image from the locally built Docker image.
#
# Requirements:
#   - Docker (with daemon access, i.e. docker group membership or root)
#   - singularityware/docker2singularity (pulled automatically by Docker)
#
# For end users: no Docker or root required. The run_lit_containerized.sh --singularity
# flag pulls directly from Docker Hub using singularity/apptainer.

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build Docker container
docker build "$SCRIPT_DIR/.." -f "$SCRIPT_DIR/Dockerfile" -t deepmi/lit:singularity_preparation

# Convert to Singularity SIF using docker2singularity
# Output file will be named deepmi_lit_singularity_preparation_<timestamp>.simg
docker run --privileged -t --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$SCRIPT_DIR/:/output" \
    singularityware/docker2singularity \
    deepmi/lit:singularity_preparation

# Fix ownership and rename to a stable filename (handle both .simg and .sif output)
shopt -s nullglob
output_files=("$SCRIPT_DIR"/deepmi_lit_singularity_preparation*.simg "$SCRIPT_DIR"/deepmi_lit_singularity_preparation*.sif)
if [[ ${#output_files[@]} -ne 1 ]]; then
    echo "Error: expected exactly one output image, found ${#output_files[@]}"
    exit 1
fi
docker run --rm -v "$SCRIPT_DIR/:/output" --entrypoint bash ubuntu \
    -c "chown $(id -u):$(id -g) /output/$(basename "${output_files[0]}")"
mv "${output_files[0]}" "$SCRIPT_DIR/deepmi_lit_dev.sif"

docker rmi deepmi/lit:singularity_preparation

echo "Singularity image saved to: $SCRIPT_DIR/deepmi_lit_dev.sif"
