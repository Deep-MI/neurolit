# The FastSurfer image already provides torch, neurolit's dependencies and the LIT weights,
# so only neurolit itself is replaced. Other variants: deepmi/fastsurfer:cu126-v2.6.0, cpu-v2.6.0, ...
ARG BASE_IMAGE=deepmi/fastsurfer:cu132-v2.6.0
FROM ${BASE_IMAGE}

ARG NEUROLIT_WHEEL=""

USER root

# Install the neurolit package from a bind mount, so the build context does not become an image layer
RUN --mount=type=bind,source=.,target=/tmp/neurolit,rw \
    if [ -n "${NEUROLIT_WHEEL}" ]; then \
        pip3 install --no-cache-dir --no-deps "/tmp/neurolit/${NEUROLIT_WHEEL}"; \
    else \
        pip3 install --no-cache-dir --no-deps /tmp/neurolit; \
    fi

# The weights are baked into the base image under XDG_DATA_HOME; this only downloads missing files.
RUN lit-download-models && \
    test -s "${XDG_DATA_HOME}/LIT/weights/model_coronal.pt" && \
    test -s "${XDG_DATA_HOME}/LIT/weights/model_axial.pt" && \
    test -s "${XDG_DATA_HOME}/LIT/weights/model_sagittal.pt" && \
    chmod -R a+rX "${XDG_DATA_HOME}/LIT"

# Update FastSurfer integration to use the new entrypoint.
# The original FastSurfer segmentation script is moved to run_fastsurfer_segmentation.sh.
RUN mv /fastsurfer/run_fastsurfer.sh /fastsurfer/run_fastsurfer_segmentation.sh && \
    echo '#!/bin/bash' > /fastsurfer/run_fastsurfer.sh && \
    echo 'lit-inpainting "$@"' >> /fastsurfer/run_fastsurfer.sh && \
    chmod +x /fastsurfer/run_fastsurfer.sh

USER nonroot

# Entrypoints available: lit-inpainting, lit-download-models, lit-postprocessing
ENTRYPOINT ["lit-inpainting"]
