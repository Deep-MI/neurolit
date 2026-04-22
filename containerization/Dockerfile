FROM deepmi/fastsurfer:gpu-v2.2.0
#TODO: update to latest FastSurfer image

ARG NEUROLIT_WHEEL=""

# Install the neurolit package
WORKDIR /inpainting
COPY . /inpainting/
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0+cu128 torchvision==0.25.0+cu128 && \
    pip3 install --no-cache-dir --no-deps monai==1.5.2 lapy==1.5.0 && \
    if [ -n "${NEUROLIT_WHEEL}" ]; then \
        pip3 install --no-cache-dir --no-deps "/inpainting/${NEUROLIT_WHEEL}"; \
    else \
        pip3 install --no-cache-dir --no-deps .; \
    fi

# Download models during build so they are baked into the image.
# We set XDG_DATA_HOME to /usr/local/share to make models accessible to all users.
ENV XDG_DATA_HOME=/usr/local/share \
    XDG_CACHE_HOME=/tmp/xdg-cache
RUN mkdir -p /tmp/xdg-cache/fontconfig && chmod -R 1777 /tmp/xdg-cache && \
    lit-download-models && \
    test -s /usr/local/share/LIT/weights/model_coronal.pt && \
    test -s /usr/local/share/LIT/weights/model_axial.pt && \
    test -s /usr/local/share/LIT/weights/model_sagittal.pt && \
    chmod -R a+rX /usr/local/share/LIT

# Update FastSurfer integration to use the new entrypoint.
# The original FastSurfer segmentation script is moved to run_fastsurfer_segmentation.sh.
RUN mv /fastsurfer/run_fastsurfer.sh /fastsurfer/run_fastsurfer_segmentation.sh && \
    echo '#!/bin/bash' > /fastsurfer/run_fastsurfer.sh && \
    echo 'lit-inpainting "$@"' >> /fastsurfer/run_fastsurfer.sh && \
    chmod +x /fastsurfer/run_fastsurfer.sh

# Entrypoints available: lit-inpainting, lit-download-models, lit-postprocessing
ENTRYPOINT ["lit-inpainting"]
