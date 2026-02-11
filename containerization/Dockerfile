FROM deepmi/fastsurfer:gpu-v2.2.0

# Install the neurolit package
WORKDIR /inpainting
COPY . /inpainting/
RUN pip3 install --no-cache-dir .

# Download models during build so they are baked into the image.
# We set XDG_DATA_HOME to /usr/local/share to make models accessible to all users.
ENV XDG_DATA_HOME=/usr/local/share
RUN lit-download-models && chmod -R a+rX /usr/local/share/LIT

# Update FastSurfer integration to use the new entrypoint.
# The original FastSurfer segmentation script is moved to run_fastsurfer_segmentation.sh.
RUN mv /fastsurfer/run_fastsurfer.sh /fastsurfer/run_fastsurfer_segmentation.sh && \
    echo '#!/bin/bash' > /fastsurfer/run_fastsurfer.sh && \
    echo 'lit-inpainting "$@"' >> /fastsurfer/run_fastsurfer.sh && \
    chmod +x /fastsurfer/run_fastsurfer.sh

# Entrypoints available: lit-inpainting, lit-download-models, lit-postprocessing
ENTRYPOINT ["lit-inpainting"]
