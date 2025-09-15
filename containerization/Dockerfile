FROM deepmi/fastsurfer:gpu-v2.2.0

#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN mkdir -p /inpainting/weights && pip3 install monai==1.2.0

RUN apt update && apt install -y curl

ENV Generative_Models_SHA=7428fce193771e9564f29b91d29e523dd1b6b4cd
RUN cd /inpainting && ( curl -L https://github.com/Project-MONAI/GenerativeModels/archive/${Generative_Models_SHA}.tar.gz | tar xz ) && cd GenerativeModels-${Generative_Models_SHA} && python setup.py install && cd .. && rm -rf GenerativeModels-${Generative_Models_SHA}

RUN curl https://zenodo.org/records/14510136/files/model_axial.pt -o /inpainting/weights/model_axial.pt && \
    curl https://zenodo.org/records/14510136/files/model_coronal.pt -o /inpainting/weights/model_coronal.pt && \
    curl https://zenodo.org/records/14510136/files/model_sagittal.pt -o /inpainting/weights/model_sagittal.pt

COPY run_lit.sh pyproject.toml git.hash /inpainting/
COPY lit /inpainting/lit

ENV INPAINTING_DIR=/inpainting
RUN mv /fastsurfer/run_fastsurfer.sh /fastsurfer/run_fastsurfer_segmentation.sh && \
    echo "cd /inpainting/ && ./run_lit.sh \"\$@"\" >> /fastsurfer/run_fastsurfer.sh && \
    chmod +x /fastsurfer/run_fastsurfer.sh

ENTRYPOINT ["/bin/bash", "/inpainting/run_lit.sh"]

WORKDIR /inpainting
