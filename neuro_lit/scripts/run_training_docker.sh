#!/bin/bash

# Validate input parameters
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_ids> <output_path> [additional_args]"
    exit 1
fi

# Convert output path to absolute path
outpath="$2"
if [[ "$outpath" != /* ]]; then
    outpath="$(pwd)/$outpath"
fi

# Create output directory
mkdir -p "$outpath"

echo "Running training script on GPUs: $1 with output path: $outpath"

docker run --gpus "device=$1" -it --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm \
    -v /groups/ag-reuter/projects/datasets/fs711_hires/:/groups/ag-reuter/projects/datasets/fs711_hires/:ro \
    -v "$outpath:/output_dir" \
    -v "$(pwd):/workspace" \
    deepmi/LIT:dev \
    /bin/bash -c "cd /workspace && python3 training/train_3d_ddpm.py --out_dir /output_dir ${3:-}"
