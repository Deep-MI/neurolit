#!/bin/bash

set -e

# Initialize default values
DILATE=0
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Get project directory for git hash (if available)
# This works for git clones; for pip installs it points to site-packages
PROJ_DIR=$(realpath $SCRIPT_DIR/../..)

VERSION="$(python3 -c 'import neurolit; print(neurolit.__version__)' 2>/dev/null)"
VERSION="${VERSION/version = /}"
VERSION="${VERSION//\"/}"

function usage() {
    echo "Usage: $0 -i <input_t1w> -m <lesion_mask> -o <output_dir>"
    echo "Required arguments:"
    echo "  -i, --input_image     : Input T1w image"
    echo "  -m, --lesion_mask     : Lesion mask"
    echo "  -o, --sd              : Output directory"
    echo "Optional arguments:"
    echo "  --dilate              : Number of times to dilate the lesion mask (default: 0)"
    echo "Other arguments:"
    echo "  --version             : Print version number and exit"
    echo ""
    echo "If you use LIT for research publications, please cite:"
    echo ""
    echo "Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole"
    echo "  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience."
    exit
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    usage
fi

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input_image|--t1)
      INPUT_IMAGE="$(realpath "$2")"
      shift 2
      ;;
    -m|--lesion_mask)
      MASK_IMAGE="$(realpath "$2")"
      shift 2
      ;;
    -o|--sd)
      OUT_DIR="$(realpath "$2")"
      shift 2
      ;;
    --dilate)
      DILATE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --version)
      hash_file="$PROJ_DIR/git.hash"
      if [[ -n "$(which git)" ]] && (git -C "$PROJ_DIR" rev-parse 2>/dev/null ) ; then
        HASH="+$(git -C "$PROJ_DIR" rev-parse --short HEAD)"
      elif [[ -e "$hash_file" ]] ; then
        HASH="+$(cat "$hash_file")"
      else
        HASH=""
      fi
      echo "$VERSION$HASH"
      exit
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate required parameters
if [ -z "$INPUT_IMAGE" ] || [ -z "$OUT_DIR" ]; then
    echo "Error: Input image and output directory are required"
    usage
fi

# Validate input files exist
if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Error: Input file $INPUT_IMAGE does not exist"
  exit 1
fi

if [ ! -z "$MASK_IMAGE" ] && [ ! -f "$MASK_IMAGE" ]; then
  echo "Error: Mask file $MASK_IMAGE does not exist"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Set up paths - use platformdirs for consistent location across all installations
WEIGHTS_DIR=$(python3 -c "from platformdirs import user_data_dir; from pathlib import Path; print(Path(user_data_dir('LIT', 'Deep-MI')) / 'weights')")

CKPT_CORONAL="$WEIGHTS_DIR/model_coronal.pt"
CKPT_AXIAL="$WEIGHTS_DIR/model_axial.pt"
CKPT_SAGITTAL="$WEIGHTS_DIR/model_sagittal.pt"
INPAINTED_IMG="$OUT_DIR/inpainting_volumes/inpainting_result.nii.gz"

# Run inpainting if mask is provided
if [ ! -z "$MASK_IMAGE" ]; then
  if [ ! -e "$INPAINTED_IMG" ]; then
    echo "Running inpainting..."
    mkdir -p "$OUT_DIR/inpainting_volumes"

    # Download checkpoints (uses platformdirs for consistent location)
    python3 -m neurolit.utils.download_checkpoints

    # Check for required model files
    for model in "$CKPT_CORONAL" "$CKPT_AXIAL" "$CKPT_SAGITTAL"; do
        if [ ! -f "$model" ]; then
            echo "Error: Required model file not found: $model"
            exit 1
        fi
    done

    
    # Assemble inpainting command
    inpainting_command="python3 -m neurolit.inpaint_image \
--input_image \"$INPUT_IMAGE\" \
--mask_image \"$MASK_IMAGE\" \
--out_dir \"$OUT_DIR\" \
--checkpoint_axial \"$CKPT_AXIAL\" \
--checkpoint_sagittal \"$CKPT_SAGITTAL\" \
--checkpoint_coronal \"$CKPT_CORONAL\" \
--dilate \"$DILATE\""

    # Print command
    echo "Running command:"
    echo "$inpainting_command"

    # Execute command
    eval "$inpainting_command"
  else
    echo "Inpainted image already exists: $INPAINTED_IMG"
  fi
fi

echo "Finished inpainting"
exit 0
