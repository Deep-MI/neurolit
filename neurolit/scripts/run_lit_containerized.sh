#!/bin/bash

set -e

function usage()
{
cat << EOF

Usage: run_lit_containerized.sh --input_image <input_t1w_volume> --mask_image <lesion_mask_volume> --output_directory <output_directory>  [OPTIONS]

run_lit_containerized.sh takes a T1 full head image and creates:
     (i)  an inpainted T1w image using a lesion mask

FLAGS:
  -h, --help
      Print this message and exit
  --version
      Print the version number and exit
  --gpus <gpus>
      GPUs to use. Default: all
  --device <auto|cpu|cuda>
      Inference device inside the container. Default: auto
  --cpu
      Shorthand for --device cpu
  -i, --input_image <input_image>
      Path to the input T1w volume
  -m, --mask_image <mask_image>
      Path to the lesion mask volume (same dimensions as input_image, >0 for lesion, 0 for background)
  -o, --output_directory <output_directory>
      Path to the output directory
  --singularity
      Use singularity instead of docker
  --tag <tag>
      Docker tag to use (e.g., 'latest' or '0.6dev'). If a full image name is 
      provided (containing '/' or ':'), it overrides the default repository.

Examples:
  ./run_lit_containerized.sh -i t1w.nii.gz -m lesion.nii.gz -o ./output 

REFERENCES:

If you use LIT for research publications, please cite:

Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole
  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience.
EOF
}

# Validate required parameters
if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJ_DIR=$(realpath $SCRIPT_DIR/../..)

POSITIONAL_ARGS=()

# Define the Docker Hub repository
DOCKER_REPO="deepmi/lit"

# Initialize variables
USE_SINGULARITY=false
VERSION=""
DEVICE="auto"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --gpus requires a value"
        usage
        exit 1
      fi
      GPUS="$2"
      shift 2
      ;;
    --device)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --device requires a value"
        usage
        exit 1
      fi
      if [[ "$2" != "auto" && "$2" != "cpu" && "$2" != "cuda" ]]; then
        echo "Error: --device must be one of: auto, cpu, cuda"
        usage
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    -i|--input_image)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --input_image requires a value"
        usage
        exit 1
      fi
      INPUT_IMAGE="$2"
      shift 2
      ;;
    -m|--mask_image)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --mask_image requires a value"
        usage
        exit 1
      fi
      MASK_IMAGE="$2"
      shift 2
      ;;
    -o|--output_directory)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --output_directory requires a value"
        usage
        exit 1
      fi
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit
      ;;
    --version)
      # If version is not set yet, we might need a default for this flag
      if [[ -z "$VERSION" ]]; then
        # Quick fetch for version if --version is called early.
        # Guard curl to avoid exiting under 'set -e' if curl is unavailable or the request fails.
        if command -v curl >/dev/null 2>&1; then
          VERSION=$((curl -s "https://hub.docker.com/v2/repositories/${DOCKER_REPO}/tags/?page_size=100" || true) | \
                    grep -oP '"name":\s*"\K[0-9]+\.[0-9]+\.[0-9]+' | \
                    sort -V | tail -n 1)
        fi
        [[ -z "$VERSION" ]] && VERSION="0.5.0"
      fi
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
    --singularity)
      USE_SINGULARITY=true
      shift # past argument
      ;;
    --tag)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --tag requires a value"
        usage
        exit 1
      fi
      VERSION="$2"
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# Get the latest version from Docker Hub if not specified
if [[ -z "$VERSION" ]]; then
  # Fetch tags from Docker Hub API, sort them, and get the latest numeric one
  if command -v curl >/dev/null 2>&1; then
    VERSION=$(curl -s "https://hub.docker.com/v2/repositories/${DOCKER_REPO}/tags/?page_size=100" | \
              grep -oP '"name":\s*"\K[0-9]+\.[0-9]+\.[0-9]+' | \
              sort -V | tail -n 1)
  fi
  if [[ -z "$VERSION" ]]; then
    VERSION="0.5.0"
  fi
fi

# Determine full image name
if [[ "$VERSION" == *"/"* ]] || [[ "$VERSION" == *":"* ]]; then
  IMAGE="$VERSION"
else
  IMAGE="${DOCKER_REPO}:${VERSION}"
fi

set -- "${POSITIONAL_ARGS[@]}"

# Validate required parameters and files
if [ -z "$INPUT_IMAGE" ] || [ -z "$MASK_IMAGE" ] || [ -z "$OUT_DIR" ]; then
  echo "Error: input_image, mask_image, and output_directory are required parameters"
  usage
  exit 1
fi

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Error: Input image not found: $INPUT_IMAGE"
  exit 1
fi

if [ ! -f "$MASK_IMAGE" ]; then
  echo "Error: Mask image not found: $MASK_IMAGE"
  exit 1
fi


mkdir -p "$OUT_DIR"

# Make all inputs absolute paths
INPUT_IMAGE=$(realpath "$INPUT_IMAGE")
MASK_IMAGE=$(realpath "$MASK_IMAGE")
OUT_DIR=$(realpath "$OUT_DIR")

if [ -z "$GPUS" ]; then
  GPUS="all"
fi

if [[ "$DEVICE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    RESOLVED_DEVICE="cuda"
  else
    RESOLVED_DEVICE="cpu"
  fi
else
  RESOLVED_DEVICE="$DEVICE"
fi

# Run command based on the containerization tool
if [ "$USE_SINGULARITY" = true ]; then
  if [ ! -f "$PROJ_DIR/containerization/deepmi_lit.simg" ]; then
    echo "=============== Pulling Singularity image from Docker Hub... ==============="
    singularity pull "$PROJ_DIR/containerization/deepmi_lit.simg" docker://${IMAGE}
  fi

  if [ ! -f "$PROJ_DIR/containerization/deepmi_lit.simg" ]; then
    echo "Error: Singularity image not found: $PROJ_DIR/containerization/deepmi_lit.simg"
    exit 1
  fi

  SINGULARITY_ARGS=()
  if [[ "$RESOLVED_DEVICE" == "cuda" ]]; then
    SINGULARITY_ARGS+=(--nv)
  fi

  singularity exec "${SINGULARITY_ARGS[@]}" \
    -B "${INPUT_IMAGE}":"${INPUT_IMAGE}":ro \
    -B "${MASK_IMAGE}":"${MASK_IMAGE}":ro \
    -B "${OUT_DIR}":"${OUT_DIR}" \
    $PROJ_DIR/containerization/deepmi_lit.simg \
    lit-inpainting -i "${INPUT_IMAGE}" -m "${MASK_IMAGE}" -o "${OUT_DIR}" --device "${RESOLVED_DEVICE}" "${POSITIONAL_ARGS[@]}"
else
  DOCKER_ARGS=(run --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864 --rm
    -v "${INPUT_IMAGE}":"${INPUT_IMAGE}":ro
    -v "${MASK_IMAGE}":"${MASK_IMAGE}":ro
    -v "${OUT_DIR}":"${OUT_DIR}"
    -u "$(id -u):$(id -g)")

  if [[ "$RESOLVED_DEVICE" == "cuda" ]]; then
    DOCKER_ARGS+=(--gpus "device=$GPUS")
  fi

  docker "${DOCKER_ARGS[@]}" \
    ${IMAGE} -i "${INPUT_IMAGE}" -m "${MASK_IMAGE}" -o "${OUT_DIR}" --device "${RESOLVED_DEVICE}" "${POSITIONAL_ARGS[@]}"
fi
