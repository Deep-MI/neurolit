#!/bin/bash

# Script to process volumetric statistics with input validation
# Usage: ./test_volumestats.sh -sid SUBJECT_ID -sd SUBJECTS_DIR [--local]

set -e
set -x

# Default values
SUBJECTS_DIR=""
SID=""
DOCKER_VERSION="dev"
LOCAL_MODE=false
LIT_PATH=""
FASTSURFER_PATH=""
LIT_PATH_PROVIDED=false
FASTSURFER_PATH_PROVIDED=false

# Function to display usage information
usage() {
    echo "Usage: $0 -sid SUBJECT_ID -sd SUBJECTS_DIR [OPTIONS]"
    echo "  -sid SUBJECT_ID       : Subject ID"
    echo "  -sd SUBJECTS_DIR      : Subjects directory"
    echo "  --local               : Run commands locally instead of using Docker"
    echo "  --lit-path PATH       : Path to LIT installation (auto-detected if not provided)"
    echo "  --fastsurfer-path PATH: Path to FastSurfer installation (uses \$FASTSURFER_HOME if set)"
    echo "  --docker-version VER  : Docker image version (default: dev)"
    echo ""
    echo "Environment variables:"
    echo "  FASTSURFER_HOME       : Path to FastSurfer installation (used if --fastsurfer-path not provided)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -sid)
            SID="$2"
            shift 2
            ;;
        -sd)
            SUBJECTS_DIR="$2"
            shift 2
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --lit-path)
            LIT_PATH="$2"
            LIT_PATH_PROVIDED=true
            shift 2
            ;;
        --fastsurfer-path)
            FASTSURFER_PATH="$2"
            FASTSURFER_PATH_PROVIDED=true
            shift 2
            ;;
        --docker-version)
            DOCKER_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$SID" || -z "$SUBJECTS_DIR" ]]; then
    echo "Error: Subject ID and Subjects directory are required."
    usage
fi

# Auto-detect and validate paths for local mode
if [[ "$LOCAL_MODE" == true ]]; then
    echo "=== Local Mode Setup ==="
    
    # Auto-detect LIT path if not provided
    if [[ "$LIT_PATH_PROVIDED" == false ]]; then
        # Get the directory where this script is located
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        
        # Try to find LIT installation
        if [[ -f "$SCRIPT_DIR/LIT/postprocessing/lesion_to_segmentation.py" ]]; then
            LIT_PATH="$SCRIPT_DIR"
            echo "Auto-detected LIT installation: $LIT_PATH"
        elif [[ -f "$SCRIPT_DIR/postprocessing/lesion_to_segmentation.py" ]]; then
            LIT_PATH="$(dirname "$SCRIPT_DIR")"
            echo "Auto-detected LIT installation: $LIT_PATH"
        else
            echo "Could not auto-detect LIT installation"
            echo "Please provide --lit-path explicitly"
            exit 1
        fi
    else
        echo "Using provided LIT path: $LIT_PATH"
    fi
    
    # Auto-detect FastSurfer path if not provided
    if [[ "$FASTSURFER_PATH_PROVIDED" == false ]]; then
        if [[ -n "$FASTSURFER_HOME" ]]; then
            FASTSURFER_PATH="$FASTSURFER_HOME"
            echo "Using FASTSURFER_HOME: $FASTSURFER_PATH"
        else
            echo "FastSurfer path not found"
            echo "  Set FASTSURFER_HOME environment variable or use --fastsurfer-path"
            exit 1
        fi
    else
        echo "Using provided FastSurfer path: $FASTSURFER_PATH"
    fi
    
    # Verify paths exist
    if [[ ! -d "$LIT_PATH" ]]; then
        echo " Error: LIT path not found: $LIT_PATH"
        exit 1
    fi
    
    if [[ ! -d "$FASTSURFER_PATH" ]]; then
        echo "Error: FastSurfer path not found: $FASTSURFER_PATH"
        exit 1
    fi
    
    if [[ ! -f "$FASTSURFER_PATH/FastSurferCNN/segstats.py" ]]; then
        echo "Error: segstats.py not found in FastSurfer path: $FASTSURFER_PATH"
        echo "Checked: $FASTSURFER_PATH/FastSurferCNN/segstats.py"
        exit 1
    fi
    
    echo "All required scripts found"
    echo "=== Running in LOCAL mode ==="
else
    echo "=== Running in DOCKER mode ==="
    echo "  Docker version: $DOCKER_VERSION"
fi

# Check if required input files exist
echo "Checking for required input files..."
required_files=(
    "$SUBJECTS_DIR/$SID/mri/aparc.DKTatlas+aseg.deep.mgz"
    "$SUBJECTS_DIR/$SID/inpainting_volumes/inpainting_mask.nii.gz"
    "$SUBJECTS_DIR/$SID/mri/orig_nu.mgz"
    "$SUBJECTS_DIR/$SID/mri/mask.mgz"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

echo "All required files found. Proceeding with processing..."

# Create output directories if they don't exist
mkdir -p "$SUBJECTS_DIR/$SID/stats"

# Run lesion to segmentation
echo "Running lesion to segmentation..."
if [[ "$LOCAL_MODE" == true ]]; then
    python3 "$LIT_PATH/LIT/postprocessing/lesion_to_segmentation.py" \
        -i "$SUBJECTS_DIR/$SID/mri/aparc.DKTatlas+aseg.deep.mgz" \
        -m "$SUBJECTS_DIR/$SID/inpainting_volumes/inpainting_mask.nii.gz" \
        -o "$SUBJECTS_DIR/$SID/mri/aparc.DKTatlas+aseg+lesion.deep.mgz"
else
    docker run -u $(id -u):$(id -g) --rm -v "$SUBJECTS_DIR/$SID/:/fastsurfer_output/" --entrypoint "/bin/bash" deepmi/lit:$DOCKER_VERSION -c \
        "python3 /inpainting/postprocessing/lesion_to_segmentation.py \
            -i '/fastsurfer_output/mri/aparc.DKTatlas+aseg.deep.mgz' \
            -m '/fastsurfer_output/inpainting_volumes/inpainting_mask.nii.gz' \
            -o '/fastsurfer_output/mri/aparc.DKTatlas+aseg+lesion.deep.mgz'"
fi

# Check if the previous command was successful
if [[ $? -ne 0 ]]; then
    echo "Error: Lesion to segmentation failed."
    exit 1
fi

# Run volumetric statistics
echo "Running volumetric statistics..."
if [[ "$LOCAL_MODE" == true ]]; then
    # Set PYTHONPATH for FastSurfer imports
    export PYTHONPATH="$FASTSURFER_PATH:$PYTHONPATH"
    
    python3 "$FASTSURFER_PATH/FastSurferCNN/segstats.py" \
        --segfile "$SUBJECTS_DIR/$SID/mri/aparc.DKTatlas+aseg+lesion.deep.mgz" \
        --segstatsfile "$SUBJECTS_DIR/$SID/stats/aseg+DKT+lesion.stats" \
        --normfile "$SUBJECTS_DIR/$SID/mri/orig_nu.mgz" \
        --threads 1 --empty --excludeid 0 --sd "$SUBJECTS_DIR" --sid "$SID" \
        --ids 2 4 5 7 8 10 11 12 13 14 15 16 17 18 24 26 28 31 41 43 44 46 47 49 50 51 52 53 54 58 60 63 77 99 251 252 253 254 255 1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1034 1035 2002 2003 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2034 2035 \
        --lut "$FASTSURFER_PATH/FastSurferCNN/config/FreeSurferColorLUT.txt" measures \
        --compute "Mask($SUBJECTS_DIR/$SID/mri/mask.mgz)" BrainSeg BrainSegNotVent SupraTentorial SupraTentorialNotVent SubCortGray rhCerebralWhiteMatter lhCerebralWhiteMatter CerebralWhiteMatter
else
    docker run -u $(id -u):$(id -g) --rm -v "$SUBJECTS_DIR/$SID/:/fastsurfer_output/" --entrypoint "/bin/bash" deepmi/fastsurfer:cpu-v2.4.2 -c \
        "python3.10 -s /fastsurfer/FastSurferCNN/segstats.py \
            --segfile /fastsurfer_output/mri/aparc.DKTatlas+aseg+lesion.deep.mgz \
            --segstatsfile /fastsurfer_output/stats/aseg+DKT+lesion.stats \
            --normfile /fastsurfer_output/mri/orig_nu.mgz \
            --threads 1 --empty --excludeid 0 --sd /fastsurfer_output --sid $SID \
            --ids 2 4 5 7 8 10 11 12 13 14 15 16 17 18 24 26 28 31 41 43 44 46 47 49 50 51 52 53 54 58 60 63 77 99 251 252 253 254 255 1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1034 1035 2002 2003 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2034 2035 \
            --lut /fastsurfer/FastSurferCNN/config/FreeSurferColorLUT.txt measures \
            --compute Mask\(/fastsurfer_output/mri/mask.mgz\) BrainSeg BrainSegNotVent SupraTentorial SupraTentorialNotVent SubCortGray rhCerebralWhiteMatter lhCerebralWhiteMatter CerebralWhiteMatter"
fi

# Check if the previous command was successful
if [[ $? -ne 0 ]]; then
    echo "Error: Volumetric statistics calculation failed."
    exit 1
fi

echo "Running surface masking..."
for hemisphere in lh rh; do
    echo "Processing hemisphere: $hemisphere"
    
    if [[ "$LOCAL_MODE" == true ]]; then
        python3 "$LIT_PATH/LIT/postprocessing/lesion_to_surface.py" \
            --inseg "$SUBJECTS_DIR/$SID/inpainting_volumes/inpainting_mask.nii.gz" \
            --insurf "$SUBJECTS_DIR/$SID/surf/$hemisphere.white.preaparc" \
            --incort "$SUBJECTS_DIR/$SID/label/$hemisphere.cortex.label" \
            --outaparc "$SUBJECTS_DIR/$SID/label/$hemisphere.lesion.annot" \
            --surflut "$LIT_PATH/LIT/postprocessing/DKTatlaslookup_lesion.txt" \
            --seglut "$LIT_PATH/LIT/postprocessing/hemi.DKTatlaslookup_lesion.txt" \
            --projmm 0 \
            --radius 0 \
            --single_label \
            --to_annot "$SUBJECTS_DIR/$SID/label/$hemisphere.aparc.DKTatlas.annot"
    else
        docker run -u $(id -u):$(id -g) --rm -v "$SUBJECTS_DIR/$SID/:/fastsurfer_output/" --entrypoint "/bin/bash" deepmi/lit:$DOCKER_VERSION -c \
        "python3 /inpainting/postprocessing/lesion_to_surface.py \
            --inseg '/fastsurfer_output/inpainting_volumes/inpainting_mask.nii.gz' \
            --insurf '/fastsurfer_output/surf/$hemisphere.white.preaparc' \
            --incort '/fastsurfer_output/label/$hemisphere.cortex.label' \
            --outaparc '/fastsurfer_output/label/$hemisphere.lesion.label' \
            --surflut 'postprocessing/DKTatlaslookup_lesion.txt' \
            --seglut 'postprocessing/hemi.DKTatlaslookup_lesion.txt' \
            --projmm 0 \
            --radius 0 \
            --single_label \
            --to_annot '/fastsurfer_output/label/$hemisphere.aparc.DKTatlas.annot'"
    fi
    
    # Check if the previous command was successful
    if [[ $? -ne 0 ]]; then
        echo "Warning: Surface masking failed for hemisphere $hemisphere."
        # Continue with other hemisphere instead of exiting
    fi
done

echo "Processing completed successfully."
echo "Output statistics file: $SUBJECTS_DIR/$SID/stats/aseg+DKT+lesion.stats"
