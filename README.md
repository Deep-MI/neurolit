# Neuro Lesion Inpainting Tool (neuroLIT) 🔥

![teaser](https://github.com/Deep-MI/neurolit/blob/dev/doc/overview.png)

## Overview
This repository contains sourcecode and documentation related to our publication [**FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation With Tumors, Cavities and Abnormalities**](https://doi.org/10.1162/imag_a_00446).
This tool can inpaint lesions independent of their shape or appearance for further downstream analysis. This allows subsequent analysis to run as if no lesion were present, enabling for example, FastSurfer's whole brain segmentation in cases with large brain tumors or surgical cavities.

## Quickstart

```bash
git clone https://github.com/Deep-MI/neurolit.git && cd neurolit
./neuroLIT/scripts/run_lit_containerized.sh --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory
# Add --singularity to use singularity instead of docker
```

## How to run neuroLIT

We recommend using containerization in combination with the [neurolit/scripts/run_lit_containerized.sh](neurolit/scripts/run_lit_containerized.sh) wrapper script.
This will automatically build the docker image from [dockerhub](https://hub.docker.com/r/deepmi/neurolit) and singularity image and run the neuroLIT inpainting.
We also have a pip release of neuroLIT.


### Running neuroLIT

The most straight forward way of doing the inpainting is just providing 
1. The T1w image
2. The lesion mask
3. An output directory
4. (optional) The number times to dilate the lesion mask (default: 0)

```bash
./neuroLIT/scripts/run_lit_containerized.sh --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory --dilate 2
```
The default is to use docker. Add the `--singularity` flag to use singularity instead. To use the containerized version of this tool either docker or singularity should be installed. To build the singularity image docker is also required, otherwise please download the prebuild image.


The outputs will be placed in the output directory in the folder inpainting_volumes and contain
- The inpainted T1w image (`inpainting_result.nii.gz`)
- The (dilated) mask used for inpainting in the same space as the input image (`inpainting_mask.nii.gz`)
- The conformed original image (`inpainting_original_image.nii.gz`)

We recommend performing dilation, since undersegmentation can negatively impact the performance of the inpainting, while oversegmentation should not have significant impact.


If the source image was isotropic, the output images should have the same resolution as the input image and the area outside of the lesion mask should be preserved, except for a robust rescaling of the intensity values.


#### Installation from PyPI

The same interface as above can be accessed from pypi:

```bash
# Install the package
pip install neurolit

# Download model checkpoints (recommended - download once, ~700MB)
lit-download-models

# Run neuroLIT
lit-inpainting --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory --dilate 2
```

**Note:** If you skip the `lit-download-models` step, models will be automatically downloaded on first use.


## Integration with FastSurfer

neuroLIT is being integrated into [FastSurfer](https://github.com/deep-mi/FastSurfer) for whole brain segmentation and cortical surface reconstruction of images with lesions. 

For standalone usage with FastSurfer, neuroLIT provides post-processing scripts for integrating lesions into FastSurfer/FreeSurfer outputs. We recommend using the unified `lit-postprocessing` script which handles mapping the lesion mask to multiple segmentation files, running volume statistics (segstats), and performing surface masking.

### Postprocessing

```bash
# Setup paths
export FASTSURFER_HOME=/path/to/FastSurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Run unified postprocessing
lit-postprocessing \
    -sid SUBJECT_ID \
    -sd /path/to/subjects_dir
```

**Key Features:**
- **Validation**: Automatically checks for FastSurfer/FreeSurfer installations.
- **Dynamic Configuration**: Uses `segstats_config.json` for volumetric stats and `surfstats_config.json` for surface stats.
- **Support for All Outputs**: Maps lesions to all relevant `.mgz` files and runs `segstats`.
- **Surface Stats**: Runs `mris_anatomical_stats` calls defined in `surfstats_config.json`.
- **Surface Masking**: Automatically runs surface masking for both hemispheres.
- **Adjacent Label Reports**: Generates adjacency reports for lesion segmentations, allowing user to asses which regions are affected/replaced by the lesion.

## Training

The training script can be found [here](neurolit/train_ddpm.py). The same docker image can be used for training, but you need to mount the training data directory using the `-v` flag. Note that training data are expected to be conformed (with the script [conform.py](neurolit/data/conform.py)).

## Documentation

Comprehensive documentation is available in the `doc/` directory and will be made public at [DeepMI.org/neurolit](https://Deep-MI.org/neurolit). To build and view:

```bash
# Install documentation dependencies
pip install -r doc/requirements.txt

# Build HTML documentation
cd doc && make html

# View in browser
firefox _build/html/index.html  # Or your preferred browser
```

Documentation includes:
- Installation guides
- Usage tutorials
- API reference
- Contributing guidelines

## References

If you use neuroLIT for research publications, please cite:

_Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience._
