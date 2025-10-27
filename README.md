# Lesion Inpainting Tool (LIT) 🔥

![teaser](https://github.com/ClePol/LIT/blob/dev/docs/overview.png?raw=true)

## Overview
This repository contains sourcecode and documentation related to our publication [**FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation With Tumors,
Cavities and Abnormalities**](https://doi.org/10.1162/imag_a_00446).
This tool can inpaint lesions independent of their shape or appearance for further downstream analysis. The tool can be run standalone and in conjuction with FastSurfer for whole brain segmentation and cortical surface reconstruction. It can also mask tumor regions in the FastSurfer outputs.

## Quickstart

```bash
git clone https://github.com/Deep-MI/LIT.git && cd LIT
./LIT/scripts/run_lit_containerized.sh --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory
# Add --singularity to use singularity instead of docker
```

## How to run LIT

We recommend using containerization in combination with the [LIT/scripts/run_lit_containerized.sh](LIT/scripts/run_lit_containerized.sh) wrapper script.
This will automatically build the docker image from [dockerhub](https://hub.docker.com/r/deepmi/lit) and singularity image and run the LIT and optionally FastSurfer.
We also have a pip release of LIT, currently in a beta version on Test pypi (see below)


### Running LIT (only)

The most straight forward way of doing the inpainting is just providing 
1. The T1w image
2. The lesion mask
3. An output directory
4. (optional) The number times to dilate the lesion mask (default: 0)

```bash
./LIT/scripts/run_lit_containerized.sh --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory --dilate 2
```
The default is to use docker. Add the `--use_singularity` flag to use singularity instead. To use the containerized version of this tool either docker or singularity should be installed. To build the singularity image docker is also required, otherwise please download the prebuild image.


The outputs will be placed in the output directory in the folder inpainting_volumes and contain
- The inpainted T1w image
- The (dilated) mask used for inpainting (in the same space as the input image)
- The inpainted T1w image, where the lesion is cropped out

We recommend performing dilation, since undersegmentation can negatively impact the performance of the inpainting, while oversegmentation should not have significant impact.


If the source image was isotropic, the output images should have the same resolution as the input image and the area outside of the lesion mask should be preversed, except for a robust rescaling of the intensity values.


#### Installation from PyPI

The same interface as above can be accessed from pypi (currently on test-pypi)

```bash
# Install the package
pip install -i https://test.pypi.org/simple/ lesion-inpainting-tool

# Download model checkpoints (recommended - download once, ~500MB)
lit-download-models

# Run LIT
run-lit --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_directory --dilate 2
```

**Note:** If you skip the `lit-download-models` step, models will be automatically downloaded on first use.


### Running LIT in combination with FastSurfer

Currently, LIT is still being integrated into FastSurfer. Until then, you can run LIT first and then run FastSurfer on the inpainted image.
The FastSurfer [repository](https://github.com/deep-mi/FastSurfer) for more information.

If you want to mask the FastSurfer outputs, please use the python scripts [lit/postprocessing/lesion_to_segmentation.py](lit/postprocessing/lesion_to_segmentation.py) and [lit/postprocessing/lesion_to_surfaces.py](lit/postprocessing/lesion_to_surfaces.py) as shown below:



Masking segmentation files:

```bash
# Replace /fastsurfer_output and /inpainting_output with the actual paths
python lit/postprocessing/lesion_to_segmentation.py \
-i "/fastsurfer_output/mri/aparc+aseg.mgz" \
-m "/inpainting_output/inpainting_volumes/inpainting_mask.nii.gz" \
-o "/fastsurfer_output/mri/aparc+aseg+lesion.mgz"
```


Masking surfaces:

```bash
# Replace /fastsurfer_output and /inpainting_output with the actual paths
hemisphere="lh"
python lit/postprocessing/lesion_to_surface.py \
    --inseg "/inpainting_output/inpainting_volumes/inpainting_mask.nii.gz" \
    --insurf "/fastsurfer_output/surf/$hemisphere.white.preaparc" \
    --incort "/fastsurfer_output/label/$hemisphere.cortex.label" \
    --outaparc "/fastsurfer_output/label/$hemisphere.lesion.label" \
    --surflut "lit/postprocessing/DKTatlaslookup_lesion.txt" \  # both lookup files are in the repository
    --seglut "lit/postprocessing/hemi.DKTatlaslookup_lesion.txt" \ 
    --projmm 0 \
    --radius 0 \
    --single_label \
    --to_annot "/fastsurfer_output/labe/$hemisphere.aparc.DKTatlas.annot"
```

Useful FastSurfer flags:
- `--seg_only` skip cortical surface reconstruction (much faster!)
- `--fs_license` has to be set to a valid FreeSurfer license file
- `--threads 2` accelerate cortical surface reconstruction by processing both hemispheres in parallel



## Training

The training script can be found [here](lit/train_ddpm.py). The same docker image can be used for training, but you need to mount the training data directory using the `-v` flag. Note that training data are excpected to be conformed (with the script [conform.py](lit/data/conform.py)).

## Documentation

Comprehensive documentation is available in the `docs/` directory. To build and view:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
cd docs && make html

# View in browser
firefox _build/html/index.html  # Or your preferred browser
```

Documentation includes:
- Installation guides
- Usage tutorials
- API reference
- FastSurfer integration guide
- Training instructions
- Contributing guidelines

## References

If you use LIT for research publications, please cite:

_Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience._
