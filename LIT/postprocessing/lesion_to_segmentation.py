#!/usr/bin/env python3


# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import nibabel as nib
import nibabel.processing
import numpy as np

from LIT.utils.logging import get_logger

logger = get_logger(__name__)


def mask_lesion(to_mask_path, mask_path):
    """Mask the provided image volume with the tumor segmentation.

    Parameters
    ----------
    to_mask_path : str
        Path to the volume that should have the lesion masked out.
    mask_path : str
        Path to the tumor mask volume.

    Returns
    -------
    nibabel.spatialimages.SpatialImage
        Masked image volume.
    """
    
    resampled_tumor_mask = nibabel.processing.resample_from_to(tumor_mask_img, orig_img, order=0, mode='constant', cval=0)
    #nib.save(resampled_tumor_mask, os.path.join(subj_output_dir, 'tumor_mask_conf.mgz'))

    if (resampled_tumor_mask.get_fdata() == 0).all():
        logger.info('Tumor mask is all zeros, skipping mask volume')
        return orig_img
    elif (resampled_tumor_mask.get_fdata() > 0).all():
        logger.warning('Tumor mask is greater than 0 everywhere, returning all zeros')
        zeros_data = np.zeros(orig_img.shape, dtype=orig_dtype)
        # Use the same image class as input with original header (like reference script)
        zeros_img = orig_img.__class__(zeros_data, orig_img.affine, orig_img.header)
        return zeros_img

    #mask_volume
    assert(resampled_tumor_mask.shape == orig_img.shape), 'Shape mismatch between tumor mask and orig image ' + str(resampled_tumor_mask.shape) + ' vs ' + str(orig_img.shape)
    assert((resampled_tumor_mask.affine == orig_img.affine).all()), 'Affine mismatch between tumor mask and orig image ' + str(resampled_tumor_mask.affine) + ' vs ' + str(orig_img.affine)
    #assert((np.unique(resampled_tumor_mask.get_fdata()) == [0,1]).all()), 'Tumor mask should be binary, but has values: ' + str(np.unique(resampled_tumor_mask.get_fdata()))
    #masked_orig = orig_img.get_fdata() * (resampled_tumor_mask.get_fdata() == 0).astype(int) # invert and mask
    
    # Load data with np.asanyarray (preserves dtype) and modify
    # Pattern from FastSurfer's paint_cc_into_pred.py
    masked_orig = np.asanyarray(orig_img.dataobj).copy()
    masked_orig[resampled_tumor_mask.get_fdata() > 0] = 99
    
    # Ensure the output maintains the original data type
    masked_orig = masked_orig.astype(orig_dtype)

    # Use the same image class as input with original affine and header
    # This pattern preserves the dtype from the original file
    new_img = orig_img.__class__(masked_orig, orig_img.affine, orig_img.header)
    
    return new_img


def main(image: str, mask: str, output: str) -> None:
    """
    Main function to mask lesion in segmentation.
    
    Parameters
    ----------
    image : str
        Path to volume to mask
    mask : str
        Path to tumor mask
    output : str
        Path to output masked volume
    """
    masked_img = mask_lesion(image, mask)
    nib.save(masked_img, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask tumor from a volume')
    parser.add_argument('-i','--image', help='Path to volume to mask', type=str, required=True)
    parser.add_argument('-m','--mask', help='Path to tumor mask', type=str, required=True)
    parser.add_argument('-o','--output', help='Path to output masked volume', type=str, required=True)
    args = parser.parse_args()
    
    main(args.image, args.mask, args.output)
