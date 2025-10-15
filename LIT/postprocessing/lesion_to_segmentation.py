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


def mask_lesion(to_mask_path, mask_path):
    
    tumor_mask_img = nib.load(mask_path)
    
    orig_img = nib.load(to_mask_path)
    resampled_tumor_mask = nibabel.processing.resample_from_to(tumor_mask_img, orig_img, order=0, mode='constant', cval=0)
    #nib.save(resampled_tumor_mask, os.path.join(subj_output_dir, 'tumor_mask_conf.mgz'))

    if (resampled_tumor_mask.get_fdata() == 0).all():
        print('Tumor mask is all zeros, skipping mask volume')
        return orig_img
    elif (resampled_tumor_mask.get_fdata() > 0).all():
        print('Tumor mask is greater than 0 everywhere, returning all zeros')
        return nib.Nifti1Image(np.zeros(orig_img.shape), orig_img.affine, orig_img.header)

    #mask_volume
    assert(resampled_tumor_mask.shape == orig_img.shape), 'Shape mismatch between tumor mask and orig image ' + str(resampled_tumor_mask.shape) + ' vs ' + str(orig_img.shape)
    assert((resampled_tumor_mask.affine == orig_img.affine).all()), 'Affine mismatch between tumor mask and orig image ' + str(resampled_tumor_mask.affine) + ' vs ' + str(orig_img.affine)
    #assert((np.unique(resampled_tumor_mask.get_fdata()) == [0,1]).all()), 'Tumor mask should be binary, but has values: ' + str(np.unique(resampled_tumor_mask.get_fdata()))
    #masked_orig = orig_img.get_fdata() * (resampled_tumor_mask.get_fdata() == 0).astype(int) # invert and mask
    
    # set tumor area to 99
    masked_orig = orig_img.get_fdata()
    masked_orig[resampled_tumor_mask.get_fdata() > 0] = 99

    return nib.Nifti1Image(masked_orig, orig_img.affine, orig_img.header)


def main():
    print('running')
    parser = argparse.ArgumentParser(description='Mask tumor from a volume')
    parser.add_argument('-i','--image', help='Path to volume to mask', type=str, required=True)
    parser.add_argument('-m','--mask', help='Path to tumor mask', type=str, required=True)
    parser.add_argument('-o','--output', help='Path to output masked volume', type=str, required=True)
    args = parser.parse_args()

    masked_img = mask_lesion(args.image, args.mask)
    nib.save(masked_img, args.output)

if __name__ == '__main__':
    main()
