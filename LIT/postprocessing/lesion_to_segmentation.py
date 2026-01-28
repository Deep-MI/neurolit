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
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np
from scipy import ndimage

from LIT.utils.logging import get_logger

logger = get_logger(__name__)


def read_lut(lut_path: str) -> dict[int, str]:
    """
    Read FreeSurfer lookup table.

    Parameters
    ----------
    lut_path : str
        Path to FreeSurfer color lookup table file

    Returns
    -------
    lut_dict : Dict[int, str]
        Dictionary mapping label ID to label name
    """
    lut_dict = {}

    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    label_id = int(parts[0])
                    label_name = parts[1]
                    lut_dict[label_id] = label_name
                except (ValueError, IndexError):
                    continue

    return lut_dict


def get_anatomy_info(orig_data: np.ndarray, target_mask: np.ndarray) -> tuple[set[int], set[int], set[int]]:
    """
    Categorize labels from original segmentation based on their relationship with the lesion mask.

    Parameters
    ----------
    orig_data : np.ndarray
        Original segmentation data array.
    target_mask : np.ndarray (bool)
        Binary mask of the lesion.

    Returns
    -------
    replaced : Set[int]
        Labels fully replaced by the lesion mask.
    reduced : Set[int]
        Labels partially replaced by the lesion mask.
    adjacent : Set[int]
        Labels touching the lesion mask but not overlapping with it.
    """
    # 6-connectivity for adjacency
    struct = ndimage.generate_binary_structure(3, 1)
    dilated_mask = ndimage.binary_dilation(target_mask, structure=struct, iterations=1)
    adjacent_zone = dilated_mask & ~target_mask

    labels_in_mask = set(np.unique(orig_data[target_mask]))
    labels_outside_mask = set(np.unique(orig_data[~target_mask]))
    labels_in_adj_zone = set(np.unique(orig_data[adjacent_zone]))

    # Remove background
    labels_in_mask.discard(0)
    labels_outside_mask.discard(0)
    labels_in_adj_zone.discard(0)

    # Categorize
    replaced = labels_in_mask - labels_outside_mask
    reduced = labels_in_mask & labels_outside_mask
    adjacent = labels_in_adj_zone - labels_in_mask

    return replaced, reduced, adjacent


def write_report(
    output_path: str,
    replaced: set[int],
    reduced: set[int],
    adjacent: set[int],
    lut_dict: dict[int, str] | None = None
):
    """
    Write anatomy report to text file.

    Parameters
    ----------
    output_path : str
        Path to output text file.
    replaced : Set[int]
        Set of replaced label IDs.
    reduced : Set[int]
        Set of reduced label IDs.
    adjacent : Set[int]
        Set of adjacent label IDs.
    lut_dict : Dict[int, str], optional
        Lookup table for label names.
    """
    with open(output_path, 'w') as f:
        f.write("# Lesion Anatomy Report\n")
        f.write("#" + "=" * 60 + "\n")
        f.write(f"# Number of replaced labels: {len(replaced)}\n")
        f.write(f"# Number of reduced labels:  {len(reduced)}\n")
        f.write(f"# Number of adjacent labels: {len(adjacent)}\n")
        f.write("#" + "=" * 60 + "\n\n")

        categories = [
            ("Replaced Labels (fully covered by lesion)", replaced),
            ("Reduced Labels (partially covered by lesion)", reduced),
            ("Adjacent Labels (touching lesion boundary)", adjacent)
        ]

        for title, labels in categories:
            f.write(f"# {title}\n")
            f.write("#" + "-" * 60 + "\n")
            if not labels:
                f.write("# None found\n")
            else:
                if lut_dict:
                    f.write("# LabelID    LabelName\n")
                    for lid in sorted(labels):
                        name = lut_dict.get(lid, "Unknown")
                        f.write(f"{lid:8d}    {name}\n")
                else:
                    f.write("# LabelID\n")
                    for lid in sorted(labels):
                        f.write(f"{lid:8d}\n")
            f.write("\n")

    logger.info(f"Anatomy report written to: {output_path}")


def mask_lesion(to_mask_path, mask_path, report_path=None, lut_path=None):
    """Mask the provided image volume with the tumor segmentation.

    Parameters
    ----------
    to_mask_path : str
        Path to the volume that should have the lesion masked out.
    mask_path : str
        Path to the tumor mask volume.
    report_path : str, optional
        Path to write the anatomy report.
    lut_path : str, optional
        Path to FreeSurfer color lookup table.

    Returns
    -------
    nibabel.spatialimages.SpatialImage
        Masked image volume.
    """
    tumor_mask_img = nib.load(mask_path)
    
    orig_img = nib.load(to_mask_path)
    
    # Get the original data type to preserve it
    orig_dtype = orig_img.get_data_dtype()
    
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
    assert (
        resampled_tumor_mask.shape == orig_img.shape
    ), (
        'Shape mismatch between tumor mask and orig image '
        + str(resampled_tumor_mask.shape)
        + ' vs '
        + str(orig_img.shape)
    )
    assert (
        np.allclose(resampled_tumor_mask.affine, orig_img.affine)
    ), (
        'Affine mismatch between tumor mask and orig image '
        + str(resampled_tumor_mask.affine)
        + ' vs '
        + str(orig_img.affine)
    )
    
    # Load data with np.asanyarray (preserves dtype) and modify
    # Pattern from FastSurfer's paint_cc_into_pred.py
    orig_data = np.asanyarray(orig_img.dataobj)
    masked_orig = orig_data.copy()
    
    target_mask = resampled_tumor_mask.get_fdata() > 0
    masked_orig[target_mask] = 99
    
    if report_path:
        logger.info("Generating anatomy report...")
        replaced, reduced, adjacent = get_anatomy_info(orig_data, target_mask)
        
        lut_dict = None
        if lut_path:
            if Path(lut_path).exists():
                lut_dict = read_lut(lut_path)
            else:
                logger.warning(f"LUT file not found: {lut_path}. Proceeding without label names.")
                
        write_report(report_path, replaced, reduced, adjacent, lut_dict)

    # Ensure the output maintains the original data type
    masked_orig = masked_orig.astype(orig_dtype)

    # Use the same image class as input with original affine and header
    # This pattern preserves the dtype from the original file
    new_img = orig_img.__class__(masked_orig, orig_img.affine, orig_img.header)
    
    return new_img


def main(image: str, mask: str, output: str, report: str = None, lut: str = None) -> None:
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
    report : str, optional
        Path to output anatomy report
    lut : str, optional
        Path to FreeSurfer lookup table
    """
    masked_img = mask_lesion(image, mask, report_path=report, lut_path=lut)
    nib.save(masked_img, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask tumor from a volume and optionally generate anatomy report')
    parser.add_argument('-i','--image', help='Path to volume to mask', type=str, required=True)
    parser.add_argument('-m','--mask', help='Path to tumor mask', type=str, required=True)
    parser.add_argument('-o','--output', help='Path to output masked volume', type=str, required=True)
    parser.add_argument('-r','--report', help='Path to output anatomy report (optional)', type=str)
    parser.add_argument('-l','--lut', help='Path to FreeSurfer lookup table (optional, for report)', type=str)
    args = parser.parse_args()
    
    main(args.image, args.mask, args.output, args.report, args.lut)
