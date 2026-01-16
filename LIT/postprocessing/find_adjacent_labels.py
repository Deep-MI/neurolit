#!/usr/bin/env python3

# Copyright 2025 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

"""
Script to find adjacent labels in a segmentation volume.

Identifies all labels that are spatially adjacent (neighboring) to a target label
in a 3D segmentation volume. Useful for analyzing lesion-region relationships.

Author: Claude AI Assistant
Date: October 29, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Set, Optional

import numpy as np
import nibabel as nib
from scipy import ndimage

from LIT.utils.logging import get_logger

logger = get_logger(__name__)


def read_lut(lut_path: str) -> Dict[int, str]:
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
    
    with open(lut_path, 'r') as f:
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


def get_label_id_from_name(lut_dict: Dict[int, str], label_name: str) -> Optional[int]:
    """
    Find label ID from label name in lookup table.
    
    Parameters
    ----------
    lut_dict : Dict[int, str]
        Dictionary mapping label ID to label name
    label_name : str
        Label name to search for
        
    Returns
    -------
    label_id : int or None
        Label ID if found, None otherwise
    """
    for label_id, name in lut_dict.items():
        if name == label_name:
            return label_id
    return None


def dilate_label(seg_data: np.ndarray, label_id: int, iterations: int = 1) -> np.ndarray:
    """
    Dilate a specific label in the segmentation.
    
    Parameters
    ----------
    seg_data : np.ndarray
        3D segmentation array
    label_id : int
        Label ID to dilate
    iterations : int
        Number of dilation iterations (default: 1)
        
    Returns
    -------
    dilated_mask : np.ndarray (bool)
        Binary mask of dilated label
    """
    # Create binary mask for target label
    label_mask = (seg_data == label_id)
    
    if iterations <= 0:
        return label_mask
    
    # Perform binary dilation
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    dilated_mask = ndimage.binary_dilation(label_mask, structure=struct, iterations=iterations)
    
    return dilated_mask


def find_adjacent_labels(
    seg_data: np.ndarray,
    target_label: int,
    dilation: int = 0
) -> Set[int]:
    """
    Find all labels adjacent to the target label.
    
    Parameters
    ----------
    seg_data : np.ndarray
        3D segmentation array
    target_label : int
        Target label ID to find neighbors of
    dilation : int
        Number of dilation iterations before finding neighbors (default: 0)
        
    Returns
    -------
    adjacent_labels : Set[int]
        Set of label IDs adjacent to target label
    """
    logger.info(f"Finding labels adjacent to label ID {target_label}...")
    
    # Check if target label exists
    if target_label not in seg_data:
        logger.warning(f"Target label {target_label} not found in segmentation!")
        return set()
    
    # Get target label mask (possibly dilated)
    if dilation > 0:
        logger.info(f"  Dilating target label {dilation} iteration(s)...")
        target_mask = dilate_label(seg_data, target_label, iterations=dilation)
    else:
        target_mask = (seg_data == target_label)
    
    target_voxels = np.sum(target_mask)
    logger.info(f"  Target region contains {target_voxels:,} voxels")
    
    # Create a dilated boundary around the target
    # This gives us the "adjacency zone"
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    boundary_mask = ndimage.binary_dilation(target_mask, structure=struct, iterations=1)
    
    # The adjacent zone is the dilated boundary minus the target itself
    adjacent_zone = boundary_mask & ~target_mask
    
    # Find all unique labels in the adjacent zone
    adjacent_labels = set(seg_data[adjacent_zone].tolist())
    
    # Remove background (0) if present
    adjacent_labels.discard(0)
    
    # Remove target label if somehow included
    adjacent_labels.discard(target_label)
    
    logger.info(f"  Found {len(adjacent_labels)} adjacent label(s)")
    
    return adjacent_labels


def write_results(
    output_path: str,
    target_label: int,
    adjacent_labels: Set[int],
    lut_dict: Optional[Dict[int, str]] = None,
    target_name: Optional[str] = None,
    dilation: int = 0
):
    """
    Write results to output file.
    
    Parameters
    ----------
    output_path : str
        Path to output text file
    target_label : int
        Target label ID
    adjacent_labels : Set[int]
        Set of adjacent label IDs
    lut_dict : Dict[int, str], optional
        Lookup table for label names
    target_name : str, optional
        Name of target label
    dilation : int
        Dilation factor used
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Adjacent Labels Report\n")
        f.write("#" + "=" * 60 + "\n")
        
        if target_name:
            f.write(f"# Target Label: {target_label} ({target_name})\n")
        else:
            f.write(f"# Target Label: {target_label}\n")
        
        if dilation > 0:
            f.write(f"# Dilation: {dilation} iteration(s)\n")
        
        f.write(f"# Number of adjacent labels: {len(adjacent_labels)}\n")
        f.write("#" + "=" * 60 + "\n\n")
        
        if not adjacent_labels:
            f.write("# No adjacent labels found\n")
            return
        
        # Write column headers
        if lut_dict:
            f.write("# LabelID    LabelName\n")
            f.write("#" + "-" * 60 + "\n")
        else:
            f.write("# LabelID\n")
            f.write("#" + "-" * 60 + "\n")
        
        # Sort labels for consistent output
        sorted_labels = sorted(adjacent_labels)
        
        # Write each adjacent label
        for label_id in sorted_labels:
            if lut_dict and label_id in lut_dict:
                label_name = lut_dict[label_id]
                f.write(f"{label_id:8d}    {label_name}\n")
            else:
                f.write(f"{label_id:8d}\n")
    
    logger.info(f"Results written to: {output_path}")


def main():
    """CLI entry point for finding labels adjacent to a target label.

    Parses arguments, loads the segmentation, finds adjacent labels,
    and writes a report to disk.
    """
    parser = argparse.ArgumentParser(
        description='Find labels adjacent to a target label in a segmentation volume.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find labels adjacent to lesion label (99)
  %(prog)s -i aparc+aseg.mgz -t 99 -o adjacent_to_lesion.txt
  
  # With label names from lookup table
  %(prog)s -i aparc+aseg.mgz -t 99 -l FreeSurferColorLUT.txt -o adjacent_to_lesion.txt
  
  # Dilate lesion 3 times before finding neighbors
  %(prog)s -i aparc+aseg.mgz -t 99 -d 3 -o adjacent_to_lesion_dilated.txt
  
  # Find using label name instead of ID
  %(prog)s -i aparc+aseg.mgz -n Left-Hippocampus -l FreeSurferColorLUT.txt -o adjacent_to_hippo.txt
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        type=str,
        help='Input segmentation file (.mgz or .nii.gz)'
    )
    
    parser.add_argument(
        '-t', '--target-label',
        type=int,
        help='Target label ID to find adjacent labels for'
    )
    
    parser.add_argument(
        '-n', '--target-name',
        type=str,
        help='Target label name (requires --lut). Alternative to --target-label'
    )
    
    parser.add_argument(
        '-l', '--lut',
        type=str,
        help='FreeSurfer color lookup table file (optional)'
    )
    
    parser.add_argument(
        '-d', '--dilate',
        type=int,
        default=0,
        help='Number of dilation iterations to apply to target label before finding neighbors (default: 0)'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='Output text file with adjacent label IDs (and names if LUT provided)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.target_label is None and args.target_name is None:
        parser.error("Either --target-label or --target-name must be specified")
    
    if args.target_name and not args.lut:
        parser.error("--target-name requires --lut to be specified")
    
    if args.target_label is not None and args.target_name is not None:
        parser.error("Cannot specify both --target-label and --target-name")
    
    # Check input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("FIND ADJACENT LABELS")
    logger.info("=" * 70)
    
    # Load lookup table if provided
    lut_dict = None
    if args.lut:
        if not Path(args.lut).exists():
            logger.error(f"Lookup table not found: {args.lut}")
            sys.exit(1)
        
        logger.info(f"Loading lookup table: {args.lut}")
        lut_dict = read_lut(args.lut)
        logger.info(f"  Loaded {len(lut_dict)} label definitions")
    
    # Determine target label ID
    if args.target_name:
        logger.info(f"Searching for label name: {args.target_name}")
        target_label = get_label_id_from_name(lut_dict, args.target_name)
        if target_label is None:
            logger.error(f"Label name '{args.target_name}' not found in lookup table")
            sys.exit(1)
        logger.info(f"  Found label ID: {target_label}")
        target_name = args.target_name
    else:
        target_label = args.target_label
        target_name = lut_dict.get(target_label) if lut_dict else None
    
    # Load segmentation
    logger.info(f"Loading segmentation: {args.input}")
    seg_img = nib.load(args.input)
    seg_data = np.asarray(seg_img.dataobj, dtype=int)
    logger.info(f"  Shape: {seg_data.shape}")
    logger.info(f"  Contains {len(np.unique(seg_data))} unique labels")
    
    # Find adjacent labels
    adjacent_labels = find_adjacent_labels(seg_data, target_label, dilation=args.dilate)
    
    # Write results
    write_results(
        args.output,
        target_label,
        adjacent_labels,
        lut_dict=lut_dict,
        target_name=target_name,
        dilation=args.dilate
    )
    
    # Print summary to console
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    if target_name:
        logger.info(f"Target: {target_label} ({target_name})")
    else:
        logger.info(f"Target: {target_label}")
    
    if args.dilate > 0:
        logger.info(f"Dilation: {args.dilate} iteration(s)")
    
    logger.info(f"Adjacent labels found: {len(adjacent_labels)}")
    
    if adjacent_labels and lut_dict:
        logger.info("Adjacent regions:")
        for label_id in sorted(adjacent_labels)[:10]:  # Show first 10
            label_name = lut_dict.get(label_id, "Unknown")
            logger.info(f"  {label_id:4d}: {label_name}")
        if len(adjacent_labels) > 10:
            logger.info(f"  ... and {len(adjacent_labels) - 10} more")
    
    logger.info("=" * 70)
    logger.info("✓ Complete!")


if __name__ == '__main__':
    main()

