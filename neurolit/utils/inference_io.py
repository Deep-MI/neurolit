# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

import nibabel as nib
import nibabel.processing
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InferencePadCropMetadata:
    """Store deterministic padding metadata for inference tensor restoration."""

    original_shape: tuple[int, int, int]
    pad_spec: tuple[int, int, int, int, int, int]


def _compute_target_shape(shape: tuple[int, int, int], multiple: int) -> tuple[int, int, int]:
    """Compute a shape where each spatial axis is divisible by ``multiple``."""
    return tuple(((size + multiple - 1) // multiple) * multiple for size in shape)


def _normalize_min_shape(min_size: int | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    """Normalize minimum-size inputs to a 3D tuple."""
    if min_size is None:
        return None
    if isinstance(min_size, int):
        if min_size <= 0:
            raise ValueError("min_size must be > 0")
        return (min_size, min_size, min_size)
    if len(min_size) != 3:
        raise ValueError("min_size tuple must have length 3")
    if any(v <= 0 for v in min_size):
        raise ValueError("all min_size tuple values must be > 0")
    return tuple(int(v) for v in min_size)


def _compute_pad_spec(shape: tuple[int, int, int], target_shape: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    """Build symmetric pad spec for torch.nn.functional.pad on 3D spatial tensors."""
    pad_spec: list[int] = []
    for size, target in zip(reversed(shape), reversed(target_shape), strict=True):
        delta = target - size
        left = delta // 2
        right = delta - left
        pad_spec.extend([left, right])
    return tuple(pad_spec)


def pad_for_inference(
    image: torch.Tensor,
    mask: torch.Tensor,
    masked_image: torch.Tensor,
    *,
    multiple: int = 16,
    min_size: int | tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, InferencePadCropMetadata]:
    """Pad image/mask tensors jointly to inference-safe dimensions."""
    if multiple <= 0:
        raise ValueError("multiple must be > 0")

    spatial_shape = tuple(int(v) for v in image.shape[-3:])
    target_shape = _compute_target_shape(spatial_shape, multiple)
    min_shape = _normalize_min_shape(min_size)
    if min_shape is not None:
        target_shape = tuple(max(t, m) for t, m in zip(target_shape, min_shape, strict=True))
        target_shape = _compute_target_shape(target_shape, multiple)
    pad_spec = _compute_pad_spec(spatial_shape, target_shape)

    if not any(pad_spec):
        metadata = InferencePadCropMetadata(original_shape=spatial_shape, pad_spec=pad_spec)
        return image, mask, masked_image, metadata

    image_padded = F.pad(image, pad_spec, mode="constant", value=0)
    mask_padded = F.pad(mask, pad_spec, mode="constant", value=0)
    masked_image_padded = F.pad(masked_image, pad_spec, mode="constant", value=0)

    metadata = InferencePadCropMetadata(original_shape=spatial_shape, pad_spec=pad_spec)
    return image_padded, mask_padded, masked_image_padded, metadata


def crop_after_inference(inpainted: torch.Tensor, metadata: InferencePadCropMetadata) -> torch.Tensor:
    """Undo inference-time padding to recover the original spatial shape."""
    x0, x1 = metadata.pad_spec[4], metadata.pad_spec[5]
    y0, y1 = metadata.pad_spec[2], metadata.pad_spec[3]
    z0, z1 = metadata.pad_spec[0], metadata.pad_spec[1]

    x_end = inpainted.shape[-3] - x1 if x1 > 0 else inpainted.shape[-3]
    y_end = inpainted.shape[-2] - y1 if y1 > 0 else inpainted.shape[-2]
    z_end = inpainted.shape[-1] - z1 if z1 > 0 else inpainted.shape[-1]

    return inpainted[..., x0:x_end, y0:y_end, z0:z_end]


def resample_result_to_reference(
    inference_result: np.ndarray,
    inference_image: nib.analyze.SpatialImage,
    reference_image: nib.analyze.SpatialImage,
    *,
    order: int = 1,
) -> nib.Nifti1Image:
    """Resample inference-space result to a native/reference image grid."""
    inference_nib = nib.Nifti1Image(inference_result, inference_image.affine)
    resampled = nibabel.processing.resample_from_to(
        inference_nib,
        reference_image,
        order=order,
        mode="constant",
        cval=0,
    )
    return nib.Nifti1Image(np.asarray(resampled.dataobj), reference_image.affine, reference_image.header)
