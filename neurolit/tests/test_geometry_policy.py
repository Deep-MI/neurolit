import nibabel as nib
import numpy as np
import pytest

from neurolit.data.conform import conformed_vox_img_size
from neurolit.utils.geometry_policy import (
    FASTSURFER_CONFORM_TO_1MM_THRESHOLD,
    FASTSURFER_MIN_AUTO_IMG_SIZE,
    VINN_REFERENCE_FOV_MM,
    default_conform_kwargs,
    vinn_internal_resolution_mm,
    vinn_scale_factor_from_zooms,
)


def test_default_conform_kwargs_match_expected_defaults():
    kwargs = default_conform_kwargs()

    assert kwargs["vox_size"] == "min"
    assert kwargs["img_size"] == "auto"
    assert kwargs["orientation"] == "lia"
    assert kwargs["rescale"] == 255
    assert kwargs["dtype"] is None
    assert kwargs["threshold_1mm"] == FASTSURFER_CONFORM_TO_1MM_THRESHOLD
    assert kwargs["min_auto_img_size"] is None


def test_auto_conform_standalone_uses_input_fov():
    image = nib.Nifti1Image(np.zeros((160, 255, 189), dtype=np.uint8), np.eye(4))

    voxel_size, image_size = conformed_vox_img_size(
        image,
        vox_size="min",
        img_size="auto",
        threshold_1mm=FASTSURFER_CONFORM_TO_1MM_THRESHOLD,
    )

    np.testing.assert_array_equal(voxel_size, (1.0, 1.0, 1.0))
    np.testing.assert_array_equal(image_size, (255, 255, 255))


def test_auto_conform_uses_fastsurfer_minimum_when_requested():
    image = nib.Nifti1Image(np.zeros((160, 255, 189), dtype=np.uint8), np.eye(4))

    voxel_size, image_size = conformed_vox_img_size(
        image,
        vox_size="min",
        img_size="auto",
        threshold_1mm=FASTSURFER_CONFORM_TO_1MM_THRESHOLD,
        min_auto_img_size=FASTSURFER_MIN_AUTO_IMG_SIZE,
    )

    np.testing.assert_array_equal(voxel_size, (1.0, 1.0, 1.0))
    np.testing.assert_array_equal(image_size, (256, 256, 256))


def test_auto_conform_rejects_nonpositive_minimum():
    image = nib.Nifti1Image(np.zeros((160, 160, 160), dtype=np.uint8), np.eye(4))

    with pytest.raises(ValueError, match="min_auto_img_size must be > 0"):
        conformed_vox_img_size(image, vox_size="min", img_size="auto", min_auto_img_size=0)


def test_auto_conform_preserves_larger_high_resolution_fov():
    affine = np.diag((0.7, 0.7, 0.7, 1.0))
    image = nib.Nifti1Image(np.zeros((360, 350, 340), dtype=np.uint8), affine)

    voxel_size, image_size = conformed_vox_img_size(
        image,
        vox_size="min",
        img_size="auto",
        threshold_1mm=FASTSURFER_CONFORM_TO_1MM_THRESHOLD,
    )

    np.testing.assert_allclose(voxel_size, (0.7, 0.7, 0.7))
    np.testing.assert_array_equal(image_size, (360, 360, 360))


def test_vinn_internal_resolution_mm_matches_legacy_formula():
    internal_shape = (128, 128)
    legacy = 256 / internal_shape[0]

    assert vinn_internal_resolution_mm(internal_shape) == legacy


def test_vinn_scale_factor_from_zooms_matches_legacy_formula():
    internal_shape = (128, 128)
    zooms = (1.0, 1.7, 2.9)
    legacy = (256 / internal_shape[0]) / zooms[0]

    assert vinn_scale_factor_from_zooms(internal_shape, zooms) == legacy


def test_vinn_internal_resolution_uses_policy_constant():
    expected = VINN_REFERENCE_FOV_MM / 128
    assert vinn_internal_resolution_mm((128, 128)) == expected


def test_vinn_scale_factor_rejects_invalid_axis():
    with pytest.raises(ValueError):
        vinn_scale_factor_from_zooms((128, 128), (1.0, 1.0, 1.0), axis=3)
