"""Regression checks for saved intensities outside the lesion (FastSurfer #933)."""

import nibabel as nib
import numpy as np
import pytest

from neurolit.utils.inference_io import make_inpainting_output


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_output_preserves_unmasked_voxels_after_save(tmp_path, dtype):
    original = np.linspace(10, 240, 64).reshape(4, 4, 4).astype(dtype)
    source = nib.Nifti1Image(original, np.eye(4))
    mask = np.zeros(source.shape, dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    prediction = np.full(source.shape, 0.5)
    prediction[1, 1, 1] = 2.0  # An overshoot must not trigger global integer autoscaling.
    path = tmp_path / "result.nii.gz"

    nib.save(make_inpainting_output(prediction, mask, source), path)
    result = nib.load(path)
    data = np.asanyarray(result.dataobj)

    np.testing.assert_array_equal(data[~mask], original[~mask])
    np.testing.assert_array_equal(result.affine, source.affine)
    assert result.get_data_dtype() == original.dtype
    assert data[2, 2, 2] == 125  # Inverse min/max scaling, not 0.5 * 255.
    assert data[1, 1, 1] == (255 if dtype == np.uint8 else 470)


def test_keepgeom_preserves_native_values_and_restores_generated_scale(tmp_path):
    original = np.arange(512, dtype=np.float32).reshape(8, 8, 8) + 0.125
    reference = nib.Nifti1Image(original, np.eye(4))
    source = nib.Nifti1Image(np.arange(64, dtype=np.uint8).reshape(4, 4, 4), np.diag([2, 2, 2, 1]))
    mask = np.zeros(source.shape, dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    # With 2 mm source spacing, nearest-neighbor samples at native indices 1..4
    # land on source indices 1 or 2 (halfway samples round upward).
    effective_mask = np.zeros(reference.shape, dtype=bool)
    effective_mask[1:5, 1:5, 1:5] = True
    path = tmp_path / "native.nii.gz"

    nib.save(make_inpainting_output(np.full(source.shape, 0.5), mask, source, reference, (10, 0.5)), path)
    result = nib.load(path)
    data = np.asanyarray(result.dataobj)

    np.testing.assert_array_equal(data[~effective_mask], original[~effective_mask])
    np.testing.assert_allclose(data[effective_mask], 73)  # (0.5 * 63) / 0.5 + 10.
    np.testing.assert_array_equal(result.affine, reference.affine)


def test_scaled_integer_reference_preserves_loaded_intensities(tmp_path):
    source = nib.Nifti1Image(np.arange(64, dtype=np.int16).reshape(4, 4, 4), np.eye(4))
    source.header.set_slope_inter(1.25, -3.5)
    source_path = tmp_path / "scaled.nii.gz"
    nib.save(source, source_path)
    reference = nib.load(source_path)
    original = np.asanyarray(reference.dataobj)
    mask = np.zeros(source.shape, dtype=bool)
    mask[1, 1, 1] = True
    output_path = tmp_path / "output.nii.gz"

    nib.save(make_inpainting_output(np.full(source.shape, 0.5), mask, reference), output_path)

    np.testing.assert_array_equal(nib.load(output_path).get_fdata()[~mask], original[~mask])
