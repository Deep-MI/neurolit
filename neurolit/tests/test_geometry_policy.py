import pytest

from neurolit.utils.geometry_policy import (
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