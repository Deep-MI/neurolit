import nibabel as nib
import numpy as np
import pytest

from neurolit.data import conform


def _make_image(shape=(37, 41, 29), zooms=(1.0, 1.2, 2.5)):
    data = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    return nib.Nifti1Image(data, affine)


def _make_image_with_axcodes(
    axcodes: tuple[str, str, str],
    shape=(25, 27, 29),
    zooms=(0.9, 1.1, 2.2),
):
    rng = np.random.default_rng(11)
    ras_data = rng.normal(size=shape).astype(np.float32)
    ras_affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])

    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    target_ornt = nib.orientations.axcodes2ornt(axcodes)
    transform = nib.orientations.ornt_transform(ras_ornt, target_ornt)

    data = nib.orientations.apply_orientation(ras_data, transform)
    affine = ras_affine @ nib.orientations.inv_ornt_aff(transform, ras_data.shape)
    return nib.Nifti1Image(data, affine)


def test_conformed_vox_img_size_keepgeom_none():
    image = _make_image()
    vox_size, img_size = conform.conformed_vox_img_size(image, vox_size=None, img_size=None)

    assert vox_size is None
    assert img_size is None


def test_conform_keepgeom_preserves_affine_and_shape():
    image = _make_image(shape=(33, 35, 27), zooms=(0.9, 1.1, 2.2))

    output = conform.conform(
        image,
        vox_size=None,
        img_size=None,
        orientation="native",
        dtype=None,
        rescale=None,
    )

    assert output.shape[:3] == image.shape[:3]
    np.testing.assert_allclose(output.affine, image.affine, atol=1e-4)


@pytest.mark.parametrize("axcodes", [("R", "A", "S"), ("L", "I", "A"), ("R", "P", "S")])
def test_conform_keepgeom_native_preserves_orientation_codes(axcodes):
    image = _make_image_with_axcodes(axcodes)

    output = conform.conform(
        image,
        vox_size=None,
        img_size=None,
        orientation="native",
        dtype=None,
        rescale=None,
    )

    assert output.shape[:3] == image.shape[:3]
    assert nib.orientations.aff2axcodes(output.affine) == axcodes


def test_is_conform_native_keepgeom_criteria():
    image = _make_image(shape=(32, 34, 36), zooms=(1.0, 1.0, 2.0))

    assert conform.is_conform(
        image,
        vox_size=None,
        img_size=None,
        orientation="native",
        dtype=None,
        verbose=False,
    )
