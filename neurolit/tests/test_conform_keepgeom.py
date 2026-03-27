import nibabel as nib
import numpy as np

from neurolit.data import conform


def _make_image(shape=(37, 41, 29), zooms=(1.0, 1.2, 2.5)):
    data = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
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
