import nibabel as nib
import numpy as np
import pytest

from neurolit.inpaint_image import conform_nifti


def _make_lia_image(dtype, shape=(32, 32, 32), zooms=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(5)
    ras_data = rng.normal(loc=100.0, scale=20.0, size=shape).astype(dtype)
    ras_affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])

    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    lia_ornt = nib.orientations.axcodes2ornt(("L", "I", "A"))
    transform = nib.orientations.ornt_transform(ras_ornt, lia_ornt)

    data = nib.orientations.apply_orientation(ras_data, transform)
    affine = ras_affine @ nib.orientations.inv_ornt_aff(transform, ras_data.shape)
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_conform_nifti_reconforms_non_uint8_lia_inputs(dtype):
    image = _make_lia_image(dtype)

    output = conform_nifti(image)

    assert output is not image
    assert output.shape[:3] == image.shape[:3]
    assert nib.orientations.aff2axcodes(output.affine) == ("L", "I", "A")
    assert np.dtype(output.get_data_dtype()) == np.dtype(np.uint8)
