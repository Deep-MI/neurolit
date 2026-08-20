import pytest
import torch

from neurolit.inference import SliceWiseInpaintingInferer


@pytest.fixture
def inferer():
    """Create an inferer shell with the production slab thickness."""
    instance = object.__new__(SliceWiseInpaintingInferer)
    instance.slice_thickness = 7
    return instance


@pytest.mark.parametrize("volume_size", [1, 4, 7, 8, 255, 256, 257])
@pytest.mark.parametrize("offset", [0, 3])
def test_inference_slice_centers_cover_full_axis(inferer, volume_size, offset):
    centers = inferer.get_inference_slice_centers(volume_size, offset)
    covered_indices = set()

    for center in centers:
        _, _, valid_start, valid_end = inferer.get_slab_slices(center, 0, volume_size)
        covered_indices.update(range(valid_start, valid_end))

    assert covered_indices == set(range(volume_size))


def test_get_slice_from_volume_zero_pads_left_edge(inferer):
    volume = torch.arange(2 * 3 * 10).reshape(2, 3, 10)

    slab = inferer.get_slice_from_volume(volume, 0, 2)

    assert slab.shape == (2, 3, 7)
    torch.testing.assert_close(slab[..., :3], torch.zeros_like(slab[..., :3]))
    torch.testing.assert_close(slab[..., 3:], volume[..., :4])


def test_get_slice_from_volume_zero_pads_right_edge(inferer):
    volume = torch.arange(2 * 3 * 10).reshape(2, 3, 10)

    slab = inferer.get_slice_from_volume(volume, 9, 2)

    assert slab.shape == (2, 3, 7)
    torch.testing.assert_close(slab[..., :4], volume[..., 6:])
    torch.testing.assert_close(slab[..., 4:], torch.zeros_like(slab[..., 4:]))


def test_slab_slices_copy_only_valid_prediction_region(inferer):
    output = torch.zeros(10, 2, 3)
    prediction = torch.arange(7 * 2 * 3, dtype=output.dtype).reshape(7, 2, 3)
    volume_slice, slab_slice, _, _ = inferer.get_slab_slices(9, 0, output.shape[0])

    output[volume_slice] = prediction[slab_slice]

    torch.testing.assert_close(output[6:], prediction[:4])
    torch.testing.assert_close(output[:6], torch.zeros_like(output[:6]))
