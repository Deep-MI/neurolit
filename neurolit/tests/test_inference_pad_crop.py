import torch

from neurolit.utils.inference_io import crop_after_inference, pad_for_inference


def test_pad_and_crop_identity_for_odd_shapes():
    image = torch.randn(1, 97, 103, 111)
    mask = (torch.rand(1, 97, 103, 111) > 0.5).float()
    masked = image * (1 - mask)

    image_padded, mask_padded, masked_padded, metadata = pad_for_inference(image, mask, masked, multiple=16)

    assert image_padded.shape[-3:] == (112, 112, 112)
    assert mask_padded.shape == image_padded.shape
    assert masked_padded.shape == image_padded.shape

    image_cropped = crop_after_inference(image_padded, metadata)
    mask_cropped = crop_after_inference(mask_padded, metadata)

    assert image_cropped.shape == image.shape
    assert mask_cropped.shape == mask.shape
    torch.testing.assert_close(image_cropped, image)
    torch.testing.assert_close(mask_cropped, mask)


def test_pad_noop_for_aligned_shape():
    image = torch.randn(1, 96, 112, 128)
    mask = torch.zeros_like(image)
    masked = image.clone()

    image_padded, mask_padded, masked_padded, metadata = pad_for_inference(image, mask, masked, multiple=16)

    assert image_padded.shape == image.shape
    assert mask_padded.shape == mask.shape
    assert masked_padded.shape == masked.shape
    assert metadata.pad_spec == (0, 0, 0, 0, 0, 0)

    cropped = crop_after_inference(image_padded, metadata)
    torch.testing.assert_close(cropped, image)
