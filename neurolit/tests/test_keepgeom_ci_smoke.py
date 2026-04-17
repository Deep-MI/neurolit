from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from neurolit import inpaint_image


class _DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.internal_size = (128, 128)
        self.conv_in = SimpleNamespace(spatial_dims=2)
        self.is_vinn = True

    def load_state_dict(self, state_dict):
        return None

    def to(self, device):
        return self

    def eval(self):
        return self


def _make_dummy_pair(image_path: Path, mask_path: Path, shape=(37, 53, 19), zooms=(0.8, 1.7, 2.9)):
    rng = np.random.default_rng(7)
    data = rng.normal(loc=110, scale=20, size=shape).astype(np.float32)

    xx, yy, zz = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rx, ry, rz = max(4, shape[0] // 6), max(4, shape[1] // 8), max(3, shape[2] // 5)
    mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2 <= 1.0).astype(np.float32)

    affine = np.array(
        [
            [zooms[0], 0.0, 0.0, -10.0],
            [0.0, zooms[1], 0.0, 5.0],
            [0.0, 0.0, zooms[2], -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    nib.save(nib.Nifti1Image(data, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)


def test_keepgeom_anisotropic_smoke_30_steps_ci(tmp_path, monkeypatch):
    tracker: dict[str, int] = {}

    class _DummyInferer:
        def __init__(self, *, inference_steps, scheduler, diffusion_model_dict):
            tracker["steps"] = int(inference_steps)
            self._models = diffusion_model_dict

        def __call__(
            self,
            mask,
            image_masked,
            num_resample_steps=10,
            num_resample_jumps=15,
            batch_size=8,
            get_intermediates=False,
            scale_factor=None,
        ):
            # Keep the tensor shape unchanged and return a deterministic result.
            return image_masked.clone()

    monkeypatch.setattr(inpaint_image, "DiffusionModelUNetVINN", _DummyModel)
    monkeypatch.setattr(inpaint_image, "OffsetTwoAndHalfDInpaintingInferer", _DummyInferer)
    monkeypatch.setattr(inpaint_image.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(inpaint_image, "plot_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(inpaint_image, "plot_inpainting", lambda *args, **kwargs: None)

    image_path = tmp_path / "odd_aniso_image.nii.gz"
    mask_path = tmp_path / "odd_aniso_mask.nii.gz"
    out_dir = tmp_path / "out"

    _make_dummy_pair(image_path, mask_path)

    inpaint_image.main(
        [
            "--input_image",
            str(image_path),
            "--mask_image",
            str(mask_path),
            "--out_dir",
            str(out_dir),
            "--checkpoint_coronal",
            "dummy_coronal.pt",
            "--checkpoint_axial",
            "dummy_axial.pt",
            "--checkpoint_sagittal",
            "dummy_sagittal.pt",
            "--num_inference_steps",
            "30",
            "--keepgeom",
        ]
    )

    assert tracker["steps"] == 30

    output_path = out_dir / "inpainting_volumes" / "inpainting_result.nii.gz"
    intermediate_path = out_dir / "inpainting_volumes" / "inpainting_original_image.nii.gz"
    assert output_path.exists()
    assert intermediate_path.exists()

    src_img = nib.load(str(image_path))
    out_img = nib.load(str(output_path))
    intermediate_img = nib.load(str(intermediate_path))

    assert out_img.shape[:3] == src_img.shape[:3]
    np.testing.assert_allclose(out_img.affine, src_img.affine, atol=1e-6)
    assert nib.orientations.aff2axcodes(src_img.affine) == ("R", "A", "S")
    assert nib.orientations.aff2axcodes(out_img.affine) == nib.orientations.aff2axcodes(src_img.affine)

    # In keepgeom mode, inference runs in conformed/model space, while final
    # output is resampled back to native geometry.
    assert intermediate_img.shape[:3] != src_img.shape[:3]
