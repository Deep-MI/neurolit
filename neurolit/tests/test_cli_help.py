import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neurolit.cli as cli
from neurolit.inpaint_image import resolve_inference_device


def run_help_test(module_path):
    """Helper to run --help on a module and check exit code."""
    result = subprocess.run(
        [sys.executable, "-m", module_path, "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "help" in result.stdout.lower() or "usage" in result.stdout.lower()
    return result

def test_lit_inpainting_help():
    """Test lit-inpainting (neurolit.cli) help."""
    # Note: lit-inpainting entrypoint calls neurolit.cli:run_lit.
    # Testing the module directly:
    result = run_help_test("neurolit.cli")
    assert "--keepgeom" in result.stdout
    assert "--fastsurfer_dir" in result.stdout
    assert "--device" in result.stdout
    assert "auto, cpu, or cuda" in result.stdout


def test_inpaint_image_help():
    """Test neurolit.inpaint_image help includes keepgeom."""
    result = run_help_test("neurolit.inpaint_image")
    assert "--keepgeom" in result.stdout

def test_lesion_to_segmentation_help():
    """Test lit-lesion-to-segmentation help."""
    run_help_test("neurolit.postprocessing.lesion_to_segmentation")

def test_lesion_to_surface_help():
    """Test lit-lesion-to-surface help."""
    run_help_test("neurolit.postprocessing.lesion_to_surface")


def test_resolve_inference_device_auto_prefers_cuda(monkeypatch):
    """Auto should resolve to CUDA when available."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_inference_device("auto").type == "cuda"


def test_resolve_inference_device_auto_falls_back_to_cpu(monkeypatch):
    """Auto should resolve to CPU when CUDA is unavailable."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_inference_device("auto").type == "cpu"


def test_resolve_inference_device_rejects_unavailable_cuda(monkeypatch):
    """Explicit CUDA should fail when no CUDA device is available."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_inference_device("cuda")


def test_lit_inpainting_fastsurfer_dir_materializes_outputs(tmp_path, monkeypatch):
    """FastSurfer mode should write public outputs in the subject directory."""
    input_image = tmp_path / "input.nii.gz"
    mask_image = tmp_path / "mask.nii.gz"
    subject_dir = tmp_path / "subject"
    mask_data = np.zeros((4, 4, 4), dtype=np.float32)
    mask_data[1:3, 1:3, 1:3] = 1
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), input_image)
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_image)

    user_data_root = tmp_path / "user-data"
    weights_dir = user_data_root / "weights"
    weights_dir.mkdir(parents=True)
    for name in ("model_coronal.pt", "model_axial.pt", "model_sagittal.pt"):
        (weights_dir / name).write_bytes(b"stub")

    monkeypatch.setattr(cli, "download_main", lambda argv=None: None)
    monkeypatch.setattr(cli, "user_data_dir", lambda *args, **kwargs: str(user_data_root))

    def fake_inpaint_main(argv: list[str]) -> None:
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        assert out_dir == subject_dir / "inpainting"
        volumes_dir = out_dir / "inpainting_volumes"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(np.full((4, 4, 4), 7, dtype=np.float32), np.eye(4)),
            volumes_dir / "inpainting_result.nii.gz",
        )

    monkeypatch.setattr(cli, "inpaint_main", fake_inpaint_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lit-inpainting",
            "--input_image",
            str(input_image),
            "--lesion_mask",
            str(mask_image),
            "--sd",
            str(subject_dir),
            "--fastsurfer_dir",
        ],
    )

    cli.run_lit()

    public_result = subject_dir / "mri" / "inpainted.lit.nii.gz"
    public_mask = subject_dir / "mri" / "orig" / "mask.lit.nii.gz"
    assert public_result.exists()
    assert public_mask.exists()
    np.testing.assert_allclose(nib.load(str(public_result)).get_fdata(), 7.0)
    np.testing.assert_allclose(
        nib.load(str(public_mask)).get_fdata(),
        nib.load(str(mask_image)).get_fdata(),
    )
