import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neurolit.cli as cli
from neurolit.inpaint_image import resolve_inference_device
from neurolit.scripts.lesion_postprocessing import resolve_inpainting_mask_path


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
    assert "--min-auto-img-size" in result.stdout
    assert "--fastsurfer_dir" in result.stdout
    assert "--device" in result.stdout
    assert "auto, cpu, or cuda" in result.stdout


def test_inpaint_image_help():
    """Test neurolit.inpaint_image help includes keepgeom."""
    result = run_help_test("neurolit.inpaint_image")
    assert "--keepgeom" in result.stdout
    assert "--min-auto-img-size" in result.stdout

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


def test_postprocessing_resolves_fastsurfer_mask_path_only(tmp_path):
    """Postprocessing should require the FastSurfer-mode public mask path."""
    subject_id = "subject"
    subject_dir = tmp_path / subject_id
    legacy_mask = subject_dir / "inpainting" / "inpainting_volumes" / "inpainting_mask.nii.gz"
    legacy_mask.parent.mkdir(parents=True)
    legacy_mask.write_bytes(b"legacy")

    resolved = resolve_inpainting_mask_path(tmp_path, subject_id)

    assert resolved == subject_dir / "mri" / "mask.lit.nii.gz"
    assert resolved != legacy_mask


def test_copy_file_same_path_is_noop(tmp_path):
    """Copying a path onto itself should not delete the source."""
    source = tmp_path / "source.txt"
    source.write_text("content")

    cli._copy_file(source, source)

    assert source.read_text() == "content"


def test_lit_inpainting_standalone_does_not_force_minimum_size(tmp_path, monkeypatch):
    """Standalone inpainting should retain field-of-view-based automatic sizing."""
    input_image = tmp_path / "input.nii.gz"
    mask_image = tmp_path / "mask.nii.gz"
    out_dir = tmp_path / "output"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), input_image)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), mask_image)

    user_data_root = tmp_path / "user-data"
    weights_dir = user_data_root / "weights"
    weights_dir.mkdir(parents=True)
    for name in ("model_coronal.pt", "model_axial.pt", "model_sagittal.pt"):
        (weights_dir / name).write_bytes(b"stub")

    monkeypatch.setattr(cli, "download_main", lambda argv=None: None)
    monkeypatch.setattr(cli, "user_data_dir", lambda *args, **kwargs: str(user_data_root))
    calls: list[list[str]] = []

    def fake_inpaint_main(argv: list[str]) -> None:
        calls.append(argv)
        volumes_dir = out_dir / "inpainting_volumes"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), volumes_dir / "inpainting_result.nii.gz")

    monkeypatch.setattr(cli, "inpaint_main", fake_inpaint_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lit-inpainting", "--input_image", str(input_image), "--lesion_mask", str(mask_image), "--sd", str(out_dir)],
    )

    cli.run_lit()

    assert len(calls) == 1
    assert "--min_auto_img_size" not in calls[0]


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
        assert argv[argv.index("--min_auto_img_size") + 1] == "256"
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        assert out_dir != subject_dir
        volumes_dir = out_dir / "inpainting_volumes"
        images_dir = out_dir / "inpainting_images"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(np.full((4, 4, 4), 7, dtype=np.float32), np.eye(4)),
            volumes_dir / "inpainting_result.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(mask_data, np.eye(4)),
            volumes_dir / "inpainting_mask.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)),
            volumes_dir / "inpainting_original_image.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
            volumes_dir / "inpainting_masked_image.nii.gz",
        )
        (images_dir / "inpainting_result.png").write_bytes(b"png")
        (images_dir / "inpainting_mask.png").write_bytes(b"png")
        (images_dir / "inpainting_original_image.png").write_bytes(b"png")
        (images_dir / "inpainting_masked_image.png").write_bytes(b"png")

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
    processed_mask = subject_dir / "mri" / "mask.lit.nii.gz"
    public_mask = subject_dir / "mri" / "orig" / "mask.lit.nii.gz"
    original_image = subject_dir / "mri" / "orig" / "inpainting_original_image.lit.nii.gz"
    masked_image = subject_dir / "mri" / "orig" / "inpainting_masked_image.lit.nii.gz"
    result_png = subject_dir / "scripts" / "inpainting_result.lit.png"
    assert public_result.exists()
    assert processed_mask.exists()
    assert public_mask.exists()
    assert original_image.exists()
    assert masked_image.exists()
    assert result_png.exists()
    assert not (subject_dir / "inpainting").exists()
    np.testing.assert_allclose(nib.load(str(public_result)).get_fdata(), 7.0)
    np.testing.assert_allclose(
        nib.load(str(processed_mask)).get_fdata(),
        mask_data,
    )
    np.testing.assert_allclose(
        nib.load(str(public_mask)).get_fdata(),
        nib.load(str(mask_image)).get_fdata(),
    )


def test_lit_inpainting_fastsurfer_dir_forwards_keepgeom(tmp_path, monkeypatch):
    """FastSurfer mode should pass keepgeom through to the inpainting backend."""
    input_image = tmp_path / "input.nii.gz"
    mask_image = tmp_path / "mask.nii.gz"
    subject_dir = tmp_path / "subject"
    native_affine = np.diag([1.0, 1.0, 1.2, 1.0])
    native_data = np.ones((4, 4, 3), dtype=np.float32)
    native_mask = np.zeros((4, 4, 3), dtype=np.float32)
    native_mask[1:3, 1:3, 1:2] = 1
    nib.save(nib.Nifti1Image(native_data, native_affine), input_image)
    nib.save(nib.Nifti1Image(native_mask, native_affine), mask_image)

    user_data_root = tmp_path / "user-data"
    weights_dir = user_data_root / "weights"
    weights_dir.mkdir(parents=True)
    for name in ("model_coronal.pt", "model_axial.pt", "model_sagittal.pt"):
        (weights_dir / name).write_bytes(b"stub")

    monkeypatch.setattr(cli, "download_main", lambda argv=None: None)
    monkeypatch.setattr(cli, "user_data_dir", lambda *args, **kwargs: str(user_data_root))

    def fake_inpaint_main(argv: list[str]) -> None:
        assert "--keepgeom" in argv
        assert argv[argv.index("--min_auto_img_size") + 1] == "256"
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        volumes_dir = out_dir / "inpainting_volumes"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(np.full(native_data.shape, 7, dtype=np.float32), native_affine),
            volumes_dir / "inpainting_result.nii.gz",
        )
        conformed_mask = np.zeros((5, 5, 5), dtype=np.uint8)
        conformed_mask[2:4, 2:4, 2:4] = 1
        nib.save(nib.Nifti1Image(conformed_mask, np.eye(4)), volumes_dir / "inpainting_mask.nii.gz")

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
            "--keepgeom",
        ],
    )

    cli.run_lit()

    processed_mask = nib.load(str(subject_dir / "mri" / "mask.lit.nii.gz"))
    assert processed_mask.shape == native_data.shape
    assert np.dtype(processed_mask.get_data_dtype()) == np.dtype(np.uint8)
    np.testing.assert_allclose(processed_mask.affine, native_affine)


def test_lit_inpainting_fastsurfer_dir_reruns_when_public_outputs_are_partial(tmp_path, monkeypatch):
    """FastSurfer mode should not skip when only the core public outputs exist."""
    input_image = tmp_path / "input.nii.gz"
    subject_dir = tmp_path / "subject"
    mask_image = subject_dir / "mri" / "orig" / "mask.lit.nii.gz"
    public_result = subject_dir / "mri" / "inpainted.lit.nii.gz"
    processed_mask = subject_dir / "mri" / "mask.lit.nii.gz"
    original_image = subject_dir / "mri" / "orig" / "inpainting_original_image.lit.nii.gz"
    masked_image = subject_dir / "mri" / "orig" / "inpainting_masked_image.lit.nii.gz"
    result_png = subject_dir / "scripts" / "inpainting_result.lit.png"
    mask_png = subject_dir / "scripts" / "inpainting_mask.lit.png"
    original_png = subject_dir / "scripts" / "inpainting_original_image.lit.png"
    masked_png = subject_dir / "scripts" / "inpainting_masked_image.lit.png"

    subject_dir.joinpath("mri", "orig").mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), input_image)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), mask_image)
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), 3, dtype=np.float32), np.eye(4)), public_result)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), processed_mask)

    user_data_root = tmp_path / "user-data"
    weights_dir = user_data_root / "weights"
    weights_dir.mkdir(parents=True)
    for name in ("model_coronal.pt", "model_axial.pt", "model_sagittal.pt"):
        (weights_dir / name).write_bytes(b"stub")

    monkeypatch.setattr(cli, "download_main", lambda argv=None: None)
    monkeypatch.setattr(cli, "user_data_dir", lambda *args, **kwargs: str(user_data_root))
    calls = []

    def fake_inpaint_main(argv: list[str]) -> None:
        calls.append(argv)
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        volumes_dir = out_dir / "inpainting_volumes"
        images_dir = out_dir / "inpainting_images"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full((4, 4, 4), 7, dtype=np.float32), np.eye(4)), volumes_dir / "inpainting_result.nii.gz")
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), volumes_dir / "inpainting_mask.nii.gz")
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), volumes_dir / "inpainting_original_image.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), volumes_dir / "inpainting_masked_image.nii.gz")
        (images_dir / "inpainting_result.png").write_bytes(b"result")
        (images_dir / "inpainting_mask.png").write_bytes(b"mask")
        (images_dir / "inpainting_original_image.png").write_bytes(b"original")
        (images_dir / "inpainting_masked_image.png").write_bytes(b"masked")

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

    assert len(calls) == 1
    for path in (public_result, processed_mask, mask_image, original_image, masked_image, result_png, mask_png, original_png, masked_png):
        assert path.exists()
    np.testing.assert_allclose(nib.load(str(public_result)).get_fdata(), 7.0)
