import subprocess
import sys

import pytest

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

def test_lit_inpainting_help():
    """Test lit-inpainting (neurolit.cli) help."""
    # Note: lit-inpainting entrypoint calls neurolit.cli:run_lit.
    # Testing the module directly:
    result = subprocess.run(
        [sys.executable, "-m", "neurolit.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--device" in result.stdout
    assert "auto, cpu, or cuda" in result.stdout

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
