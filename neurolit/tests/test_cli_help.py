import subprocess
import sys


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
    # Note: lit-inpainting entrypoint calls neurolit.cli:run_lit
    # which runs the bash script. Testing the module directly:
    run_help_test("neurolit.cli")

def test_lesion_to_segmentation_help():
    """Test lit-lesion-to-segmentation help."""
    run_help_test("neurolit.postprocessing.lesion_to_segmentation")

def test_lesion_to_surface_help():
    """Test lit-lesion-to-surface help."""
    run_help_test("neurolit.postprocessing.lesion_to_surface")
