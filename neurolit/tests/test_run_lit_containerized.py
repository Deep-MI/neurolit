"""Tests for the ``run_lit_containerized.sh`` wrapper using stub container tools."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Resolve from the source tree: the wrapper is not shipped in the wheel, and CI runs
# these tests against a non-editable install.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_lit_containerized.sh"

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="requires bash"),
    pytest.mark.skipif(not SCRIPT.is_file(), reason="run_lit_containerized.sh not found in source tree"),
]

STUB = """#!/bin/sh
echo "tool=$(basename "$0")"
echo "cuda_visible=${SINGULARITYENV_CUDA_VISIBLE_DEVICES-unset}"
for arg in "$@"; do
  echo "arg=$arg"
done
"""


@pytest.fixture
def wrapper(tmp_path):
    """Return a runner for the wrapper with stub ``docker`` and ``singularity`` on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "singularity"):
        stub = bin_dir / tool
        stub.write_text(STUB)
        stub.chmod(0o755)
    for name in ("t1.nii.gz", "mask.nii.gz", "image.sif"):
        (tmp_path / name).touch()

    env = {key: value for key, value in os.environ.items() if not key.endswith("ENV_CUDA_VISIBLE_DEVICES")}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    def run(*args):
        cmd = [
            "bash",
            str(SCRIPT),
            "--tag",
            "test",
            "-i",
            str(tmp_path / "t1.nii.gz"),
            "-m",
            str(tmp_path / "mask.nii.gz"),
            "-o",
            str(tmp_path / "out"),
            *args,
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout + result.stderr

    run.sif = str(tmp_path / "image.sif")
    return run


@pytest.mark.parametrize("device", ["gpu", "cuda:x"])
def test_wrapper_rejects_invalid_device(wrapper, device):
    """Malformed device strings are rejected before any container tool runs."""
    code, out = wrapper("--device", device)
    assert code != 0
    assert "--device must be one of" in out
    assert "tool=" not in out


def test_wrapper_rejects_mps_with_hint(wrapper):
    """MPS is not offered inside containers and the error says why."""
    code, out = wrapper("--device", "MPS")
    assert code != 0
    assert "not available inside containers" in out
    assert "tool=" not in out


def test_wrapper_lowercases_device(wrapper):
    """Device names are case-insensitive, matching lit-inpainting."""
    code, out = wrapper("--device", "CUDA:1", "--gpus", "1,2")
    assert code == 0, out
    lines = out.splitlines()
    assert "tool=docker" in lines
    assert "arg=device=1,2" in lines
    assert lines[lines.index("arg=--device") + 1] == "arg=cuda:1"
