import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from neurolit.scripts.lesion_postprocessing import (
    _run_segstats_step,
    build_segstats_command,
    build_stats_overview,
    ensure_backup,
    resolve_segstats_input_candidates,
    setup_argparse,
    validate_segstats_installation,
)


def _write_image(path: Path, shape: tuple[int, int, int], affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), path)


def test_ensure_backup_supports_python_310(tmp_path):
    """Backup creation must only use filesystem APIs available on Python 3.10."""
    source = tmp_path / "segmentation.mgz"
    source.write_bytes(b"segmentation")

    backup = ensure_backup(source)

    assert backup == tmp_path / "segmentation.lit.mgz"
    assert backup.read_bytes() == source.read_bytes()


def test_ensure_backup_preserves_broken_symlink(tmp_path):
    """The compatibility existence check must retain no-follow symlink behavior."""
    source = tmp_path / "segmentation.mgz"
    try:
        source.symlink_to("missing.mgz")
    except OSError as e:
        pytest.skip(f"Symbolic links are unavailable: {e}")

    backup = ensure_backup(source)

    assert backup is not None
    assert backup.is_symlink()
    assert os.readlink(backup) == "missing.lit.mgz"


def test_postprocessing_uses_current_python_interpreter_by_default(tmp_path):
    """FastSurfer subprocesses must inherit the environment running Neurolit."""
    args = setup_argparse().parse_args(["--subject-id", "subject", "--subjects-dir", str(tmp_path)])
    command = build_segstats_command(
        tmp_path / "fastsurfer",
        tmp_path,
        "subject",
        {"segfile": "mri/segmentation.mgz"},
    )

    assert args.python_cmd == sys.executable
    assert command[0] == sys.executable


def test_fastsurfer_probe_uses_selected_python(tmp_path, monkeypatch):
    """A standalone Python 3.10 process may delegate newer FastSurfer to Python 3.14."""
    script = tmp_path / "FastSurferCNN/segstats.py"
    script.parent.mkdir()
    script.touch()
    monkeypatch.setenv("FASTSURFER_HOME", str(tmp_path))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("neurolit.scripts.lesion_postprocessing.subprocess.run", run)

    assert validate_segstats_installation("fastsurfer-python") == (tmp_path, False)
    assert commands == [["fastsurfer-python", str(script), "--help"]]


def test_resolve_segstats_candidates_prefers_matching_conformed_image(tmp_path):
    """A submillimeter subject should use FastSurfer's matching 1 mm image."""
    subject_id = "subject"
    mri_dir = tmp_path / subject_id / "mri"
    _write_image(mri_dir / "cerebellum.CerebNet.nii.gz", (4, 4, 4), np.eye(4))
    _write_image(mri_dir / "orig_nu.10mm.mgz", (4, 4, 4), np.eye(4))
    _write_image(mri_dir / "orig_nu.mgz", (5, 5, 5), np.diag([0.8, 0.8, 0.8, 1.0]))
    config = {
        "segfile": "mri/cerebellum.CerebNet.nii.gz",
        "normfile": "mri/orig_nu.mgz",
        "pvfile": "mri/orig_nu.mgz",
        "normfile_candidates": ["mri/orig_nu.10mm.mgz", "mri/orig_nu.mgz"],
        "pvfile_candidates": ["mri/orig_nu.10mm.mgz", "mri/orig_nu.mgz"],
    }

    resolved = resolve_segstats_input_candidates(tmp_path, subject_id, config)

    assert resolved["normfile"] == "mri/orig_nu.10mm.mgz"
    assert resolved["pvfile"] == "mri/orig_nu.10mm.mgz"
    assert config["normfile"] == "mri/orig_nu.mgz"


def test_resolve_segstats_candidates_keeps_matching_native_image(tmp_path):
    """An already-conformed 1 mm subject should continue using orig_nu.mgz."""
    subject_id = "subject"
    mri_dir = tmp_path / subject_id / "mri"
    _write_image(mri_dir / "cerebellum.CerebNet.nii.gz", (4, 4, 4), np.eye(4))
    _write_image(mri_dir / "orig_nu.mgz", (4, 4, 4), np.eye(4))
    config_path = Path(__file__).parents[1] / "postprocessing" / "segstats_config.json"
    config = next(item for item in json.loads(config_path.read_text())["segstats_calls"] if item["name"] == "cerebellum.CerebNet")

    resolved = resolve_segstats_input_candidates(tmp_path, subject_id, config)

    assert resolved["normfile"] == "mri/orig_nu.mgz"
    assert resolved["pvfile"] == "mri/orig_nu.mgz"


def test_stats_overview_reports_resolved_geometry_inputs(tmp_path, monkeypatch):
    """The output overview should report the files actually passed to segstats."""
    subject_id = "subject"
    subject_dir = tmp_path / subject_id
    mri_dir = subject_dir / "mri"
    stats_dir = subject_dir / "stats"
    _write_image(mri_dir / "cerebellum.CerebNet.nii.gz", (4, 4, 4), np.eye(4))
    _write_image(mri_dir / "orig_nu.10mm.mgz", (4, 4, 4), np.eye(4))
    _write_image(mri_dir / "orig_nu.mgz", (5, 5, 5), np.diag([0.8, 0.8, 0.8, 1.0]))
    stats_dir.mkdir()
    (stats_dir / "cerebellum.CerebNet.stats").write_text("stats")
    config = {
        "segstats_calls": [{
            "segfile": "mri/cerebellum.CerebNet.nii.gz",
            "segstatsfile": "stats/cerebellum.CerebNet.stats",
            "normfile": "mri/orig_nu.mgz",
            "pvfile": "mri/orig_nu.mgz",
            "normfile_candidates": ["mri/orig_nu.10mm.mgz", "mri/orig_nu.mgz"],
            "pvfile_candidates": ["mri/orig_nu.10mm.mgz", "mri/orig_nu.mgz"],
        }]
    }

    monkeypatch.setattr("neurolit.scripts.lesion_postprocessing.run_segstats", lambda *args: True)
    result = _run_segstats_step(tmp_path / "fastsurfer", tmp_path, subject_id,
                               config["segstats_calls"][0] | {"name": "CerebNet"}, None, "fastsurfer", sys.executable, set())
    overview = build_stats_overview([result])

    assert overview == [
        "mri/cerebellum.CerebNet.nii.gz + mri/orig_nu.10mm.mgz => stats/cerebellum.CerebNet.stats"
    ]
