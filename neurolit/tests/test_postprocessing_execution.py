"""Focused regressions for statistics scheduling, outcomes, and first backups."""

import sys
from pathlib import PureWindowsPath

import nibabel as nib
import numpy as np
import pytest

from neurolit.postprocessing.backups import ensure_backup
from neurolit.postprocessing.statistics import segstats_requires_surfaces, select_segstats_backend
from neurolit.scripts import lesion_postprocessing as postprocessing


@pytest.mark.parametrize("link_kind", ["regular_file", "same_directory", "relative_directory", "absolute_directory"])
def test_backups_and_reruns_preserve_first_snapshot(tmp_path, link_kind):
    """A live target may change on reruns, but its first snapshot must not."""
    target_dir = tmp_path if link_kind == "same_directory" else tmp_path / "originals"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / "segmentation.nii.gz"
    target.write_bytes(b"original")
    source = target
    if link_kind != "regular_file":
        source = tmp_path / "alias.nii.gz"
        link_target = target if link_kind == "absolute_directory" else target.relative_to(tmp_path)
        try:
            source.symlink_to(link_target)
        except OSError as error:
            pytest.skip(f"Symbolic links are unavailable: {error}")

    backup = ensure_backup(source)
    assert backup.is_symlink() == (link_kind != "regular_file")
    assert backup.resolve() == target.with_name("segmentation.lit.nii.gz")
    assert backup.read_bytes() == b"original"

    target.write_bytes(b"mapped")
    assert ensure_backup(source) == backup
    assert backup.read_bytes() == b"original"
    assert ensure_backup(target).read_bytes() == b"original"


@pytest.mark.parametrize(("config", "default", "backend", "surfaces"), [
    ({"segfile": "mri/aseg.mgz"}, "fastsurfer", "fastsurfer", False),
    ({"segfile": "mri/aseg.mgz"}, "freesurfer", "freesurfer", False),
    ({"segfile": "mri/aseg.mgz", "use_mri_segstats": True}, "fastsurfer", "freesurfer", False),
    ({"in_file": "mri/norm.mgz"}, "fastsurfer", "freesurfer", False),
    ({"in_file": "surf/lh.thickness"}, "fastsurfer", "freesurfer", True),
    ({"annot": ["lh", "aparc"]}, "fastsurfer", "freesurfer", True),
])
def test_backend_selection_is_independent_of_surface_scheduling(config, default, backend, surfaces):
    assert select_segstats_backend(config, default) == backend
    assert segstats_requires_surfaces(config) == surfaces


def test_freesurfer_command_uses_native_flags(tmp_path, monkeypatch):
    """The supported volume fallback must not receive FastSurfer-only flags."""
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    config = {"segfile": "mri/aseg.mgz", "segstatsfile": "stats/aseg.stats", "normfile": "mri/norm.mgz",
              "pvfile": "mri/norm.mgz", "lut": "config/FreeSurferColorLUT.txt", "ids": [2, 3], "empty": True}
    command = postprocessing.build_segstats_command(None, tmp_path, "subject", config, tmp_path / "freesurfer", use_mri_segstats=True)
    subject = tmp_path / "subject"
    assert command[0] == "mri_segstats"
    expected = {"--sd": tmp_path, "--seg": subject / "mri/aseg.mgz", "--sum": subject / "stats/aseg.stats",
                "--in": subject / "mri/norm.mgz", "--pv": subject / "mri/norm.mgz",
                "--ctab": tmp_path / "freesurfer/FreeSurferColorLUT.txt"}
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == str(value)
    assert command[command.index("--id") + 1:command.index("--id") + 3] == ["2", "3"]
    assert "--empty" in command
    assert not {"--sid", "--threads", "--segfile", "--segstatsfile", "--normfile", "--pvfile", "--lut", "--ids"}.intersection(command)
    with pytest.raises(ValueError, match="requires FastSurfer"):
        postprocessing.build_segstats_command(None, tmp_path, "subject", config | {"measures": {"compute": ["BrainSeg"]}},
                                             use_mri_segstats=True)


def test_subject_relative_paths_use_config_notation():
    """Windows separators must not prevent matching against config paths such as stats/aseg.stats."""
    subject = PureWindowsPath("C:/subjects/subject")
    assert postprocessing._subject_relative_path(subject / "stats" / "aseg.stats", subject) == "stats/aseg.stats"


def test_failed_statistics_block_stale_downstream_files(tmp_path, monkeypatch):
    """Failed producers cannot be replaced by stats left over from a prior run."""
    subject = tmp_path / "subject"
    (subject / "mri").mkdir(parents=True)
    (subject / "stats").mkdir()
    nib.save(nib.MGHImage(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)), subject / "mri/aseg.mgz")
    (subject / "stats/producer.stats").write_text("old producer")
    (subject / "stats/consumer.stats").write_text("old consumer")
    calls = []

    def fail(*args):
        calls.append(args)
        return False

    monkeypatch.setattr(postprocessing, "run_segstats", fail)
    unavailable = set()
    producer = {"name": "producer", "segfile": "mri/aseg.mgz", "segstatsfile": "stats/producer.stats"}
    consumer = {"name": "consumer", "segfile": "mri/aseg.mgz", "segstatsfile": "stats/consumer.stats",
                "measures": {"file": "stats/producer.stats", "import_all": True}}
    results = [postprocessing._run_segstats_step(tmp_path, tmp_path, "subject", config,
                                               None, "fastsurfer", sys.executable, unavailable)
               for config in [producer, consumer]]

    assert [result.status for result in results] == ["failed", "skipped"]
    assert not any(results)
    assert "stats/producer.stats" in results[1].reason
    assert unavailable == {"stats/producer.stats", "stats/consumer.stats"}
    assert len(calls) == 1
    assert postprocessing.build_stats_overview(results) == []


@pytest.mark.parametrize("skip_flag", ["--skip-surface-masking", "--skip-segstats"])
def test_cli_skip_flags_and_freesurfer_volume_scheduling(tmp_path, monkeypatch, skip_flag):
    """FreeSurfer volume calls still run without surfaces; skip-stats skips all tools."""
    subject = tmp_path / "subject"
    (subject / "mri").mkdir(parents=True)
    (subject / "stats").mkdir()
    nib.save(nib.MGHImage(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)), subject / "mri/aseg.mgz")
    volume = {"name": "volume", "segfile": "mri/aseg.mgz", "segstatsfile": "stats/volume.stats"}
    surface = {"name": "surface", "annot": ["lh", "aparc"], "sum": "stats/surface.stats"}
    configs = iter([{"segstats_calls": [volume, surface]},
                    {"surfstats_calls": [{"name": "anatomical", "statsfile": "stats/{hemi}.anatomical.stats"}]}])
    monkeypatch.setattr(sys, "argv", ["lit-postprocessing", "--subject-id", "subject", "--subjects-dir", str(tmp_path), skip_flag])
    monkeypatch.setattr(postprocessing, "load_config", lambda path: next(configs))
    monkeypatch.setattr(postprocessing, "check_required_files", lambda *args: None)
    monkeypatch.setattr(postprocessing, "validate_segstats_installation", lambda python_cmd: (None, True))
    monkeypatch.setattr(postprocessing, "generate_lesion_impact_summary", lambda *args: None)
    calls = []
    results = []

    def run(*args):
        calls.append(args)
        (subject / args[3]["segstatsfile"]).write_text("new stats")
        return True

    monkeypatch.setattr(postprocessing, "run_segstats", run)
    monkeypatch.setattr(postprocessing, "build_stats_overview", lambda items: results.extend(items) or [])
    postprocessing.main()

    by_name = {result.name: result for result in results}
    assert set(by_name) == {"volume", "surface", "lh.anatomical", "rh.anatomical"}
    assert all(result.status == "skipped" for name, result in by_name.items() if name != "volume")
    if skip_flag == "--skip-surface-masking":
        assert len(calls) == 1
        assert calls[0][5] is True
        assert by_name["volume"].status == "succeeded"
        assert by_name["volume"].backend == "freesurfer"
    else:
        assert calls == []
        assert all(result.status == "skipped" for result in results)
