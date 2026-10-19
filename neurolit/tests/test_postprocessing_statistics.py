"""Regressions for CC coverage, label lookup, and old statistics filenames."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurolit.scripts import lesion_postprocessing as postprocessing


@pytest.mark.parametrize(("existing_name", "already_processed", "no_symlinks"), [
    ("aseg.VINN.withCC.stats", False, False), ("aseg.auto.stats", False, False),
    ("aseg.auto.mgz", False, False), ("aseg.auto.stats", True, False),
    ("aseg.auto.stats", True, True),
])
def test_legacy_stats_alias_migration_preserves_first_backups(tmp_path, monkeypatch, no_symlinks, existing_name, already_processed):
    subject = tmp_path / "subject"
    stats = subject / "stats"
    stats.mkdir(parents=True)
    old = stats / existing_name
    canonical = stats / "aseg.VINN.withCC.stats"
    old.write_text("original")
    if already_processed:
        postprocessing.ensure_backup(old)
        old.write_text("previous lesion-modified result")
    config = {"name": "CC", "segstatsfile": "stats/aseg.VINN.withCC.stats",
              "output_aliases": ["stats/aseg.auto.stats", "stats/aseg.auto.mgz"]}
    if no_symlinks:
        def reject_symlink(*args, **kwargs):
            raise OSError("symlinks unavailable")
        monkeypatch.setattr(Path, "symlink_to", reject_symlink)

    for content in ("first", "rerun"):
        def write_stats(*args, content=content, **kwargs):
            canonical.write_text(content)
            return True
        monkeypatch.setattr(postprocessing, "run_command", write_stats)
        result = postprocessing._run_segstats_step(tmp_path, tmp_path, "subject", config, None, "fastsurfer", sys.executable, set())
        assert result.status == "succeeded"
        assert result.outputs == ("stats/aseg.VINN.withCC.stats", "stats/aseg.auto.stats", "stats/aseg.auto.mgz")
        for name in ("aseg.VINN.withCC.stats", "aseg.auto.stats", "aseg.auto.mgz"):
            path = stats / name
            assert path.read_text() == content
            assert path.with_name(f"{path.stem}.lit{path.suffix}").read_text() == "original"


def test_mapping_uses_fastsurfer_lut_without_freesurfer(tmp_path, monkeypatch):
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    monkeypatch.delenv("FASTSURFER_HOME", raising=False)
    fastsurfer = tmp_path / "fastsurfer"
    lut = fastsurfer / "FastSurferCNN/config/FreeSurferColorLUT.txt"
    lut.parent.mkdir(parents=True)
    lut.write_text("49 Right-Thalamus 0 0 0 0\n")
    subject = tmp_path / "subject"
    (subject / "mri").mkdir(parents=True)
    (subject / "stats").mkdir()
    image = np.full((3, 3, 3), 49, dtype=np.int16)
    nib.save(nib.MGHImage(image, np.eye(4)), subject / "mri/aparc.DKTatlas+aseg.deep.mgz")
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), subject / "mri/mask.lit.nii.gz")
    output = "mri/aparc.DKTatlas+aseg.deep.mgz"
    report = "stats/aparc.DKTatlas+aseg.lesion_report.txt"
    assert postprocessing.map_lesion_to_segmentation(tmp_path, "subject", output, output, report, fastsurfer_path=fastsurfer)
    summary = json.loads(postprocessing.generate_lesion_impact_summary(tmp_path, "subject", [(output, report)], []).read_text())
    assert summary["affected_structures"] == ["Right-Thalamus"]
    assert summary["detailed_involvement"]["right_subcortical_affected"]
    freesurfer = tmp_path / "freesurfer"
    freesurfer.mkdir()
    (freesurfer / "FreeSurferColorLUT.txt").write_text("preferred LUT")
    assert postprocessing._find_lut_file(freesurfer, fastsurfer) == freesurfer / "FreeSurferColorLUT.txt"


@pytest.mark.parametrize(("label", "key"), [(49, "right_subcortical_affected"), (3, "left_cortical_affected")])
def test_bare_label_summary_still_identifies_hemisphere(tmp_path, label, key):
    stats = tmp_path / "subject/stats"
    stats.mkdir(parents=True)
    report = "stats/aparc.DKTatlas+aseg.lesion_report.txt"
    (tmp_path / "subject" / report).write_text(f"# Reduced Labels\n{label}\n")
    summary = json.loads(postprocessing.generate_lesion_impact_summary(tmp_path, "subject", [("seg", report)], []).read_text())
    assert summary["detailed_involvement"][key]


def test_optional_registration_measures_require_transform(tmp_path):
    config = {"segfile": "mri/seg.mgz", "measures": {"compute": ["BrainSeg"],
              "compute_if_exists": {"mri/transforms/talairach.xfm": ["EstimatedTotalIntraCranialVol"]}}}
    assert postprocessing.resolve_segstats_input_candidates(tmp_path, "subject", config) == config
    transform = tmp_path / "subject/mri/transforms/talairach.xfm"
    transform.parent.mkdir(parents=True)
    transform.write_text("transform")
    resolved = postprocessing.resolve_segstats_input_candidates(tmp_path, "subject", config)
    assert resolved["measures"]["compute"] == ["BrainSeg", "EstimatedTotalIntraCranialVol"]
    assert transform in postprocessing._collect_segstats_dependencies(tmp_path, "subject", resolved)
    assert config["measures"]["compute"] == ["BrainSeg"]


def test_cli_returns_failure_for_failed_statistics(tmp_path, monkeypatch):
    subject = tmp_path / "subject"
    (subject / "mri").mkdir(parents=True)
    nib.save(nib.MGHImage(np.ones((2, 2, 2), dtype=np.int16), np.eye(4)), subject / "mri/seg.mgz")
    configs = iter([{"segstats_calls": [{"name": "failed", "segfile": "mri/seg.mgz", "segstatsfile": "stats/failed.stats"}]}, {}])
    monkeypatch.setattr(sys, "argv", ["lit-postprocessing", "--subject-id", "subject", "--subjects-dir", str(tmp_path)])
    monkeypatch.setattr(postprocessing, "load_config", lambda *args: next(configs))
    monkeypatch.setattr(postprocessing, "check_required_files", lambda *args: None)
    monkeypatch.setattr(postprocessing, "validate_segstats_installation", lambda python_cmd: (tmp_path, False))
    monkeypatch.setattr(postprocessing, "run_segstats", lambda *args: False)
    monkeypatch.setattr(postprocessing, "generate_lesion_impact_summary", lambda *args: None)
    assert postprocessing.main() == 1
