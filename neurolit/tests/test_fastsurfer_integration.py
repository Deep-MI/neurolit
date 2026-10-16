"""Optional integration test against an actual FastSurfer checkout."""

import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest


def test_fastsurfer_segonly_postprocessing(tmp_path):
    fastsurfer = os.environ.get("NEUROLIT_TEST_FASTSURFER")
    if not fastsurfer:
        pytest.skip("Set NEUROLIT_TEST_FASTSURFER to a FastSurfer checkout")
    subject = tmp_path / "subject"
    mri = subject / "mri"
    mri.mkdir(parents=True)
    original = np.zeros((16, 16, 16), dtype=np.int16)
    original[2:14, 2:14, 2:14] = 41
    original[4:10, 4:10, 4:10] = 49
    lesion = np.zeros(original.shape, dtype=np.uint8)
    lesion[6:8, 6:8, 6:8] = 1
    images = {
        "aparc.DKTatlas+aseg.deep.mgz": original,
        "aseg.auto_noCCseg.mgz": original,
        "aparc.DKTatlas+aseg.deep.withCC.mgz": original.copy(),
        "aseg.auto.mgz": original.copy(),
        "cerebellum.CerebNet.nii.gz": np.where(original > 0, 601, 0).astype(np.int16),
        "hypothalamus.HypVINN.nii.gz": np.where(original > 0, 961, 0).astype(np.int16),
        "orig_nu.mgz": np.random.default_rng(42).uniform(50, 150, original.shape).astype(np.float32),
        "mask.mgz": (original > 0).astype(np.uint8),
        "mask.lit.nii.gz": lesion,
    }
    for name in ("aparc.DKTatlas+aseg.deep.withCC.mgz", "aseg.auto.mgz"):
        images[name][2:4, 2:4, 2:4] = 251
    for name, data in images.items():
        image_type = nib.MGHImage if name.endswith(".mgz") else nib.Nifti1Image
        nib.save(image_type(data, np.eye(4)), mri / name)

    env = {**os.environ, "FASTSURFER_HOME": fastsurfer}
    env.pop("FREESURFER_HOME", None)
    tables = {}
    names = ("aseg+DKT.VINN", "aseg.VINN", "aseg+DKT.VINN.withCC", "aseg.VINN.withCC",
             "cerebellum.CerebNet", "hypothalamus.HypVINN")
    for iteration in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "neurolit.scripts.lesion_postprocessing", "--subject-id", "subject",
             "--subjects-dir", str(tmp_path), "--skip-surface-masking"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((subject / "stats/lesion_impact_summary.json").read_text())
        assert summary["affected_structures"] == ["Right-Thalamus"]
        assert summary["hemisphere_impact"]["right_hemisphere_affected"]
        for name in names:
            path = subject / "stats" / f"{name}.stats"
            rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
            assert rows
            numeric = np.array([[*row[:4], *row[5:]] for row in rows], dtype=float)
            assert np.isfinite(numeric).all()
            if iteration:
                assert rows == tables[name]
            tables[name] = rows
        for name in names[:4]:
            rows = {int(row[1]): row for row in tables[name]}
            assert int(rows[99][2]) == 8
            assert int(rows[49][2]) == 6 ** 3 - 8
        for name in images:
            if name.startswith(("aparc.", "aseg.")):
                backup = mri / name.replace(".mgz", ".lit.mgz")
                np.testing.assert_array_equal(nib.load(backup).get_fdata(), images[name])
        canonical = subject / "stats/aseg.VINN.withCC.stats"
        for name in ("aseg.auto.stats", "aseg.auto.mgz"):
            assert (subject / "stats" / name).read_bytes() == canonical.read_bytes()
