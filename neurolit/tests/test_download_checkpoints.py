from pathlib import Path

import pytest
import requests

from neurolit.utils import download_checkpoints


def test_model_urls_prefer_b2share_then_zenodo():
    """Checkpoint downloads should try B2SHARE before the Zenodo mirror."""
    for model_name, urls in download_checkpoints.MODEL_URLS.items():
        assert urls[0] == f"{download_checkpoints.B2SHARE_FILES_URL}/{model_name}"
        assert urls[1] == f"{download_checkpoints.ZENODO_RECORD_URL}/{model_name}?download=1"


def test_fallback_multiple_urls_raises_after_all_sources_fail(monkeypatch, tmp_path):
    """The fallback helper should fail loudly when no mirror produced a file."""
    checkpoint_path = tmp_path / "model_axial.pt"

    def fail_download(*args, **kwargs):
        raise requests.exceptions.RequestException("offline")

    monkeypatch.setattr(download_checkpoints, "check_and_download_ckpts", fail_download)

    with pytest.raises(requests.exceptions.RequestException, match="Failed downloading"):
        download_checkpoints.fallback_multiple_urls(str(checkpoint_path), ["https://primary", "https://backup"])


def test_main_downloads_missing_models_in_parallel(monkeypatch, tmp_path):
    """The CLI should dispatch all missing model downloads with distinct progress positions."""
    calls = []

    monkeypatch.setattr(download_checkpoints, "user_data_dir", lambda *args, **kwargs: str(tmp_path))

    def fake_download(checkpoint_name, urls, verbose=False, show_progress=True, position=None):
        calls.append((Path(checkpoint_name).name, tuple(urls), position))
        Path(checkpoint_name).write_bytes(b"stub")

    monkeypatch.setattr(download_checkpoints, "fallback_multiple_urls", fake_download)

    download_checkpoints.main(argv=[])

    assert sorted(name for name, _, _ in calls) == sorted(download_checkpoints.MODEL_URLS)
    assert {position for _, _, position in calls} == {0, 1, 2}
    for model_name in download_checkpoints.MODEL_URLS:
        assert (tmp_path / "weights" / model_name).exists()
