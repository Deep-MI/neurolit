import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from neurolit.utils import download_checkpoints as dc


class BrokenStreamResponse:
    """Fake streaming response that fails after writing partial data."""

    headers = {"content-length": "2"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"x"
        raise OSError("simulated write failure")


class MissingResponse:
    """Fake response that raises HTTPError before streaming starts."""

    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        raise requests.exceptions.HTTPError("404 not found")

    def iter_content(self, chunk_size):
        yield from ()


class TruncatedResponse:
    """Fake response that terminates cleanly before content-length is satisfied."""

    headers = {"content-length": "2"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"x"


def test_model_urls_prefer_b2share_then_zenodo():
    """Checkpoint downloads should try B2SHARE before the Zenodo mirror."""
    for model_name, urls in dc.MODEL_URLS.items():
        assert urls[0] == f"{dc.B2SHARE_FILES_URL}/{model_name}"
        assert urls[1] == f"{dc.ZENODO_RECORD_URL}/{model_name}?download=1"


def test_download_checkpoint_cleans_partial_file_on_failure(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"

    monkeypatch.setattr(dc.requests, "get", lambda *args, **kwargs: BrokenStreamResponse())

    with pytest.raises(OSError, match="simulated write failure"):
        dc.download_checkpoint("model.pt", model_path, ["https://example.test/model.pt"], show_progress=False)

    assert not model_path.exists()
    assert not list(tmp_path.glob("model.pt.*.part"))


def test_download_checkpoint_preserves_http_error(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"

    monkeypatch.setattr(dc.requests, "get", lambda *args, **kwargs: MissingResponse())

    with pytest.raises(requests.exceptions.HTTPError, match="404 not found"):
        dc.download_checkpoint("model.pt", model_path, ["https://example.test/model.pt"], show_progress=False)


def test_download_checkpoint_rejects_truncated_response(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"

    monkeypatch.setattr(dc.requests, "get", lambda *args, **kwargs: TruncatedResponse())

    with pytest.raises(requests.exceptions.RequestException, match="Incomplete download for model.pt"):
        dc.download_checkpoint("model.pt", model_path, ["https://example.test/model.pt"], show_progress=False)

    assert not model_path.exists()
    assert not list(tmp_path.glob("model.pt.*.part"))


def test_fallback_multiple_urls_raises_after_all_sources_fail(monkeypatch, tmp_path):
    """The fallback helper should fail loudly when no mirror produced a file."""
    checkpoint_path = tmp_path / "model_axial.pt"
    errors = [requests.exceptions.Timeout("first failure"), PermissionError("second failure")]

    def fail_download(*args, **kwargs):
        raise errors.pop(0)

    monkeypatch.setattr(dc, "check_and_download_ckpts", fail_download)

    with pytest.raises(requests.exceptions.RequestException, match="Failed downloading"):
        dc.fallback_multiple_urls(str(checkpoint_path), ["https://primary", "https://backup"], show_progress=False)


def test_fallback_serializes_concurrent_downloads_of_same_checkpoint(monkeypatch, tmp_path):
    """Concurrent callers should perform only one physical download."""
    checkpoint_path = tmp_path / "model_axial.pt"
    download_started = threading.Event()
    release_download = threading.Event()
    second_caller_started = threading.Event()
    calls = []

    def fake_download(checkpoint_name, target, urls, verbose=False, show_progress=True, position=None):
        calls.append((checkpoint_name, Path(target), tuple(urls)))
        download_started.set()
        assert release_download.wait(timeout=5)
        Path(target).write_bytes(b"checkpoint")

    def download(second=False):
        if second:
            second_caller_started.set()
        dc.fallback_multiple_urls(checkpoint_path, ["https://primary"], show_progress=False)

    monkeypatch.setattr(dc, "download_checkpoint", fake_download)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(download)
        assert download_started.wait(timeout=5)
        second = executor.submit(download, True)
        assert second_caller_started.wait(timeout=5)
        release_download.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert calls == [("model_axial.pt", checkpoint_path, ("https://primary",))]
    assert checkpoint_path.read_bytes() == b"checkpoint"


def test_main_downloads_missing_models_in_parallel(monkeypatch, tmp_path):
    """The CLI should dispatch all missing model downloads with distinct progress positions."""
    calls = []

    monkeypatch.setattr(dc, "user_data_dir", lambda *args, **kwargs: str(tmp_path))

    def fake_download(checkpoint_name, urls, verbose=False, show_progress=True, position=None):
        calls.append((Path(checkpoint_name).name, tuple(urls), position))
        Path(checkpoint_name).write_bytes(b"stub")

    monkeypatch.setattr(dc, "fallback_multiple_urls", fake_download)

    dc.main(argv=[])

    assert sorted(name for name, _, _ in calls) == sorted(dc.MODEL_URLS)
    assert {position for _, _, position in calls} == {0, 1, 2}
    for model_name in dc.MODEL_URLS:
        assert (tmp_path / "weights" / model_name).exists()


def test_main_exits_if_models_are_still_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dc, "user_data_dir", lambda *args, **kwargs: str(tmp_path / "data"))
    monkeypatch.setattr(dc, "fallback_multiple_urls", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        dc.main([])

    assert exc.value.code == 1

    stdout = capsys.readouterr().out
    assert "Missing model files after download" in stdout
    assert "model_coronal.pt" in stdout
    assert "model_axial.pt" in stdout
    assert "model_sagittal.pt" in stdout
