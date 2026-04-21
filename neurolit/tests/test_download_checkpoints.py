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


def test_download_checkpoint_cleans_partial_file_on_failure(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"

    monkeypatch.setattr(dc.requests, "get", lambda *args, **kwargs: BrokenStreamResponse())

    with pytest.raises(requests.exceptions.RequestException, match="Failed downloading the checkpoint model.pt"):
        dc.download_checkpoint("model.pt", model_path, ["https://example.test/model.pt"], show_progress=False)

    assert not model_path.exists()
    assert not Path(f"{model_path}.part").exists()


def test_fallback_multiple_urls_raises_last_error(monkeypatch, tmp_path):
    errors = [requests.exceptions.Timeout("first failure"), PermissionError("second failure")]

    def fake_check_and_download(*args, **kwargs):
        raise errors.pop(0)

    monkeypatch.setattr(dc, "check_and_download_ckpts", fake_check_and_download)

    with pytest.raises(PermissionError, match="second failure"):
        dc.fallback_multiple_urls(
            str(tmp_path / "model.pt"),
            ["https://example.test/one", "https://example.test/two"],
            show_progress=False,
        )


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
