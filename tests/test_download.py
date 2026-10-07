import hashlib
import importlib
import io

import pytest


@pytest.fixture
def download(monkeypatch):
    module = importlib.import_module('scripts.download_model')
    payload = b'verified model fixture'
    monkeypatch.setattr(module, 'EXPECTED_SIZE', len(payload))
    monkeypatch.setattr(module, 'EXPECTED_SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module.urllib.request, 'urlopen',
                        lambda request, timeout: io.BytesIO(payload))
    return module, payload


def test_verified_download_and_existing_file_skip(download, tmp_path, monkeypatch):
    module, payload = download
    target = tmp_path / 'assets' / 'model.safetensors'
    assert module.download_model(target) == target
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]
    monkeypatch.setattr(module.urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('Existing model must not download'))
    assert module.download_model(target) == target


@pytest.mark.parametrize('bad', [b'short', b'x' * len(b'verified model fixture'), b'x' * 100])
def test_bad_download_preserves_previous_file_and_removes_temp(download, tmp_path, monkeypatch, bad):
    module, _ = download
    target = tmp_path / 'model.safetensors'
    target.write_bytes(b'previous')
    monkeypatch.setattr(module.urllib.request, 'urlopen',
                        lambda request, timeout: io.BytesIO(bad))
    with pytest.raises(ValueError):
        module.download_model(target)
    assert target.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [target]


def test_network_failure_cleans_temp_and_uses_timeout(download, tmp_path, monkeypatch):
    module, _ = download

    def fail(request, timeout):
        assert 0 < timeout <= 30
        assert request.full_url == module.MODEL_URL
        raise TimeoutError('network timeout')

    monkeypatch.setattr(module.urllib.request, 'urlopen', fail)
    with pytest.raises(TimeoutError):
        module.download_model(tmp_path / 'model.safetensors')
    assert not list(tmp_path.iterdir())
