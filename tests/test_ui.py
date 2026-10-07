from dataclasses import fields
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from handdraw.settings import INK_DEFAULTS, RIG_SETTINGS, Settings


def test_controls_match_native_metadata_and_defaults():
    app = importlib.import_module('app')
    demo = app.build_ui()
    try:
        assert set(demo.setting_controls) == {item.name for item in fields(Settings)}
        for item in fields(Settings):
            component = demo.setting_controls[item.name]
            assert component.label == item.metadata['label']
            if item.name not in ('rig_json', 'ink_json'):
                assert component.value == item.default
            if item.metadata['low'] is not None:
                assert component.minimum == item.metadata['low']
                assert component.maximum == item.metadata['high']
                assert component.step == item.metadata['step']
        assert demo.setting_controls['mode'].choices == [
            ('Detailed', 'd'), ('Quick', 'e')]
        rig = json.loads(demo.setting_controls['rig_json'].value)
        assert rig == {key: value for key, value in RIG_SETTINGS.items()
                       if key not in app.PROTECTED_RIG_KEYS}
        assert json.loads(demo.setting_controls['ink_json'].value) == INK_DEFAULTS
        Settings(rig_json=json.dumps(rig), ink_json=json.dumps(INK_DEFAULTS)).validate()
        assert demo.analytics_enabled is False
        assert demo._queue.max_size == 8
        assert demo._queue.default_concurrency_limit == 1
    finally:
        demo.close()


def test_render_adapter_returns_paths_and_forwards_progress(monkeypatch, tmp_path):
    app = importlib.import_module('app')
    monkeypatch.setattr(app, 'OUTPUT_DIR', tmp_path)
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    output = tmp_path / 'video.mp4'
    output.write_bytes(b'test')
    updates = []

    def render(image_numpy, settings, output_dir=None, progress=None):
        assert image_numpy is image
        assert settings == Settings()
        assert Path(output_dir) == tmp_path
        progress(.5, 'Rendering')
        return output, {'frames': 1}

    monkeypatch.setitem(sys.modules, 'handdraw.render', SimpleNamespace(render_video=render))
    result = app.render_image(image, {item.name: item.default for item in fields(Settings)},
                              lambda fraction, desc: updates.append((fraction, desc)))
    assert result == (str(output), str(output), {'frames': 1})
    assert updates == [(.5, 'Rendering')]


def test_invalid_settings_fail_before_renderer_import():
    app = importlib.import_module('app')
    with pytest.raises(app.gr.Error):
        app.render_image(np.zeros((8, 8, 4), dtype=np.uint8), {'duration': -1})
    with pytest.raises(app.gr.Error):
        app.render_image(None, {})


def test_render_adapter_rejects_output_outside_output_directory(monkeypatch, tmp_path):
    app = importlib.import_module('app')
    monkeypatch.setattr(app, 'OUTPUT_DIR', tmp_path / 'outputs')
    output = tmp_path / 'outside.mp4'
    output.write_bytes(b'test')
    monkeypatch.setitem(sys.modules, 'handdraw.render', SimpleNamespace(
        render_video=lambda *args, **kwargs: (output, {})))
    with pytest.raises(app.gr.Error, match='invalid output file'):
        app.render_image(np.zeros((8, 8, 4), dtype=np.uint8), {})
