import importlib
from dataclasses import replace

import numpy as np
import pytest


def modules():
    return importlib.import_module('handdraw.settings'), importlib.import_module('handdraw.render')


def image():
    rgb = np.full((64, 96, 3), 245, np.uint8)
    rgb[8:55, 15:80] = (50, 130, 180)
    rgb[20:44, 32:61] = (180, 50, 40)
    rgb[30:33, 8:88] = 0
    return rgb


def test_two_ratios_drive_separate_phase_durations():
    config, _ = modules()
    settings = config.Settings(duration=10, draw_percent=70, ink_percent=45)
    assert settings.phase_seconds == pytest.approx((3.15, 3.85, 3))
    assert replace(settings, ink_percent=80).phase_seconds == pytest.approx((5.6, 1.4, 3))


@pytest.mark.parametrize('changes', [dict(duration=float('nan')), dict(duration=-1),
    dict(ink_percent=100), dict(draw_percent=0), dict(mode='c'), dict(width=10000),
    dict(fps=0), dict(device='auto'), dict(adaptive_block=4), dict(palette_colors=0)])
def test_invalid_settings_fail_before_render(changes):
    config, _ = modules()
    with pytest.raises(ValueError):
        replace(config.Settings(), **changes).validate()


def test_keep_aspect_has_no_crop_or_white_border():
    config, core = modules()
    rgb = image()
    fitted = core.fit_image(rgb, config.Settings(width=96, height=96, fit='original'))
    np.testing.assert_array_equal(fitted, rgb)


@pytest.mark.parametrize('mode', ('d', 'e'))
def test_plan_has_monotone_path_subject_priority_and_complete_color(mode):
    config, core = modules()
    subject = np.zeros((64, 96), bool)
    subject[16:50, 26:72] = True
    plan = core.build_plan(image(), config.Settings(mode=mode, device='cpu'), subject_mask=subject)
    assert np.all(np.diff(plan['pen_path'][:, 0]) > 0)
    assert np.isfinite(plan['color_contact_time']).all()
    assert plan['color_contact_time'][subject].max() <= plan['color_contact_time'][~subject].min()
    assert plan['color_contact_time'].max() <= 1


def test_rig_settings_are_per_render_and_do_not_mutate_defaults():
    config, _ = modules()
    from handdraw.v181d_hand import RIG_SETTINGS
    before = config.rig_settings(config.Settings())
    changed = config.rig_settings(config.Settings(wrist_degrees=20, finger_motion=0, shadow_strength=0))
    assert changed['wrist_degrees'] == 20
    assert all(center[-2:] == [0, 0] for pose in changed['poses'].values() for center in pose['centers'])
    assert changed['shadow_max'] == 0
    assert RIG_SETTINGS == before


def test_cache_does_not_confuse_different_shapes_with_identical_bytes():
    config, core = modules()
    settings = config.Settings(device='cpu', encoder='cpu', subject_first=False)
    first = np.full((32, 64, 3), 175, np.uint8)
    second = np.full((64, 32, 3), 175, np.uint8)
    assert first.tobytes() == second.tobytes()
    a, _ = core._cached_plan(first, settings)
    b, reused = core._cached_plan(second, settings)
    assert not reused
    assert a['structure_time'].shape == (32, 64)
    assert b['structure_time'].shape == (64, 32)


def test_paper_requires_opaque_rgb_color():
    config, _ = modules()
    with pytest.raises(ValueError):
        config.Settings(paper_color='#ffffffff').validate()


def test_every_public_setting_has_a_ui_control():
    from dataclasses import fields
    config, _ = modules()
    app = importlib.import_module('app')
    demo = app.build_ui()
    assert set(demo.setting_controls) == {field.name for field in fields(config.Settings)}
    assert demo.setting_controls['mode'].choices == [('Detailed (.d)', 'd'), ('Quick (.e)', 'e')]


def test_cpu_export_is_a_real_mp4(tmp_path):
    import cv2
    config, core = modules()
    settings = config.Settings(width=96, height=64, duration=.5, fps=20, mode='e',
                               device='cpu', encoder='cpu', show_hand=False, subject_first=False)
    output, report = core.render_video(image(), settings, output_dir=tmp_path)
    assert output.is_file() and output.suffix == '.mp4'
    capture = cv2.VideoCapture(str(output))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    assert len(frames) == 10
    assert np.abs(frames[-1].astype(float) - image()).mean() < 6
    assert report['settings']['mode'] == 'e' and report['frames'] == 10
    assert report['device'] == 'cpu'
