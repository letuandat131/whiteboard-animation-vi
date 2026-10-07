from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
import math

from PIL import ImageColor

from .v181d_hand import RIG_SETTINGS


def control(default, label, group, low=None, high=None, step=1, choices=None):
    return field(default=default, metadata=dict(label=label, group=group, low=low,
                                               high=high, step=step, choices=choices))


@dataclass(frozen=True)
class Settings:
    mode: str = control('d', 'Mode', 'Video', choices=[('Detailed (.d)', 'd'), ('Quick (.e)', 'e')])
    duration: float = control(10., 'Duration (seconds)', 'Video', .1, 120, .1)
    draw_percent: float = control(70., 'Drawing & coloring (%)', 'Video', 5, 100, 1)
    ink_percent: float = control(45., 'Ink within drawing phase (%)', 'Video', 5, 95, 1)
    width: int = control(1920, 'Maximum width (px)', 'Video', 64, 3840, 2)
    height: int = control(1080, 'Maximum height (px)', 'Video', 64, 2160, 2)
    fit: str = control('original', 'Image fit', 'Video', choices=[('Keep original aspect ratio', 'original'), ('Fill frame, crop center', 'cover'), ('Fit entire image, add padding', 'contain')])
    fps: int = control(30, 'FPS', 'Video', 1, 60)
    paper_color: str = control('#ffffff', 'Paper color', 'Video')
    device: str = control('cuda', 'Rendering device', 'Video export', choices=[('GPU (CUDA)', 'cuda'), ('CPU', 'cpu')])
    encoder: str = control('nvenc', 'Encoder', 'Video export', choices=[('NVIDIA NVENC', 'nvenc'), ('CPU H.264', 'cpu')])
    quality: int = control(30, 'CQ / CRF', 'Video export', 1, 51)
    encoder_preset: str = control('p4', 'NVENC preset', 'Video export', choices=[f'p{i}' for i in range(1, 8)])
    cpu_preset: str = control('fast', 'CPU H.264 preset', 'Video export', choices=['ultrafast', 'fast', 'medium', 'slow'])
    render_batch_size: int = control(2, 'Render batch size', 'Video export', 1, 8)
    seed: int = control(42, 'Seed', 'Lines & colors', 0, 2147483647)
    subject_first: bool = control(True, 'Color characters first', 'Lines & colors')
    subject_resolution: int = control(1024, 'ISNet resolution', 'Lines & colors', 256, 1536, 128)
    subject_threshold: float = control(.5, 'Subject mask threshold', 'Lines & colors', .05, .95, .01)
    clean_ink: bool = control(True, 'Multiscale line cleanup', 'Lines & colors')
    grid_edge: int = control(10, 'Line grid cell size (px)', 'Lines & colors', 2, 40)
    adaptive_block: int = control(15, 'Line detection window (odd)', 'Lines & colors', 3, 51, 2)
    adaptive_c: float = control(10., 'Line detection threshold C', 'Lines & colors', -10, 30, 1)
    clahe_clip: float = control(2., 'CLAHE contrast', 'Lines & colors', .1, 8, .1)
    palette_colors: int = control(16, 'Maximum palette colors', 'Lines & colors', 1, 32)
    min_region_area: int = control(80, 'Minimum color region area (px)', 'Lines & colors', 1, 1000)
    saturation: float = control(1., 'Color saturation', 'Lines & colors', 0, 2, .05)
    brush_ratio: float = control(.12, 'Brush width / shorter image side', 'Coloring path', .01, .5, .01)
    brush_min: float = control(24., 'Minimum brush width (px)', 'Coloring path', 1, 128, 1)
    brush_max: float = control(92., 'Maximum brush width (px)', 'Coloring path', 1, 256, 1)
    stroke_angle: float = control(12., 'Coloring stroke angle (degrees)', 'Coloring path', -80, 80, 1)
    travel_speed_ratio: float = control(1.5, 'Travel speed / coloring speed', 'Coloring path', .1, 5, .1)
    path_samples: int = control(4097, 'Hand path interpolation samples', 'Coloring path', 257, 16385, 256)
    show_hand: bool = control(True, 'Show hand', 'Hand & pen')
    hand_scale: float = control(1., 'Hand scale', 'Hand & pen', .1, 2, .05)
    wrist_degrees: float = control(11., 'Wrist rotation amplitude (degrees)', 'Hand & pen', 0, 30, .5)
    wrist_frequency: float = control(.55, 'Wrist motion frequency (Hz)', 'Hand & pen', 0, 3, .05)
    finger_motion: float = control(1., 'Finger motion', 'Hand & pen', 0, 3, .1)
    shadow_strength: float = control(1., 'Hand shadow strength', 'Hand & pen', 0, 2, .1)
    draw_tip_x: int = control(0, 'Draw pen tip: X offset (native px)', 'Hand & pen', -100, 100)
    draw_tip_y: int = control(0, 'Draw pen tip: Y offset (native px)', 'Hand & pen', -100, 100)
    color_tip_x: int = control(0, 'Color pen tip: X offset (native px)', 'Hand & pen', -100, 100)
    color_tip_y: int = control(0, 'Color pen tip: Y offset (native px)', 'Hand & pen', -100, 100)
    rig_json: str = control('{}', 'Advanced hand rig (JSON)', 'Advanced')
    ink_json: str = control('{}', 'Advanced line filter (JSON)', 'Advanced')

    @property
    def phase_seconds(self):
        drawing = self.duration * self.draw_percent / 100
        ink = drawing * self.ink_percent / 100
        return ink, drawing - ink, self.duration - drawing

    def validate(self):
        for item in fields(self):
            value, spec = getattr(self, item.name), item.metadata
            if spec['choices']:
                choices = [entry[1] if isinstance(entry, tuple) else entry for entry in spec['choices']]
                if value not in choices:
                    raise ValueError(f'{item.name}: invalid choice')
            elif isinstance(item.default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f'{item.name}: boolean required')
            elif spec['low'] is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{item.name}: finite number required')
                if not spec['low'] <= value <= spec['high']:
                    raise ValueError(f'{item.name}: must be {spec["low"]}..{spec["high"]}')
                if isinstance(item.default, int) and int(value) != value:
                    raise ValueError(f'{item.name}: integer required')
        if self.adaptive_block % 2 != 1 or self.brush_min > self.brush_max:
            raise ValueError('Invalid adaptive window or brush width range')
        if self.duration * self.fps < 1:
            raise ValueError('Duration must cover at least one frame at the selected FPS')
        if self.encoder == 'nvenc' and self.device != 'cuda':
            raise ValueError('NVENC requires CUDA; select the CPU encoder for CPU rendering')
        if not isinstance(self.paper_color, str) or len(ImageColor.getrgb(self.paper_color)) != 3:
            raise ValueError('Paper color must be opaque RGB')
        rig_settings(self)
        ink_options(self)
        return self


INK_DEFAULTS = {'bilateral_first': [7, 12, 3], 'bilateral_second': [5, 12, 2],
                'clahe_tiles': 8, 'coarse_c': [5, 4], 'support_dilate': 3,
                'component_area': 4, 'component_span': 8, 'barrier_size': 3}


def ink_options(settings):
    if not isinstance(settings.ink_json, str) or len(settings.ink_json) > 20000:
        raise ValueError('Ink settings must be a bounded JSON string')
    extra = json.loads(settings.ink_json)
    if not isinstance(extra, dict) or extra.keys() - INK_DEFAULTS.keys():
        raise ValueError('Unknown ink setting')
    result = {**deepcopy(INK_DEFAULTS), **extra}
    for key, value in result.items():
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 100 for v in values):
            raise ValueError(f'Invalid ink setting: {key}')
    for key in ('bilateral_first', 'bilateral_second'):
        if not isinstance(result[key], list) or len(result[key]) != 3 or result[key][0] % 2 != 1:
            raise ValueError(f'Invalid bilateral parameters: {key}')
    if not isinstance(result['coarse_c'], list) or len(result['coarse_c']) != 2:
        raise ValueError('coarse_c must contain two numbers')
    for key in ('clahe_tiles', 'support_dilate', 'barrier_size'):
        if int(result[key]) != result[key]:
            raise ValueError(f'{key} must be an integer')
    return result


def rig_settings(settings):
    if not isinstance(settings.rig_json, str) or len(settings.rig_json) > 20000:
        raise ValueError('Rig settings must be a bounded JSON string')
    rig = deepcopy(RIG_SETTINGS)
    extra = json.loads(settings.rig_json)
    if not isinstance(extra, dict) or extra.keys() - rig.keys():
        raise ValueError('Unknown rig setting')
    protected = {'reference_height', 'native_scale', 'wrist_degrees', 'wrist_frequency'}
    if extra.keys() & protected:
        raise ValueError('Use the dedicated wrist/scale controls')
    rig.update(extra)
    # Validate against the native rig shape before any tensor allocation.
    def check(value, template):
        if isinstance(template, dict):
            if not isinstance(value, dict) or value.keys() != template.keys():
                raise ValueError('Rig keys must match the native schema')
            for key in template:
                check(value[key], template[key])
        elif isinstance(template, list):
            if not isinstance(value, list) or len(value) != len(template):
                raise ValueError('Rig arrays must match the native schema')
            for item, reference in zip(value, template):
                check(item, reference)
        elif isinstance(template, bool):
            if value != template:
                raise ValueError('The approved hand orientation is fixed')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 4096:
            raise ValueError('Rig values must be finite and bounded')
    check(rig, RIG_SETTINGS)
    if len(rig['oscillations']) != 4 or min(rig['contact_variance']) <= 0 or min(rig['contact_offset']) <= 0:
        raise ValueError('Invalid contact shadow')
    for key in ('marker_dilate', 'marker_falloff', 'nib_radius'):
        if not 1 <= rig[key] <= 100 or int(rig[key]) != rig[key]:
            raise ValueError(f'Invalid rig setting: {key}')
    if any(not 0 <= opacity <= 1 or not 0 < sigma <= 100 for opacity, sigma in rig['shadow']):
        raise ValueError('Invalid cast shadow')
    if (any(value < 0 for key in ('shadow_offset', 'lift_shadow_offset') for value in rig[key])
            or not 0 <= rig['lift_scale'] <= 1 or not 0 <= rig['lift_pixels'] <= 100
            or not 0 <= rig['lift_shadow_fade'] <= 1 or not 0 <= rig['shadow_max'] <= 1
            or not 0 <= rig['contact_opacity'] <= 1):
        raise ValueError('Invalid lift or shadow settings')
    rig['wrist_degrees'], rig['wrist_frequency'] = settings.wrist_degrees, settings.wrist_frequency
    rig['shadow_max'] = min(1, max(0, rig['shadow_max'] * settings.shadow_strength))
    rig['contact_opacity'] *= settings.shadow_strength
    for shadow in rig['shadow']:
        shadow[0] *= settings.shadow_strength
    for mode in ('draw', 'color'):
        pose = rig['poses'][mode]
        if pose['native_size'] != RIG_SETTINGS['poses'][mode]['native_size'] or not .1 <= pose['native_scale'] <= 2:
            raise ValueError('Hand dimensions are fixed; scale must be positive')
        pose['native_scale'] *= settings.hand_scale
        pose['tip'][0] += getattr(settings, mode + '_tip_x')
        pose['tip'][1] += getattr(settings, mode + '_tip_y')
        if any(int(value) != value for value in pose['tip']):
            raise ValueError('Nib coordinates must be integers')
        pose['tip'] = [int(value) for value in pose['tip']]
        pose['native_size'] = list(RIG_SETTINGS['poses'][mode]['native_size'])
        if any(not 0 <= p < size for p, size in zip(pose['tip'], pose['native_size'])):
            raise ValueError('Nib anchor lies outside the hand image')
        for center in pose['centers']:
            if center[2] <= 0:
                raise ValueError('Motion radius must be positive')
            center[-2] *= settings.finger_motion
            center[-1] *= settings.finger_motion
    return rig
