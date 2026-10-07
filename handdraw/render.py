from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, fields, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import uuid

import cv2
import numpy as np
from PIL import Image, ImageColor
import torch

from .pixels import draw_batch
from .settings import Settings, ink_options, rig_settings
from .v181d_hand import MovingHand, scheduled_hands
from .v181d_plan import build_v181d_plan


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
_PLAN_CACHE = OrderedDict()
_RENDER_LOCK = threading.Lock()
PLAN_FIELDS = ('mode', 'device', 'seed', 'subject_first', 'subject_resolution', 'subject_threshold',
               'clean_ink', 'grid_edge', 'adaptive_block', 'adaptive_c', 'clahe_clip', 'palette_colors',
               'min_region_area', 'brush_ratio', 'brush_min', 'brush_max', 'stroke_angle',
               'travel_speed_ratio', 'path_samples', 'ink_json')


def fit_image(image, settings):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] not in (3, 4) or min(image.shape[:2]) < 1 or image.shape[0] * image.shape[1] > 32000000):
        raise ValueError('A nonempty RGB/RGBA uint8 image of at most 32 megapixels is required')
    paper = ImageColor.getrgb(settings.paper_color)
    rgb = image[:, :, :3].copy()
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        rgb = np.rint(rgb * alpha + np.asarray(paper) * (1 - alpha)).astype(np.uint8)
    source = Image.fromarray(rgb)
    width, height = int(settings.width) // 2 * 2, int(settings.height) // 2 * 2
    if settings.fit == 'cover':
        scale = max(width / source.width, height / source.height)
        size = (max(width, round(source.width * scale)), max(height, round(source.height * scale)))
        source = source.resize(size, Image.Resampling.LANCZOS)
        left, top = (source.width - width) // 2, (source.height - height) // 2
        source = source.crop((left, top, left + width, top + height))
    else:
        scale = min(width / source.width, height / source.height)
        size = (max(2, round(source.width * scale) // 2 * 2), max(2, round(source.height * scale) // 2 * 2))
        source = source.resize(size, Image.Resampling.LANCZOS)
        if settings.fit == 'contain':
            canvas = Image.new('RGB', (width, height), paper)
            canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
            source = canvas
    rgb = np.asarray(source).copy()
    if settings.saturation != 1:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.rint(np.clip(hsv[:, :, 1].astype(float) * settings.saturation, 0, 255)).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def build_plan(rgb, settings, *, subject_mask=None):
    advanced = ink_options(settings)
    barrier = advanced.pop('barrier_size')
    if subject_mask is None:
        if settings.subject_first:
            from .subject import detect_subject_mask, release_subject_model
            try:
                subject_mask, _ = detect_subject_mask(rgb, int(settings.subject_resolution),
                                                     settings.subject_threshold, device=settings.device)
            finally:
                release_subject_model()
        else:
            subject_mask = np.zeros(rgb.shape[:2], bool)
    plan = build_v181d_plan(rgb, int(settings.seed), subject_mask=subject_mask,
        cluster_backend=settings.device, background_clusters=settings.mode == 'e',
        grid_edge=int(settings.grid_edge), clean_ink=settings.clean_ink,
        ink_options=dict(block_size=int(settings.adaptive_block), threshold_c=settings.adaptive_c,
                         clahe_clip=settings.clahe_clip, **advanced),
        palette_colors=int(settings.palette_colors), min_region_area=int(settings.min_region_area),
        barrier_size=barrier, path_samples=int(settings.path_samples),
        fill_options=dict(brush_min=settings.brush_min, brush_max=settings.brush_max,
                          brush_ratio=settings.brush_ratio, stroke_angle=settings.stroke_angle,
                          travel_speed_ratio=settings.travel_speed_ratio))
    return plan


def _cached_plan(rgb, settings):
    key = hashlib.sha256(rgb.tobytes() + json.dumps(
        {'shape': rgb.shape, **{name: getattr(settings, name) for name in PLAN_FIELDS}}, sort_keys=True).encode()).hexdigest()
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached, True
    plan = build_plan(rgb, settings)
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > 2:
        _PLAN_CACHE.popitem(last=False)
    return plan, False


def _device_state(rgb, plan, settings):
    end = float(plan['ink_end'])
    path = plan['pen_path'].copy()
    if 0 < end < 1 and settings.ink_percent != 45:
        target = settings.ink_percent / 100
        times = path[:, 0].copy()
        path[:, 0] = np.where(times < end, times * target / end,
                             target + (times - end) * (1 - target) / (1 - end))
        end = target
    device = torch.device(settings.device)
    return {'image_compact': torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device),
            'structure': torch.from_numpy(plan['structure_time']).to(device),
            'residual_time_compact': torch.from_numpy(plan['color_reveal_time']).to(device),
            'color_contact_time': torch.from_numpy(plan['color_contact_time']).to(device),
            'time_scale': 254, 'ink_end': end, 'pen_path': path, 'pen_lift': plan['pen_lift']}


def ffmpeg_binary():
    configured = os.environ.get('FFMPEG_BINARY')
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            raise ValueError('FFMPEG_BINARY does not point to a file')
        return str(candidate)
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    conventional = Path('C:/ffmpeg/bin/ffmpeg.exe')
    if os.name == 'nt' and conventional.is_file():
        return str(conventional)
    raise RuntimeError('FFmpeg is required for the CPU encoder. Add it to PATH or set FFMPEG_BINARY.')


class CpuEncoder:
    def __init__(self, path, width, height, settings):
        self.error_log = tempfile.TemporaryFile()
        command = [ffmpeg_binary(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                   '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(settings.fps),
                   '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', settings.cpu_preset,
                   '-crf', str(settings.quality), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self.error_log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def write_batch(self, frames):
        try:
            self.process.stdin.write(frames.detach().cpu().contiguous().numpy().tobytes())
        except BrokenPipeError as exc:
            self.error_log.seek(0)
            raise RuntimeError(self.error_log.read().decode('utf-8', errors='replace')[-3000:]) from exc

    def close(self):
        self.process.stdin.close()
        result = self.process.wait(timeout=120)
        self.error_log.seek(0)
        detail = self.error_log.read().decode('utf-8', errors='replace')
        self.error_log.close()
        if result != 0:
            raise RuntimeError(f'FFmpeg failed: {detail[-3000:]}')

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        if not self.process.stdin.closed:
            self.process.stdin.close()
        self.error_log.close()


def render_video(image, settings=None, *, output_dir=None, progress=None):
    settings = (settings or Settings()).validate()
    integers = {item.name: int(getattr(settings, item.name)) for item in fields(settings)
                if isinstance(item.default, int) and not isinstance(item.default, bool)}
    settings = replace(settings, **integers)
    if settings.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Select CPU and the CPU encoder explicitly.')
    notify = progress or (lambda fraction, description: None)
    with _RENDER_LOCK, torch.inference_mode():
        started = time.perf_counter()
        notify(0, 'Preparing image')
        rgb = fit_image(image, settings)
        height, width = rgb.shape[:2]
        notify(.05, 'Detecting characters and planning strokes')
        plan, cache_hit = _cached_plan(rgb, settings)
        state = _device_state(rgb, plan, settings)
        frames_count = max(1, round(settings.duration * settings.fps))
        draw_frames = max(1, min(frames_count, round(frames_count * settings.draw_percent / 100)))
        block = SimpleNamespace(draw_frames=draw_frames, start_frame=0)
        hands, counts = {}, {'draw': 0, 'color': 0}
        rig = rig_settings(settings)
        directory = Path(output_dir or OUTPUTS).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        name = f'{settings.mode.upper()}_{settings.draw_percent:g}-{100-settings.draw_percent:g}_{settings.ink_percent:g}-{100-settings.ink_percent:g}_{uuid.uuid4().hex[:10]}'
        output = directory / f'{name}.mp4'
        partial = directory / f'.{name}.part.mp4'
        encoder = None
        try:
            notify(.15, 'Initializing encoder')
            if settings.encoder == 'nvenc':
                from .nvenc import _NvencMuxer
                options = SimpleNamespace(width=width, height=height, fps=settings.fps,
                    encoder_preset=settings.encoder_preset, encoder_rate_control='vbr_cq',
                    encoder_cq=settings.quality, video_bitrate='2200k', video_maxrate='4000k', hybrid_vfr=False)
                encoder = _NvencMuxer(partial, options)
            else:
                encoder = CpuEncoder(partial, width, height, settings)
            for first in range(0, frames_count, settings.render_batch_size):
                indices = list(range(first, min(frames_count, first + settings.render_batch_size)))
                frames = draw_batch(state, indices, draw_frames, ImageColor.getrgb(settings.paper_color))
                frames = frames.permute(0, 2, 3, 1).contiguous()
                if settings.show_hand:
                    events = scheduled_hands(state, block, indices, {'x': 0, 'y': 0}, 0, settings.fps)
                    for slot, mode, point, seconds, lift in events:
                        if mode not in hands:
                            hands[mode] = MovingHand(mode, width, height, settings.device, rig)
                        hands[mode].stamp(frames[slot], point, seconds, (0, 0, width, height), lift)
                        counts[mode] += 1
                encoder.write_batch(frames)
                notify(.15 + .8 * (indices[-1] + 1) / frames_count, 'Exporting video')
            encoder.close()
            encoder = None
            os.replace(partial, output)
            ink = float(state['ink_end']) * draw_frames / settings.fps
            report = {'settings': asdict(settings), 'device': settings.device, 'encoder': settings.encoder,
                      'width': width, 'height': height, 'fps': settings.fps, 'frames': frames_count,
                      'duration': frames_count / settings.fps, 'phase_seconds': {
                          'draw': ink, 'color': draw_frames / settings.fps - ink,
                          'hold': (frames_count - draw_frames) / settings.fps},
                      'hand_frames': counts, 'plan_cache_hit': cache_hit,
                      'wall_seconds': time.perf_counter() - started, 'bytes': output.stat().st_size,
                      'plan': plan['metadata']}
            output.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            notify(1, 'Done')
            return output, report
        except BaseException:
            if encoder is not None:
                encoder.abort()
            partial.unlink(missing_ok=True)
            raise
        finally:
            hands.clear()
            if settings.device == 'cuda':
                torch.cuda.empty_cache()
