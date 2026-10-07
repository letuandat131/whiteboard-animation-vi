"""Render three supplied images and create compact, looping README previews."""
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handdraw.render import ffmpeg_binary, render_video
from handdraw.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('d', 'e'), default='e')
    parser.add_argument('images', nargs=3, type=Path)
    args = parser.parse_args()
    sources = args.images
    if len(sources) != 3 or not all(path.is_file() for path in sources):
        raise SystemExit('Supply exactly three existing input images.')
    root = Path(__file__).resolve().parents[1]
    previews = root / 'docs' / 'media'
    previews.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, 1):
        print(f'Demo {index}: {source.name}', flush=True)
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB'))
            image.convert('RGB').save(previews / f'source-{index:02d}.png')
        video, report = render_video(
            rgb, Settings(mode=args.mode, duration=8, width=1280, height=720, fps=50),
            output_dir=root / 'outputs' / 'readme-demos')
        suffix = '-d' if args.mode == 'd' else ''
        gif = previews / f'demo-{index:02d}{suffix}.gif'
        subprocess.run([
            ffmpeg_binary(), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
            '-filter_complex',
            '[0:v]fps=50,scale=640:-2:flags=lanczos,split[a][b];'
            '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
            '-loop', '0', str(gif),
        ], check=True)
        with Image.open(gif) as animation:
            assert animation.is_animated and animation.n_frames > 1
            elapsed = 0
            for frame in range(animation.n_frames):
                animation.seek(frame)
                elapsed += animation.info.get('duration', 0)
            assert abs(elapsed - 8000) < 150, elapsed
        print(f'OK: {video}\nGIF: {gif} ({gif.stat().st_size / 1048576:.2f} MiB)'
              f'\nRender: {report["wall_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
