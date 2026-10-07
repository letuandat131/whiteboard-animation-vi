"""Explicit, verified download of the pinned ISNet checkpoint (stdlib only)."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import time
import urllib.request

MODEL_URL = 'https://huggingface.co/skytnt/anime-seg/resolve/493cb60893f47441b26ec4fb9a306bce9e342982/model.safetensors'
EXPECTED_SIZE = 203982056
EXPECTED_SHA256 = '3351563ba8b61a01a66bacb79cf36aabce2da62d88b7606a719f573f23fe5d3e'
DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'assets' / 'v181d' / 'model.safetensors'
CHUNK_SIZE = 1024 * 1024


def verify_model(path):
    if not path.is_file() or path.stat().st_size != EXPECTED_SIZE:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest() == EXPECTED_SHA256


def download_model(destination=DEFAULT_PATH):
    destination = Path(destination)
    if verify_model(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    deadline = time.monotonic() + 600
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.',
                                         suffix='.part', delete=False) as target:
            temporary = Path(target.name)
            digest, size = hashlib.sha256(), 0
            request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Handdraw-model-downloader'})
            with urllib.request.urlopen(request, timeout=30) as source:
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Model download exceeded 10 minutes')
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > EXPECTED_SIZE:
                        raise ValueError('Model exceeds the expected size')
                    digest.update(chunk)
                    target.write(chunk)
            if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
                raise ValueError('Model size or SHA256 mismatch')
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    try:
        print(download_model(args.output))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Model download failed: {exc}\n')


if __name__ == '__main__':
    main()
