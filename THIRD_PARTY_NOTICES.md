# Third-Party Notices

## Anime Segmentation / ISNet

The vendored ISNet source in `vendor_runtime/anime_segmentation/isnet.py` comes
from [Anime Segmentation](https://github.com/SkyTNT/anime-segmentation).
It is licensed under Apache License 2.0. The complete upstream license is
preserved at `vendor_runtime/anime_segmentation/LICENSE`; retain it when
redistributing this source.

The optional ISNet checkpoint is downloaded separately from
[`skytnt/anime-seg`](https://huggingface.co/skytnt/anime-seg) at revision
`493cb60893f47441b26ec4fb9a306bce9e342982`. Its size and SHA256 are recorded in
the README and downloader. The source-code notice above does not independently
grant rights to model weights; consult the upstream model repository for their
applicable terms. The checkpoint is not included in Git.

## Grid / SRT Whiteboard

The Grid implementation in `handdraw/grid_stream.py` derives from the
MIT-licensed SRT Whiteboard source. Its complete upstream notice, including
copyright and permission terms, is preserved at
`vendor_runtime/srt_whiteboard/LICENSE`. Retain that notice in copies or
substantial portions of the derived implementation.

## Runtime Dependencies

Gradio, PyTorch, torchvision, NumPy, OpenCV, SciPy, scikit-image, safetensors,
Pillow, pytest, and optional PyNvVideoCodec are separately installed dependencies.
Each remains subject to its own upstream license and distribution terms.
FFmpeg, when installed separately for CPU encoding, also retains its own terms,
which depend on the chosen build and enabled codecs.

## Original Work

Original project code contributed by `letuandat131` is dedicated to the public
domain under CC0 1.0 Universal; see [LICENSE](LICENSE). This applies only to the
author's original code, not to third-party or upstream-derived code described
above, which retains its own terms and required notices.

Hand images, demo artwork and model weights are excluded from the CC0 dedication.
No redistribution license has been selected for the hand images or demo artwork;
confirm their applicable rights before redistributing them. Neither the Apache
nor MIT notices above should be interpreted as licensing the entire project.
