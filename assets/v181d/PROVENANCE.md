# SkyTNT anime segmentation

Official source: https://github.com/SkyTNT/anime-segmentation
Source revision: `55d874013a2811cdf59c365059174c7823acf5b4`
Source license: Apache-2.0, retained at `vendor_runtime/anime_segmentation/LICENSE`.

Official checkpoint: https://huggingface.co/skytnt/anime-seg
Checkpoint revision: `493cb60893f47441b26ec4fb9a306bce9e342982`
File: `model.safetensors`, FP32, Apache-2.0 per the publisher's model card.
Size: 203982056 bytes. SHA256:
`3351563ba8b61a01a66bacb79cf36aabce2da62d88b7606a719f573f23fe5d3e`.
The adapter loads only `net.*` tensors into official `model/isnet.py:ISNetDIS`.
Training-only encoder tensors are unused. No pickle checkpoint is loaded.

Bundled model path: `assets/v181d/model.safetensors`.
The reup adapter requires the local checkpoint and never downloads it implicitly.

Inference follows upstream `inference.py:get_mask`: divide RGB by 255,
preserve aspect ratio with centered zero padding, sigmoid the first ISNet
prediction, crop padding, resize to source dimensions, then threshold.
Inference uses FP32 on CUDA when available, otherwise CPU. It preserves all
semantic foreground components and may include multiple people.
