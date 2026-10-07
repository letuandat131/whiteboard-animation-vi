"""Native zero-copy Torch/CUDA to NVIDIA H.264 muxer."""
from __future__ import annotations
from collections import deque
import gc
import os
import pathlib
import importlib.util
from pathlib import Path
import torch
VIDEO_TIMEBASE = 90000

class _CudaArray:
    def __init__(self, tensor: torch.Tensor):
        self.__cuda_array_interface__ = {
            "shape": tuple(tensor.shape),
            "strides": tuple(tensor.stride(i) * tensor.element_size() for i in range(tensor.ndim)),
            "data": (tensor.data_ptr(), False),
            "typestr": "|u1",
            "version": 3,
        }


class _CudaFrame:
    """Keeps a Torch allocation alive while PyNvVideoCodec reads its CUDA pointer."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.interface = _CudaArray(tensor)

    def cuda(self):
        return self.interface


def _pack_nvenc_abgr(rgb: torch.Tensor) -> torch.Tensor:
    """Pack RGB frames into PyNvVideoCodec's ABGR surface byte layout."""
    rgba = torch.empty((*rgb.shape[:3], 4), dtype=torch.uint8, device=rgb.device)
    rgba[..., :3] = rgb
    rgba[..., 3].fill_(255)
    return rgba


def _prepare_nvenc_surface(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.shape[-1] == 4:
        return rgb
    rgba = _pack_nvenc_abgr(rgb)
    if rgb.is_cuda:
        # PyNvVideoCodec consumes the raw CUDA pointer outside Torch. Ensure
        # the asynchronous RGB -> RGBA pack is complete before NVENC reads it.
        torch.cuda.current_stream(rgb.device).synchronize()
    return rgba


def _load_pynv():
    torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
    spec = importlib.util.find_spec('PyNvVideoCodec')
    if spec is None:
        raise RuntimeError('Install requirements-gpu.txt to use NVENC.')
    package = pathlib.Path(spec.origin).parent
    handles = []
    if os.name == "nt":
        for directory in (torch_lib, package):
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    try:
        import PyNvVideoCodec as nvc
    except Exception as exc:
        raise RuntimeError(f"Could not load NVIDIA PyNvVideoCodec: {exc}") from exc
    # The handles must stay alive while the extension is loaded.
    nvc._v3_dll_handles = handles
    return nvc


class _NvencMuxer:
    def __init__(self, output: Path, settings):
        self.nvc = _load_pynv()
        self.output = output
        config = {
            "gpu_id": 0,
            "codec": "h264",
            "preset": str(settings.encoder_preset).upper(),
            "tuning_info": "high_quality",
            "fps": str(settings.fps),
            "gop": str(settings.fps * 2),
            "bf": "0",
            "rc": "vbr",
        }
        quality_rate_control = settings.encoder_rate_control == "vbr_cq"
        if quality_rate_control:
            config["cq"] = str(settings.encoder_cq)
        else:
            config["bitrate"] = str(_parse_bitrate(settings.video_bitrate))
            # PyNvVideoCodec uses maxbitrate (no underscore).
            config["maxbitrate"] = str(_parse_bitrate(settings.video_maxrate))
        self.encoder = self.nvc.CreateEncoder(
            settings.width, settings.height, "ABGR", False, **config
        )
        effective = self.encoder.GetEncodeReconfigureParams()
        requested_bitrate = _parse_bitrate(settings.video_bitrate)
        requested_max_bitrate = _parse_bitrate(settings.video_maxrate)
        self.effective_rate_control = {
            "average_bitrate": int(effective.averageBitrate),
            "max_bitrate": int(effective.maxBitRate),
            "mode": str(effective.rateControlMode),
            "target_quality": int(effective.targetQuality),
        }
        if quality_rate_control:
            if (
                "VBR" not in self.effective_rate_control["mode"].upper()
                or self.effective_rate_control["target_quality"] != settings.encoder_cq
            ):
                raise RuntimeError(
                    "NVENC did not apply the requested quality mode: "
                    f"requested CQ {settings.encoder_cq}, got "
                    f"{self.effective_rate_control['mode']} / "
                    f"CQ {self.effective_rate_control['target_quality']}."
                )
        else:
            if self.effective_rate_control["average_bitrate"] <= 0:
                raise RuntimeError("NVENC did not accept the target bitrate; stopped to prevent an oversized output file.")
            average_error = abs(
                self.effective_rate_control["average_bitrate"] - requested_bitrate
            ) / max(1, requested_bitrate)
            max_error = abs(
                self.effective_rate_control["max_bitrate"] - requested_max_bitrate
            ) / max(1, requested_max_bitrate)
            if average_error > 0.1 or max_error > 0.1:
                raise RuntimeError(
                    "NVENC did not apply the configured bitrate: "
                    f"requested {requested_bitrate}/{requested_max_bitrate}, got "
                    f"{self.effective_rate_control['average_bitrate']}/"
                    f"{self.effective_rate_control['max_bitrate']}."
                )
        self.muxer = self.nvc.FFmpegMuxer(
            file_path=str(output),
            media_format=self.nvc.GetMediaFormat(str(output)),
            codec="h264",
            width=settings.width,
            height=settings.height,
            fps_num=settings.fps,
            fps_den=1,
            timebase_num=1,
            timebase_den=VIDEO_TIMEBASE,
            extradata=self.encoder.GetSequenceParams(),
        )
        self.uniform_pts = not settings.hybrid_vfr
        self.pts_increment = max(1, VIDEO_TIMEBASE // settings.fps)
        if self.uniform_pts:
            self.muxer.SetUniformPtsIncrement(self.pts_increment)
        self.frame_index = 0
        self.last_pts = -1
        self._inflight: deque[_CudaFrame] = deque(maxlen=8)

    def write(self, rgb: torch.Tensor) -> None:
        self.write_batch(rgb.unsqueeze(0))

    def _record_vfr_timestamp(self, pts: int) -> None:
        if self.uniform_pts:
            return
        source_packet = self.nvc.PacketData()
        source_packet.pts = pts
        source_packet.dts = pts
        source_packet.duration = self.pts_increment
        source_packet.is_video = 1
        source_packet.discardable = 0
        source_packet.stream_index = 0
        self.muxer.RecordSourceVideoPacket(source_packet)

    def write_batch(self, rgb: torch.Tensor, pts_ticks: list[int] | None = None) -> int:
        if rgb.dtype != torch.uint8 or not rgb.is_cuda:
            raise TypeError("NVENC requires a CUDA uint8 batch.")
        if rgb.ndim != 4 or rgb.shape[-1] not in {3, 4}:
            raise TypeError(f"NVENC requires BHWC RGB/RGBA, got {tuple(rgb.shape)}")
        if pts_ticks is not None and len(pts_ticks) != int(rgb.shape[0]):
            raise ValueError("PTS count does not match the NVENC frame count.")
        # PyNvVideoCodec names the surface ABGR by its packed 32-bit layout,
        # while the CUDA bytes it consumes are R,G,B,A. Swapping R/B here
        # turns every warm frame blue after encode (caught by the RGB parity
        # gate), so keep the compositor's RGB byte order and only append alpha.
        rgba = _prepare_nvenc_surface(rgb)
        for index, surface in enumerate(rgba):
            params = self.nvc.NV_ENC_PIC_PARAMS()
            pts = (
                int(pts_ticks[index])
                if pts_ticks is not None
                else self.frame_index * (VIDEO_TIMEBASE // 60 if not self.uniform_pts else 1)
            )
            if pts <= self.last_pts:
                raise RuntimeError(f"NVENC PTS is not increasing: {pts} <= {self.last_pts}")
            self._record_vfr_timestamp(pts)
            params.inputTimeStamp = pts
            frame = _CudaFrame(surface)
            self._inflight.append(frame)
            for packet in self.encoder.Encode(frame, params):
                self.muxer.MuxVideoPacket(
                    bytes(packet["data"]), packet["picture_type"], packet["timestamp"]
                )
            self.frame_index += 1
            self.last_pts = pts
        return int(rgb.shape[0])

    def close(self) -> None:
        for packet in self.encoder.EndEncode():
            self.muxer.MuxVideoPacket(
                bytes(packet["data"]), packet["picture_type"], packet["timestamp"]
            )
        self.muxer.Finalize()
        # The Windows FFmpeg muxer releases its file handle in the destructor.
        self.encoder = None
        self.muxer = None
        self._inflight.clear()
        gc.collect()

    def abort(self) -> None:
        self.encoder = None
        self.muxer = None
        self._inflight.clear()
        gc.collect()


def _parse_bitrate(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    scale = 1
    if text.endswith("k"):
        scale, text = 1000, text[:-1]
    elif text.endswith("m"):
        scale, text = 1_000_000, text[:-1]
    return int(float(text) * scale)


