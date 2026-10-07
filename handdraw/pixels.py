"""Native Torch ink/contact-color compositor."""
import torch

def draw_batch(state, indices, draw_frames, background=(255, 255, 255)):
    image = state["image_compact"]
    t = torch.as_tensor(indices, device=image.device, dtype=torch.float32) / max(1, draw_frames - 1)
    active = t >= 0
    if draw_frames <= 1:
        t.fill_(1)
    ink_end = float(state["ink_end"])
    scale = int(state["time_scale"])
    ink_progress = (t / max(ink_end, 1e-6)).clamp(0, 1)
    ink_limit = (ink_progress * scale).round().to(torch.int32)
    ink = state["structure"][None] <= ink_limit[:, None, None]
    ink &= state["structure"][None] != 255
    ink &= active[:, None, None]
    paint_progress = ((t - ink_end) / max(1e-6, 1 - ink_end)).clamp(0, 1)
    paint_progress.masked_fill_(t >= 1, 1)
    paint_limit = (paint_progress * scale).round().to(torch.int32)
    paint = state["residual_time_compact"][None] <= paint_limit[:, None, None]
    if "color_contact_time" in state:
        paint = state["color_contact_time"][None] <= paint_progress[:, None, None]
    paint &= (t >= ink_end)[:, None, None]
    paint &= active[:, None, None]
    paper = torch.tensor(background, device=image.device, dtype=torch.uint8).view(1, 3, 1, 1)
    frames = torch.where(ink[:, None], 0, paper)
    return torch.where(paint[:, None], image[None], frames)


