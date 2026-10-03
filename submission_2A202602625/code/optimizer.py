"""Chọn bộ tối ưu PyTorch; gradient L2 toàn cục trước clipping."""
import math
import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9,
                    betas=(0.9, 0.999), eps=1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    if lr is None or not math.isfinite(lr) or lr <= 0 or weight_decay < 0:
        raise ValueError("lr phải hữu hạn, > 0; weight_decay >= 0")
    if name in ("sgd", "sgd_momentum"):
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=momentum if name == "sgd_momentum" else 0.0)
    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
    return cls(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def build_scheduler(optimizer, name, total_steps, **kwargs):
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")


def clip_gradients(params, max_norm=None):
    """max_norm=None chỉ đo; giá trị trả về luôn là chuẩn TRƯỚC khi cắt."""
    if max_norm is not None and (not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("clip_norm phải hữu hạn và > 0")
    return float(torch.nn.utils.clip_grad_norm_(params, float("inf") if max_norm is None else max_norm))
