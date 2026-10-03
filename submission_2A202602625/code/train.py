"""Pipeline chung, đo bằng val, AMP và checkpoint khôi phục theo epoch."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce", optimizer="sgd_momentum", lr=None, weight_decay=0.0, momentum=0.9,
    betas=(0.9, 0.999), eps=1e-8, batch=512, epochs=20, hidden=(256, 128),
    dropout=0.0, init="he", clip_norm=None, precision="fp32", seed=1,
    train_eval_size=50_000, eval_batch=8192, out_dir=None, resume=True,
    verbose=True, prediction="", notes="")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * cm.diagonal(), denominator, out=np.zeros(7), where=denominator > 0).mean())


def compute_loss(logits, y, loss_name):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        # MSE trên LOGITS và one-hot, mean B*7, không có hệ số 1/2.
        return F.mse_loss(logits, F.one_hot(y, 7).to(logits.dtype))
    raise ValueError(f"Unknown loss: {loss_name}")


@torch.no_grad()
def predict(model, X, batch_size=8192):
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name="ce", batch_size=8192):
    """FP32, eval(), trung bình có trọng số, F1 của lớp vắng mặt = 0."""
    if len(X) == 0 or len(X) != len(y) or batch_size <= 0:
        raise ValueError("Tập đánh giá rỗng/không khớp hoặc batch không hợp lệ")
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=X.device)
    cm = torch.zeros(49, dtype=torch.int64, device=X.device)
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits = model(xb)
        total += compute_loss(logits, yb, loss_name).double() * len(yb)
        cm += torch.bincount(yb * 7 + logits.argmax(1), minlength=49)
    matrix = cm.reshape(7, 7).cpu().numpy()
    return dict(loss=total.item() / len(y), acc=float(matrix.trace() / len(y)),
                macro_f1=macro_f1_from_confusion(matrix))


def _signature(cfg, data):
    payload = {k: v for k, v in cfg.items() if k not in {"out_dir", "resume", "verbose"}}
    payload["data"] = data.get("signature", {k: list(data[k].shape) for k in ("X_tr", "y_tr", "X_val", "y_val")})
    code = hashlib.sha256()
    for name in ("train.py", "model.py", "optimizer.py", "data.py"):
        code.update(Path(__file__).with_name(name).read_bytes())
    payload["code"] = code.hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_experiment(cfg, data):
    """Không truy cập eval. Lưu mỗi epoch; resume từ epoch đã hoàn tất.

    Train loss đo ở eval trên tập con cố định (seed 42, mặc định 50k);
    val dùng toàn bộ. Thời gian gồm train + đo metric, loại ghi file/vẽ.
    Summary val_acc/F1 lấy tại epoch có val_loss nhỏ nhất.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError("Chỉ dùng các kiến trúc quy định")
    if min(cfg["epochs"], cfg["batch"], cfg["eval_batch"]) <= 0:
        raise ValueError("epochs, batch, eval_batch phải > 0")
    if cfg["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision phải là fp32/fp16/bf16")
    if not cfg["exp_id"] or Path(cfg["exp_id"]).name != cfg["exp_id"]:
        raise ValueError("exp_id phải là tên file")
    device = data["X_tr"].device
    set_seed(cfg["seed"])
    model = MLP(hidden, cfg["dropout"], cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "grad_norm", "grad_norm_max", "clip_fraction", "epoch_time_s", "peak_mem_MB", "skipped_updates")}
    result = dict(cfg=cfg, history=history, summary={}, signature=_signature(cfg, data),
        environment=dict(torch=torch.__version__, device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    out = Path(cfg["out_dir"]) if cfg["out_dir"] else None
    checkpoint = out / "checkpoints" / f"{cfg['exp_id']}.pt" if out else None

    def save_artifacts():
        if out:
            from results_table import save_result
            from plots import plot_run
            save_result(result, str(out / "results"))
            plot_run(result, str(out / "figures" / f"{cfg['exp_id']}.png"))

    reason = None
    if cfg["precision"] != "fp32" and device.type != "cuda":
        reason = "AMP cần CUDA; không thay bằng FP32 để báo kết quả AMP"
    elif cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        reason = "GPU không hỗ trợ BF16"
    if reason:
        result["summary"] = dict(status="skipped", diverged=False, reason=reason, best_epoch=0)
        save_artifacts()
        print(f"{cfg['exp_id']}: SKIPPED ({reason})")
        return result
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"],
                                cfg["momentum"], cfg["betas"], cfg["eps"])
    fp16 = cfg["precision"] == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    indices = torch.randperm(len(data["X_tr"]), generator=torch.Generator().manual_seed(42))
    n_monitor = min(cfg["train_eval_size"] or len(indices), len(indices))
    indices = indices[:n_monitor].to(device)
    X_monitor, y_monitor = data["X_tr"][indices], data["y_tr"][indices]
    best_state, best_epoch, best_loss = None, 0, float("inf")
    start_epoch, diverged, peak_mem = 1, False, 0.0
    if checkpoint and checkpoint.exists() and cfg["resume"]:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["result"]["signature"] != result["signature"]:
            raise ValueError(f"Checkpoint {cfg['exp_id']} khác code/cfg/data; dùng OUT_DIR hoặc exp_id mới")
        result = saved["result"]
        result["cfg"] = cfg
        history = result["history"]
        best_state, best_epoch, best_loss = saved["best_state"], saved["best_epoch"], saved["best_loss"]
        if result["summary"].get("status") in ("complete", "diverged"):
            result["best_state"] = best_state
            save_artifacts()
            print(f"{cfg['exp_id']}: dùng kết quả đã lưu")
            return result
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["generator"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch = saved["epoch"] + 1
        peak_mem = max([v for v in history["peak_mem_MB"] if v is not None], default=0.0)
        print(f"{cfg['exp_id']}: tiếp tục epoch {start_epoch}")
    else:
        result["initial"] = dict(val=evaluate(model, data["X_val"], data["y_val"], cfg["loss"], cfg["eval_batch"]),
            activation_std=activation_stats(model, data["X_val"][:4096]),
            activation_measurement="sau mỗi Linear, eval mode", train_monitor_size=n_monitor,
            updates_per_epoch=math.ceil(len(data["X_tr"]) / cfg["batch"]))
    step0 = result["initial"]["val"]["loss"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        grad_norms, clip_count, skipped = [], 0, 0
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.bfloat16,
                                enabled=cfg["precision"] != "fp32"):
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss).item():
                diverged = True
                break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Cả khi không clip, gn phải ở thang đo thật.
            gn = clip_gradients(model.parameters(), None)
            if not math.isfinite(gn):
                if fp16:
                    scaler.step(optimizer)  # GradScaler bỏ cập nhật overflow, giảm scale.
                    scaler.update()
                    skipped += 1
                    continue
                diverged = True
                break
            grad_norms.append(gn)
            if cfg["clip_norm"] is not None:
                clip_count += int(gn > cfg["clip_norm"])
                clip_gradients(model.parameters(), cfg["clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        tr = evaluate(model, X_monitor, y_monitor, cfg["loss"], cfg["eval_batch"])
        val = evaluate(model, data["X_val"], data["y_val"], cfg["loss"], cfg["eval_batch"])
        diverged |= not math.isfinite(tr["loss"]) or not math.isfinite(val["loss"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 2**20)
        elapsed = time.perf_counter() - started
        values = dict(epoch=epoch, train_loss=tr["loss"], val_loss=val["loss"], val_acc=val["acc"],
            val_macro_f1=val["macro_f1"], grad_norm=float(np.mean(grad_norms)) if grad_norms else None,
            grad_norm_max=max(grad_norms, default=None), clip_fraction=clip_count / len(grad_norms) if grad_norms else None,
            epoch_time_s=elapsed, peak_mem_MB=peak_mem if device.type == "cuda" else None, skipped_updates=skipped)
        for key, value in values.items():
            history[key].append(value if not isinstance(value, float) or math.isfinite(value) else None)
        if not diverged and val["loss"] < best_loss:
            best_state, best_loss, best_epoch = _snapshot(model), val["loss"], epoch
        bi = history["epoch"].index(best_epoch) if best_epoch else None
        fractions = [v for v in history["clip_fraction"] if v is not None]
        result["summary"] = dict(status="diverged" if diverged else ("complete" if epoch == cfg["epochs"] else "running"),
            step0_loss=step0 if math.isfinite(step0) else None, best_val_loss=best_loss if best_epoch else None,
            best_epoch=best_epoch, final_train_loss=history["train_loss"][-1], final_val_loss=history["val_loss"][-1],
            val_acc=history["val_acc"][bi] if best_epoch else None, val_macro_f1=history["val_macro_f1"][bi] if best_epoch else None,
            time_per_epoch_s=float(np.mean(history["epoch_time_s"])), peak_mem_MB=peak_mem if device.type == "cuda" else None,
            diverged=bool(diverged), clip_fraction=float(np.mean(fractions)) if fractions else None,
            skipped_updates=sum(history["skipped_updates"]))
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(dict(result=result, model=_snapshot(model), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                best_state=best_state, best_epoch=best_epoch, best_loss=best_loss, epoch=epoch,
                generator=generator.get_state(), torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None), tmp)
            os.replace(tmp, checkpoint)
        save_artifacts()
        if cfg["verbose"]:
            print(f"{cfg['exp_id']} {epoch}/{cfg['epochs']}: train={tr['loss']:.4f} val={val['loss']:.4f} "
                  f"acc={val['acc']:.4f} F1={val['macro_f1']:.4f} {elapsed:.1f}s" + (" DIVERGED" if diverged else ""), flush=True)
        if diverged:
            break
    result["best_state"] = best_state
    return result


def write_predictions(row_id, preds, path):
    ids, labels = np.asarray(row_id), np.asarray(preds)
    if ids.ndim != 1 or labels.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("row_id/pred phải cùng shape 1D; row_id duy nhất")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("row_id và pred phải là số nguyên")
    if np.any((labels < 0) | (labels > 6)):
        raise ValueError("pred phải trong 0..6")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "pred"])
        writer.writerows(zip(ids.tolist(), labels.tolist()))


def final_eval(cfg, result, data, pred_path):
    """Chỉ dự đoán; metric chính thức lấy bằng scripts/evaluate.py."""
    state = result.get("best_state")
    if state is None:
        state = torch.load(Path(cfg["out_dir"]) / "checkpoints" / f"{cfg['exp_id']}.pt",
                           map_location="cpu", weights_only=False)["best_state"]
    if state is None or result["summary"].get("status") != "complete":
        raise ValueError("Cần kết quả hoàn tất với best_state hợp lệ")
    model = MLP(tuple(cfg["hidden"]), cfg["dropout"], cfg["init"]).to(data["X_eval"].device)
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])]
    model.load_state_dict(state)
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
