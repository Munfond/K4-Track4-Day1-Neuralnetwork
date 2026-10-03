"""Thiết kế thí nghiệm trước khi chạy; cùng pipeline và split cho mọi lượt."""
from pathlib import Path
import json
import numpy as np
from plots import plot_compare
from results_table import seed_statistics
from train import DEFAULT_CFG, run_experiment

PREDICTIONS = {
    "loss": "MSE trên logits và one-hot có thang đo/gradient khác CE. So accuracy và macro-F1, không so loss trực tiếp; chưa giả định MSE kém hơn khi chưa đo.",
    "optimizer": "Momentum tích luỹ hướng gradient; Adam/AdamW điều chỉnh bước theo moment. Mỗi optimizer được thử ít nhất hai lr trước khi so tại lr tốt nhất; wd=0 cho phép kiểm tra Adam và AdamW có trùng nhau không.",
    "hparam": "M-wide có thể khớp tốt hơn nhưng tốn thời gian/bộ nhớ. Chỉ đổi hidden; cùng batch và epoch nên số bước cập nhật bằng baseline.",
    "dropout": "q=0.3 có thể giảm khoảng cách train-val nếu đã quá khớp; có thể làm chậm học khi cả train và val còn cao. Train loss phải đo với dropout tắt.",
    "clipping": "Ngưỡng c lấy bằng 0.5 lần median chuẩn gradient trung bình theo epoch baseline. Clipping có thể giảm gai ở lr cao, nhưng không đảm bảo cứu mọi lr; phải đo clip_fraction thực tế.",
    "amp": "FP16/BF16 có thể giảm bộ nhớ và thời gian nhưng mạng nhỏ có thể bị chi phí kernel/synchronization chi phối. FP16 cần GradScaler vì khoảng biểu diễn hẹp; BF16 có khoảng gần FP32 nên không dùng scaler.",
    "init": "Zeros giữ đối xứng và ReLU(0) làm gradient lớp ẩn bằng 0; mạng chủ yếu học bias đầu ra. Normal std 0.01 dễ làm kích hoạt nhỏ; Xavier dùng Var=2/(fan_in+fan_out), He dùng 2/fan_in.",
}


def _run(cfg, data, out_dir):
    cfg = {**DEFAULT_CFG, **cfg, "out_dir": str(out_dir)}
    print(f"\nDự đoán trước {cfg['exp_id']}: {cfg.get('prediction', '')}", flush=True)
    r = run_experiment(cfg, data)
    # best_state có checkpoint; không cần giữ mọi trọng số trong RAM.
    r.pop("best_state", None)
    print("Kết quả:", r["summary"], flush=True)
    return r


def run_baselines(data, out_dir, epochs=20, tune_epochs=3, seeds=(1, 2, 3)):
    """Chọn lr bằng val; tuning ngắn tách khỏi baseline 20 epoch."""
    out_dir = Path(out_dir)
    if (out_dir / "selection.json").exists():
        # Từ thời điểm có selection: chỉ cho reuse cấu hình cũ, không chọn lại theo eval.
        print("Đã có selection.json; các lượt cùng cấu hình sẽ dùng checkpoint.")
    trials = []
    for lr in (0.01, 0.03, 0.1):
        cfg = dict(exp_id=f"tune-sgdm-lr{lr:g}", group="hparam", lr=lr, seed=1, epochs=tune_epochs,
            stage="tuning", description="Tuning lr SGDM chỉ bằng val",
            prediction="lr quá thấp có thể học chậm; lr cao hơn có thể cải thiện nhanh hoặc dao động.",
            notes=f"Tuning {tune_epochs} epoch; không so ngang baseline {epochs} epoch")
        trials.append(_run(cfg, data, out_dir))
    candidates = [r for r in trials if r["summary"].get("status") == "complete"]
    if not candidates:
        raise RuntimeError("Mọi lr tuning thất bại; kiểm tra dữ liệu/loss/gradient")
    winner = max(candidates, key=lambda r: r["summary"]["val_macro_f1"])
    base_cfg = {**DEFAULT_CFG, "lr": winner["cfg"]["lr"], "epochs": epochs, "seed": 1}
    baseline = []
    for seed in seeds:
        baseline.append(_run({**base_cfg, "exp_id": f"base-s{seed}", "seed": seed,
            "description": f"Baseline M-base, seed {seed}", "stage": "baseline",
            "prediction": "M-base SGDM sẽ vượt mốc đoán đa số; các seed cho mức dao động để đánh giá so sánh."}, data, out_dir))
    stats = seed_statistics(baseline)
    (out_dir / "baseline_selection.json").write_text(json.dumps(dict(
        chosen_lr=base_cfg["lr"], criterion="max val macro-F1 tại best-val-loss epoch của tuning",
        tuning_ids=[r["cfg"]["exp_id"] for r in trials], statistics=stats), indent=2), encoding="utf-8")
    print("Baseline seed statistics:", stats)
    plot_compare(baseline, "val_loss", out_dir / "figures/compare_baseline.png")
    return base_cfg, trials + baseline


def experiment_plan(base_cfg, baseline_results):
    """7 chủ đề; thay một yếu tố. Stress clip so theo cặp cùng lr cao."""
    base = next(r for r in baseline_results if r["cfg"]["exp_id"] == "base-s1")
    norms = [v for v in base["history"]["grad_norm"] if v is not None]
    if not norms:
        raise RuntimeError("Baseline không có gradient norm")
    clip_c = max(1e-4, 0.5 * float(np.median(norms)))
    plan = []

    def add(exp_id, group, description, **changes):
        plan.append({**base_cfg, "exp_id": exp_id, "group": group, "description": description,
                     "stage": "experiment", "prediction": PREDICTIONS[group], **changes})

    add("loss-mse", "loss", "Chỉ thay CE bằng MSE trên logits", loss="mse")
    for lr in (0.01, 0.03, 0.1):
        if lr != base_cfg["lr"]:
            add(f"opt-sgdm-lr{lr:g}", "optimizer", "SGDM full-epoch lr grid", lr=lr,
                notes="Cùng epoch/seed với baseline; hai lr còn lại bổ sung tuning ngắn để so công bằng")
    for optimizer, lrs in (("sgd", (0.03, 0.1)), ("adam", (0.0003, 0.001)), ("adamw", (0.0003, 0.001))):
        for lr in lrs:
            add(f"opt-{optimizer}-lr{lr:g}", "optimizer", f"{optimizer}: lr grid", optimizer=optimizer, lr=lr,
                notes="Optimizer và lr đi cùng grid tuning; so ở lr tốt nhất mỗi optimizer, wd=0")
    add("hparam-wide", "hparam", "Chỉ đổi độ rộng M-wide", hidden=(512, 256))
    add("drop-0.3", "dropout", "Chỉ đổi q=0.3", dropout=0.3)
    add("clip-base", "clipping", "Clip ở lr baseline", clip_norm=clip_c,
        notes=f"c={clip_c:.6g}=0.5*median epoch mean gn của base-s1; kiểm tra clip_fraction")
    high_lr = base_cfg["lr"] * 10
    add("clip-highlr-none", "clipping", "Stress lr cao, không clip", lr=high_lr,
        stress=True, notes="So theo cặp clip-highlr; lr tăng 10 lần baseline")
    add("clip-highlr-on", "clipping", "Stress cùng lr cao, có clip", lr=high_lr, clip_norm=clip_c,
        stress=True, notes="Chỉ thay clipping so với clip-highlr-none; không dùng làm ứng viên final")
    for precision in ("fp16", "bf16"):
        add(f"amp-{precision}", "amp", "Chỉ thay precision", precision=precision)
    for init in ("zeros", "normal", "xavier"):
        add(f"init-{init}", "init", "Chỉ thay khởi tạo", init=init)
    return plan


def run_menu(data, out_dir, base_cfg, baseline_results, groups=None):
    plan = experiment_plan(base_cfg, baseline_results)
    plan = [c for c in plan if groups is None or c["group"] in groups]
    path = Path(out_dir) / "experiment_plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    results = [_run(cfg, data, out_dir) for cfg in plan]
    reference = next(r for r in baseline_results if r["cfg"]["exp_id"] == "base-s1")
    for group in sorted({r["cfg"]["group"] for r in results}):
        members = [reference] + [r for r in results if r["cfg"]["group"] == group]
        # CE/MSE khác thang đo: dùng F1; init/clipping thêm loss/gn riêng.
        plot_compare(members, "val_macro_f1", Path(out_dir) / f"figures/compare_{group}.png")
        if group in ("init", "clipping"):
            metric = "grad_norm" if group == "clipping" else "val_loss"
            plot_compare(members, metric, Path(out_dir) / f"figures/compare_{group}_{metric}.png")
    return results


def choose_final(results, out_dir):
    """Chốt ID + signature trước khi nhìn eval. Seed nộp cố định bằng 1."""
    valid = [r for r in results if r["summary"].get("status") == "complete"
             and r["cfg"].get("stage") != "tuning" and not r["cfg"].get("stress") and r["cfg"]["seed"] == 1]
    if not valid:
        raise RuntimeError("Chưa có model hoàn tất để chọn")
    winner = max(valid, key=lambda r: (r["summary"]["val_macro_f1"], -r["summary"]["best_val_loss"]))
    baseline = next(r for r in valid if r["cfg"]["exp_id"] == "base-s1")
    selected = dict(final=winner["cfg"]["exp_id"], baseline="base-s1", seed=1,
        final_signature=winner["signature"], baseline_signature=baseline["signature"],
        criterion="max val_macro_f1 tại best-val-loss epoch; seed=1; loại tuning/stress/skipped/diverged")
    path = Path(out_dir) / "selection.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selected:
        raise ValueError("Cấu hình cuối đã chốt; không chọn lại sau khi xem eval. Dùng bộ kết quả cũ.")
    path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    winner["selected_final"] = True
    print("Chọn chỉ bằng val:", selected)
    return baseline, winner
