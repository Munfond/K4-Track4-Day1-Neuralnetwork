"""Part 4: chấm bằng script gốc, bảng, báo cáo số đo và ZIP sạch."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from results_table import load_results, to_row, write_xlsx, seed_statistics
from train import final_eval


def official_evaluation(result, data, repo_root, out_dir, final=False):
    """Cache theo ID/signature/hash predictions; không chọn lại bằng điểm eval."""
    root, out = Path(repo_root).resolve(), Path(out_dir).resolve()
    stem = "" if final else "baseline_"
    pred_path = out / f"{stem}predictions_eval.csv"
    score_path = out / f"{stem}eval_result.json"
    manifest_path = out / f"{stem}eval_manifest.json"
    identity = dict(exp_id=result["cfg"]["exp_id"], signature=result["signature"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {k: manifest[k] for k in identity} != identity:
            raise ValueError("Đã đánh giá eval cho cấu hình khác; không thay cấu hình sau khi nhìn eval")
        if pred_path.exists() and score_path.exists() and hashlib.sha256(pred_path.read_bytes()).hexdigest() == manifest["pred_sha256"]:
            return json.loads(score_path.read_text(encoding="utf-8"))
        raise ValueError("Artifact eval bị thiếu/thay đổi; khôi phục bản gốc trước khi tiếp tục")
    final_eval(result["cfg"], result, data, str(pred_path))
    completed = subprocess.run([sys.executable, str(root / "scripts/evaluate.py"),
        "--pred", str(pred_path), "--out", str(score_path), "--data", str(root / "data/covtype.csv.gz"),
        "--meta", str(root / "data/split_metadata.csv")], cwd=root, check=True,
        capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(completed.stdout)
    manifest_path.write_text(json.dumps({**identity, "pred_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    return json.loads(score_path.read_text(encoding="utf-8"))


def describe_comparison(result, baseline, noise):
    s, b, c = result["summary"], baseline["summary"], result["cfg"]
    if s.get("status") != "complete":
        return f"{c['exp_id']}: {s.get('status')}; {s.get('reason', 'loss/gradient không hữu hạn, đã dừng')}."
    delta = s["val_macro_f1"] - b["val_macro_f1"]
    conclusion = "Chưa đo được nhiễu vì có ít hơn 2 seed baseline."
    if noise is not None:
        conclusion = (f"|Δ| {'vượt' if abs(delta) > noise else 'không vượt'} 2σ={noise:.4f}; "
            "đây là mốc tham khảo nhiễu baseline, chưa là kiểm định thống kê của cấu hình mới.")
    return (f"`{c['exp_id']}`: best epoch {s['best_epoch']}, val F1={s['val_macro_f1']:.4f}, "
        f"accuracy={s['val_acc']:.4f}, ΔF1 so base-s1={delta:+.4f}. {conclusion}")


def write_report(results, baseline, final, scores, out_dir, student_name="", student_id="2A202602625"):
    out = Path(out_dir)
    stats = seed_statistics(results)
    noise = stats["noise_2sigma"]
    health_file = out / "results/part1_health.json"
    health = json.loads(health_file.read_text(encoding="utf-8")) if health_file.exists() else None
    c, s = baseline["cfg"], baseline["summary"]
    lines = [f"# Báo cáo Lab Day 1 — {student_name or '(điền họ tên)'} — {student_id}",
        "", "## 1. Thiết lập", "",
        f"Môi trường: {baseline['environment']}. Forest CoverType: train gốc 464 809, eval 116 203; "
        "validation phân tầng 20%, seed 42: train 371 847, val 92 962. Chỉ chuẩn hoá 10 cột liên tục bằng train.",
        f"Baseline M-base 54→256→128→7, 47 879 tham số; CE, SGD momentum 0.9, lr={c['lr']}, batch={c['batch']}, "
        f"{c['epochs']} epoch, He, dropout=0, FP32. Train-loss đo eval mode trên tập con cố định "
        f"{baseline['initial']['train_monitor_size']:,} mẫu, val-loss trên toàn bộ val. Thời gian gồm train và đo metric, loại I/O checkpoint/vẽ.",
        "", "## 2. Kiểm tra ban đầu và độ nhiễu", ""]
    if out.name == "_smoke":
        lines.insert(1, "\n> KIỂM TRA SMOKE CPU: Part 2/3 chỉ dùng 1 024 train và 512 val, 2 epoch (tuning 1 epoch). Các kích thước đầy đủ ở phần thiết lập chỉ áp dụng Part 0/1. Không dùng các kết quả này để nộp hoặc kết luận về dữ liệu đầy đủ.\n")
    if health:
        lines += [f"Part 1: {health['parameters']:,} tham số, logits (8,7); val CE bước 0={health['loss_step0']:.6f}, "
            f"ln7={health['ln7']:.6f}, lệch={health['loss_step0']-health['ln7']:+.6f}. Logits std={health['logit_std']:.6f}; "
            "He tạo điểm số chưa đều nên không ép CE về ln7. Chuẩn hoá đã được kiểm tra ở Part 0.",
            f"20 mẫu: CE {health['loss'][0]:.6f} → {health['loss'][-1]:.8f}, accuracy {health['accuracy'][-1]:.0%}; "
            f"{len(health['gradient_norms'])} tham số có gradient >0. Đây là phép thử ghi nhớ, chưa chứng minh tổng quát hoá.",
            "", "![](figures/part1_overfit20.png)"]
    lines += ["", f"Baseline: {stats['n']} seed, {', '.join(stats['exp_ids'])}."]
    for metric in ("val_acc", "val_macro_f1"):
        m = stats[metric]
        lines.append(f"- {metric}: {m['mean']:.4f} ± {m['std']:.4f} (std mẫu)." if m["std"] is not None else f"- {metric}: {m['mean']}; chưa đủ seed để tính std.")
    lines += [f"Ngưỡng nhiễu 2σ val-F1: {noise if noise is not None else 'chưa đo'}. "
        "Các so sánh dùng seed 1; ứng viên mới chưa được chạy nhiều seed nên kết luận vượt nhiễu vẫn có hạn chế.",
        f"Baseline val accuracy={s['val_acc']:.4f} so mốc đoán đa số ≈0.4876. Best epoch={s['best_epoch']}; "
        f"train-loss cuối={s['final_train_loss']:.4f}, val-loss cuối={s['final_val_loss']:.4f}. "
        f"Δ val-loss giữa epoch cuối và đầu={baseline['history']['val_loss'][-1]-baseline['history']['val_loss'][0]:+.4f}; "
        "xem hình để xác định còn giảm hay bắt đầu quá khớp.", "", "![](figures/compare_baseline.png)",
        "", "## 3. Kết quả theo chủ đề", ""]
    mechanisms = {
        "loss": "MSE ở đây là mean((logits−one_hot)^2) trên B×7, không hệ số 1/2 và không softmax. Vì không dùng MSE trên xác suất, không suy luận bão hoà softmax từ lượt này; so F1/accuracy và tốc độ, không so trị số CE với MSE.",
        "optimizer": "Momentum tích luỹ hướng cập nhật; Adam/AdamW chia theo moment bậc hai. wd=0 khiến Adam và AdamW tương đương về công thức; khác biệt lớn ở cặp cùng lr/seed cần kiểm tra. SGDM có baseline và hai lr bổ sung chạy đủ epoch, các optimizer được so ở lr tốt nhất trong grid.",
        "hparam": "M-wide tăng số tham số lên 161 287. Cùng batch/epoch giữ số bước cập nhật bằng baseline; tăng năng lực không đảm bảo cải thiện nếu mô hình chưa được huấn luyện đủ.",
        "dropout": "Dropout là chính quy hoá; có thể giảm quá khớp nhưng làm chậm học khi train và val còn cao. Các loss đều đo eval mode để so được.",
        "clipping": "Cắt chuẩn L2 toàn cục trước update; ở lr cao phải so cặp cùng lr. Clip chỉ có tác dụng khi chuẩn vượt c; clip_fraction là bằng chứng trực tiếp.",
        "amp": "FP16 có miền biểu diễn hẹp, GradScaler tăng loss và unscale trước đo/clip; overflow làm bỏ bước và giảm scale. BF16 có số bit exponent như FP32, thường không cần scaler. Chỉ gọi nhanh hơn nếu thời gian đo cho thấy; mạng nhỏ có thể bị chi phí kernel/I/O đồng bộ chi phối.",
        "init": "He: Var=2/fan_in; Xavier_normal: Var=2/(fan_in+fan_out). Zeros không phá đối xứng, ReLU tại 0 chặn gradient lớp ẩn, chỉ bias cuối có thể học phân bố lớp. Mạng 3 Linear chưa đủ sâu để kết luận về mạng 30 lớp.",
    }
    for group in ("loss", "optimizer", "hparam", "dropout", "clipping", "amp", "init"):
        members = [r for r in results if r["cfg"]["group"] == group and r["cfg"].get("stage") != "tuning"]
        if not members:
            continue
        lines += [f"### {group}", "", f"Dự đoán trước: {members[0]['cfg']['prediction']}", ""]
        completed = [r for r in members if r["summary"].get("status") == "complete"]
        selected = completed
        if group == "optimizer":
            pool = completed + [baseline]
            selected = [max([r for r in pool if r["cfg"]["optimizer"] == opt], key=lambda r: r["summary"]["val_macro_f1"])
                        for opt in sorted({r["cfg"]["optimizer"] for r in pool})]
        for r in selected + [r for r in members if r["summary"].get("status") != "complete"]:
            text = describe_comparison(r, baseline, noise)
            if r["summary"].get("status") == "complete":
                text += f" lr={r['cfg']['lr']}; time/epoch={r['summary']['time_per_epoch_s']:.2f}s; peak GPU={r['summary']['peak_mem_MB']} MiB."
                if group == "clipping":
                    text += f" clip_fraction={r['summary']['clip_fraction']:.3f}."
                if group == "init":
                    text += f" step0={r['summary']['step0_loss']:.4f}, std sau Linear={r['initial']['activation_std']}."
                if group == "dropout":
                    text += f" gap val−train cuối={r['summary']['final_val_loss']-r['summary']['final_train_loss']:+.4f}."
            lines.append("- " + text)
        lines += ["", mechanisms[group], f"Ảnh: [so sánh {group}](figures/compare_{group}.png); "
                  "ảnh từng exp_id nằm trong figures/ và tên tương ứng ở bảng.", ""]
    if scores:
        bs, fs = scores[baseline["cfg"]["exp_id"]], scores[final["cfg"]["exp_id"]]
        lines += ["## 4. Đánh giá cuối trên eval", "", "Đã chốt selection.json bằng val trước khi gọi script gốc.",
            "", "| Model | exp_id | val F1 | eval F1 | eval accuracy |", "|---|---|---:|---:|---:|"]
        for label, r, sc in (("Baseline", baseline, bs), ("Final", final, fs)):
            lines.append(f"| {label} | {r['cfg']['exp_id']} | {r['summary']['val_macro_f1']:.4f} | {sc['macro_f1']:.4f} | {sc['accuracy']:.4f} |")
        lines += ["", f"Cải thiện eval-F1={fs['macro_f1']-bs['macro_f1']:+.4f}; final val−eval F1="
            f"{final['summary']['val_macro_f1']-fs['macro_f1']:+.4f}. Chỉ đo nhiễu trên val, chưa đo nhiễu eval, nên chưa kết luận ý nghĩa thống kê của cải thiện eval.",
            "", "| Lớp | support | precision | recall | F1 |", "|---|---:|---:|---:|---:|"]
        for row in fs["per_class"]:
            lines.append(f"| {row['cls']} | {row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
        hardest = min(fs["per_class"], key=lambda r: r["f1"])
        matrix = np.asarray(fs["confusion_matrix"])
        mistaken = matrix[hardest["cls"]].copy()
        mistaken[hardest["cls"]] = 0
        partner = int(mistaken.argmax())
        lines += ["", f"Lớp khó nhất: {hardest['cls']}, F1={hardest['f1']:.4f}, support={hardest['support']}; "
            f"nhầm nhiều nhất sang {partner} ({int(mistaken[partner])} mẫu). Mất cân bằng có thể là một nguyên nhân; "
            "tương đồng đặc trưng cần phân tích thêm, chưa được chứng minh bởi ma trận nhầm lẫn.",
            "", "![](figures/confusion_final.png)"]
    else:
        lines += ["## 4. Đánh giá cuối", "", "Chưa chạy eval chính thức; các số trên chỉ là kiểm tra pipeline."]
    lines += ["", "## 5. Câu hỏi dẫn dắt và hạn chế", "",
        "Khi loss không giảm sau 2 000 bước: (1) đo loss bước 0 và logits để kiểm tra nhãn/chuẩn hoá/khởi tạo; "
        "(2) thử ghi nhớ 20 mẫu với dropout tắt để kiểm tra vòng cập nhật; (3) kiểm tra gradient từng tham số sau backward, zero_grad và danh sách optimizer.",
        "Chưa đo nhiễu của mọi cấu hình; lr grid nhỏ, tuning baseline ngắn, cùng số epoch không đảm bảo cùng độ hội tụ. "
        "So thời gian AMP chỉ hợp lệ trên cùng GPU/runtime. Cần đọc đường cong và bổ sung nhận xét kết quả nào khớp/khác dự đoán bằng số; "
        "các giải thích cơ chế trên là cơ sở diễn giải, chưa chứng minh quan hệ nhân quả bằng riêng một seed.",
        "", "## 6. Phụ lục", "", "Số đo đầy đủ: experiments.xlsx, results/*.json; ảnh riêng figures/<exp_id>.png. "
        "Notebook cloud cần được tải về kèm output sau Run All và thay vào code/lab.ipynb trước khi đóng gói nộp."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish_submission(results, baseline, final, data, repo_root, out_dir, student_name="", evaluate_eval=True):
    out, root = Path(out_dir), Path(repo_root)
    scores = {}
    if evaluate_eval:
        scores[baseline["cfg"]["exp_id"]] = official_evaluation(baseline, data, root, out)
        if final["cfg"]["exp_id"] == baseline["cfg"]["exp_id"]:
            # Dùng cùng model: không cần đánh giá lần nữa.
            shutil.copy2(out / "baseline_predictions_eval.csv", out / "predictions_eval.csv")
            shutil.copy2(out / "baseline_eval_result.json", out / "eval_result.json")
            shutil.copy2(out / "baseline_eval_manifest.json", out / "eval_manifest.json")
        else:
            scores[final["cfg"]["exp_id"]] = official_evaluation(final, data, root, out, final=True)
        matrix = np.asarray(scores[final["cfg"]["exp_id"]]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="Blues")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xlabel="Predicted class", ylabel="True class", title="Final eval confusion matrix", xticks=range(7), yticks=range(7))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "figures/confusion_final.png", dpi=150)
        plt.close(fig)
    rows = [to_row(r, scores.get(r["cfg"]["exp_id"])) for r in results]
    write_xlsx(rows, str(root / "templates/experiment_table_template.xlsx"), str(out / "experiments.xlsx"))
    write_report(results, baseline, final, scores, out, student_name)
    return scores


def package_submission(out_dir, notebook_path=None):
    """ZIP nộp không chứa data/checkpoint/cache; notebook_path là bản có output."""
    out = Path(out_dir).resolve()
    if notebook_path:
        source = Path(notebook_path).resolve()
        destination = out / "code/lab.ipynb"
        if source != destination.resolve():
            shutil.copy2(source, destination)
    for required in ("REPORT.md", "experiments.xlsx", "predictions_eval.csv", "eval_result.json", "code/lab.ipynb"):
        if not (out / required).exists():
            raise FileNotFoundError(f"Thiếu file nộp: {required}")
    target = out.parent / f"{out.name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            relative = path.relative_to(out)
            if not path.is_file() or any(p in {"checkpoints", "__pycache__", ".ipynb_checkpoints", "_smoke"} for p in relative.parts):
                continue
            if path.suffix in {".pt", ".pth", ".ckpt", ".tmp", ".pyc"}:
                continue
            archive.write(path, Path(out.name) / relative)
    return str(target)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--notebook", default=None)
    args = parser.parse_args()
    print(package_submission(args.out, args.notebook))
