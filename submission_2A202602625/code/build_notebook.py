"""Tạo notebook cloud từ phần Part 0/1 hiện có; chỉ chạy khi cập nhật code lab."""
from pathlib import Path
import nbformat

HERE = Path(__file__).resolve().parent


def build():
    old = nbformat.read(HERE / "lab.ipynb", as_version=4)
    # Nhận diện bằng heading thay vì index để builder chạy lặp lại được.
    p0 = next(i for i, c in enumerate(old.cells) if c.cell_type == "markdown" and c.source.startswith("## Part 0"))
    p1 = next(i for i, c in enumerate(old.cells) if c.cell_type == "markdown" and c.source.startswith("## Part 1"))
    part0 = old.cells[p0:p1]
    p1code = next(c.source for c in old.cells[p1 + 1:] if c.cell_type == "code")
    notebook = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook.cells = [md('''# Lab Day 1 — 2A202602625

Notebook hoàn thiện Part 0–4. Mở **bản này** và chọn GPU trên Colab/Kaggle.
Đọc `CLOUD_RUN_GUIDE.md` kèm gói. Mặc định cloud chạy `full`, máy local chạy `smoke`.
Full: baseline 20 epoch, 3 seed, menu 7 chủ đề, chọn bằng val, chấm eval cuối.
Smoke chỉ kiểm tra pipeline trên mẫu nhỏ, không dùng để nộp hoặc kết luận.

Đặt `RUN_GROUPS` ở ô cấu hình nếu muốn chạy ít chủ đề hơn; giữ cùng lựa chọn
qua các lần khôi phục. Cache/checkpoint sẽ tái dùng cùng code/cấu hình/dữ liệu.
Sau khi đã chốt `selection.json`, không thay cấu hình dựa vào điểm eval.
'''), code('''# ===== Bootstrap: upload gói local đã cập nhật, không cần git push =====
import os, sys, shutil, zipfile, subprocess, importlib.util
from pathlib import Path

BUNDLE_PATH = ""  # Kaggle: điền đường dẫn ZIP hoặc folder dataset nếu tự phát hiện không được
USE_DRIVE = True  # Colab: lưu artifact/checkpoint vào Drive để khôi phục sau mất runtime
RUN_GROUPS = None # None = cả 7; ví dụ ["optimizer", "dropout", "amp", "init"]
STUDENT_NAME = "" # điền họ tên trước khi tạo REPORT.md
IS_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
IS_KAGGLE = Path("/kaggle/working").exists()
IS_CLOUD = IS_COLAB or IS_KAGGLE
MODE = "full" if IS_CLOUD else "smoke" # có thể đổi thành smoke để kiểm tra nhanh trên cloud

workspace = Path("/content/k4_lab") if IS_COLAB else Path("/kaggle/working/k4_lab") if IS_KAGGLE else None
repo = None
if workspace and (workspace / "data/covtype.csv.gz").exists():
    repo = workspace
elif not IS_CLOUD:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data/covtype.csv.gz").exists():
            repo = candidate
            break
if repo is None:
    source = Path(BUNDLE_PATH) if BUNDLE_PATH else None
    if IS_COLAB and source is None:
        from google.colab import files
        uploaded = files.upload() # chọn cloud_bundle_2A202602625.zip
        names = [name for name in uploaded if name.endswith(".zip")]
        if len(names) != 1:
            raise ValueError("Hãy upload đúng một cloud_bundle ZIP")
        source = Path(names[0])
    elif IS_KAGGLE and source is None:
        zips = list(Path("/kaggle/input").rglob("cloud_bundle_2A202602625.zip"))
        if zips:
            source = zips[0]
        else:
            data_files = list(Path("/kaggle/input").rglob("covtype.csv.gz"))
            if data_files:
                source = data_files[0].parent.parent # Kaggle có thể đã giải nén dataset
    if source is None or not source.exists():
        raise FileNotFoundError("Không tìm thấy gói. Đặt BUNDLE_PATH tới ZIP/folder dataset đã upload")
    workspace.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                dest = (workspace / item.filename).resolve()
                if not dest.is_relative_to(workspace.resolve()):
                    raise ValueError("ZIP có đường dẫn không hợp lệ")
            archive.extractall(workspace)
    else:
        shutil.copytree(source, workspace, dirs_exist_ok=True)
    repo = workspace

REPO_ROOT = str(repo.resolve())
CODE_DIR = repo / "submission_2A202602625/code"
if not (CODE_DIR / "experiments.py").exists():
    raise FileNotFoundError("Gói chứa code cũ; hãy upload cloud_bundle mới được cung cấp")
if IS_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/K4_Track4_Day1/submission_2A202602625")
else:
    OUTPUT_ROOT = repo / "submission_2A202602625"
OUT_DIR = str(OUTPUT_ROOT / "_smoke" if MODE == "smoke" else OUTPUT_ROOT)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
if Path(OUT_DIR) / "code" != CODE_DIR:
    shutil.copytree(CODE_DIR, Path(OUT_DIR) / "code", dirs_exist_ok=True)
os.chdir(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
# Cloud có PyTorch CUDA sẵn: chỉ cài thư viện phụ đang thiếu, không thay torch.
missing = [pkg for module, pkg in [("numpy","numpy"), ("sklearn","scikit-learn"),
    ("pandas","pandas"), ("matplotlib","matplotlib"), ("openpyxl","openpyxl"),
    ("nbformat","nbformat")] if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
print("REPO_ROOT:", REPO_ROOT, "\\nOUT_DIR:", OUT_DIR, "\\nMODE:", MODE)
'''), code('''import json, time, numpy as np, torch
from data import prepare_data, iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, init_weights, activation_stats
from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval
from experiments import run_baselines, run_menu, choose_final
from results_table import load_results, to_row, write_xlsx, seed_statistics
from reporting import finish_submission, package_submission, describe_comparison

device = "cuda" if torch.cuda.is_available() else "cpu"
if MODE == "full" and device != "cuda":
    raise RuntimeError("Full cần GPU: bật accelerator rồi restart runtime; dùng MODE='smoke' để kiểm tra CPU")
if device == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 1))
print(f"PyTorch {torch.__version__}; device={device}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "BF16:", torch.cuda.is_bf16_supported())
for folder in ("figures", "results", "checkpoints"):
    Path(OUT_DIR, folder).mkdir(parents=True, exist_ok=True)
set_seed(42)
''')]
    notebook.cells += part0
    notebook.cells += [md('''## Part 1 — Model và kiểm tra sức khoẻ
M-base: 54→256→128→7, He cho mọi Linear, bias=0, 47 879 tham số.
Đo bước 0 trên toàn bộ val trước cập nhật; model riêng ghi nhớ 20 mẫu.
'''), code(p1code), code('''from IPython.display import display, Markdown
display(Markdown(f"**Nhận xét Part 1:** CE val bước 0 = {loss_step0:.6f}; ln7 = {reference:.6f}; "
    f"lệch {loss_step0-reference:+.6f}. Logits std={logit_std:.6f}; He sinh logits chưa đều nên CE "
    "không nhất thiết bằng ln7. Chuẩn hoá đã kiểm tra trên train; giữ đúng He, không ép loss về mốc. "
    f"20 mẫu sau 500 bước: loss={loss_history[-1]:.8f}, accuracy={acc_history[-1]:.0%}. "
    "Cả sáu gradient khác None/0. Phép thử này chưa chứng minh tổng quát hoá."))
'''), md('''## Part 2 — Pipeline, chọn lr baseline và nhiễu seed
Dự đoán trước: SGDM lr nhỏ có thể học chậm; lr lớn có thể xuống nhanh hoặc dao động.
Tune lr {0.01, 0.03, 0.1} trong 3 epoch bằng val, sau đó chạy baseline 20 epoch
với seed 1, 2, 3. Train loss đo trên 50k mẫu cố định ở eval, val toàn bộ.
Mỗi epoch lưu JSON, PNG và checkpoint; chạy lại sẽ tiếp tục từ epoch hoàn tất.
'''), code('''# Smoke tách riêng sau khi Part 0/1 đã kiểm tra dữ liệu đầy đủ.
if MODE == "smoke":
    data = {**data, "X_tr": data["X_tr"][:1024], "y_tr": data["y_tr"][:1024],
            "X_val": data["X_val"][:512], "y_val": data["y_val"][:512],
            "signature": data["signature"] + ":smoke1024-val512"}
base_cfg, baseline_results = run_baselines(data, OUT_DIR,
    epochs=20 if MODE == "full" else 2, tune_epochs=3 if MODE == "full" else 1)
stats = seed_statistics(baseline_results)
print(json.dumps(stats, indent=2))
base_result = next(r for r in baseline_results if r["cfg"]["exp_id"] == "base-s1")
print("Best epoch:", base_result["summary"]["best_epoch"])
print("Val-loss curve:", base_result["history"]["val_loss"])
print("Train-loss curve:", base_result["history"]["train_loss"])
display(Markdown("**Đối chiếu baseline:** " + describe_comparison(base_result, base_result, stats["noise_2sigma"]) +
    " Đọc hình compare_baseline để nhận xét còn giảm hay quá khớp; không dùng smoke để kết luận."))
display(__import__("IPython").display.Image(filename=str(Path(OUT_DIR)/"figures/compare_baseline.png")))
'''), md('''## Part 3 — Menu 7 chủ đề
Mỗi config chứa `prediction` được in **trước** lượt chạy và lưu cùng JSON.
CE/MSE chỉ so metric; optimizer thử nhiều lr; clip c dựa trên gradient baseline,
stress cặp cùng lr ×10; AMP chỉ chạy khi phần cứng phù hợp. Không dùng eval.
'''), code('''menu_results = run_menu(data, OUT_DIR, base_cfg, baseline_results, groups=RUN_GROUPS)
all_results = baseline_results + menu_results
for result in menu_results:
    text = describe_comparison(result, base_result, stats["noise_2sigma"])
    display(Markdown("**Đối chiếu sau chạy:** " + text))
    if result["summary"].get("status") == "complete":
        print("Activation std:", result["initial"]["activation_std"],
              "clip_fraction:", result["summary"]["clip_fraction"],
              "time/epoch:", result["summary"]["time_per_epoch_s"],
              "GPU MiB:", result["summary"]["peak_mem_MB"])
'''), md('''## Part 4 — Chốt bằng val, chấm eval, bảng và báo cáo
Chọn seed 1, F1 tốt nhất tại best-val-loss epoch trong các lượt hoàn tất; loại tuning/stress.
`selection.json` được ghi trước eval. Chấm bằng script gốc chỉ cho baseline và final.
Smoke tạo bảng/báo cáo kiểm tra, **không chấm eval** và không tạo ZIP nộp.
'''), code('''baseline, final = choose_final(all_results, OUT_DIR)
print("Final cfg:", final["cfg"])
scores = finish_submission(all_results, baseline, final, data, REPO_ROOT, OUT_DIR,
                           student_name=STUDENT_NAME, evaluate_eval=MODE == "full")
print("Đã ghi experiments.xlsx và REPORT.md vào", OUT_DIR)
display(Markdown(Path(OUT_DIR, "REPORT.md").read_text(encoding="utf-8")))
if MODE == "full":
    display(__import__("IPython").display.Image(filename=str(Path(OUT_DIR)/"figures/confusion_final.png")))
'''), md('''### Lưu notebook có output và đóng gói
1. Tải notebook đang mở **có output** qua menu File của Colab/Kaggle.
2. Thay bản vừa tải vào `<OUT_DIR>/code/lab.ipynb` (hoặc thay trên máy local sau khi tải ZIP).
3. Đọc REPORT.md, điền họ tên và bổ sung đối chiếu/cơ chế bằng các con số đã đo.
4. Chạy lại ô đóng gói bên dưới, hoặc trên máy local:
   `python submission_2A202602625/code/reporting.py --notebook /path/to/executed.ipynb`.

ZIP dưới đây là gói artifact; trước khi nộp phải thay notebook bằng bản có output cloud.
Checkpoint không được đưa vào ZIP. Trên Kaggle, Save & Run All chạy phiên mới;
đầu vào/checkpoint muốn khôi phục phải được gắn làm dataset (xem hướng dẫn).
'''), code('''if MODE == "full":
    submission_zip = package_submission(OUT_DIR)
    print("ZIP:", submission_zip)
    from IPython.display import FileLink
    display(FileLink(submission_zip))
    # Colab: bỏ comment để tải ZIP artifact
    # from google.colab import files
    # files.download(submission_zip)
else:
    print("Smoke đã hoàn tất; chuyển MODE='full' và bật GPU để chạy kết quả nộp.")
''')]
    notebook.metadata = dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                             language_info=dict(name="python"))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, HERE / "lab.ipynb")


if __name__ == "__main__":
    build()
