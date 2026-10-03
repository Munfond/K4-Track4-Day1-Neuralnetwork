# Chạy Lab Part 0–4 trên Colab hoặc Kaggle

Code đã hoàn thiện nằm trong `submission_2A202602625/code/`. Dùng **code này**,
không copy lại khung TODO từ thư mục `code/` ở gốc repo.

Hai file để bắt đầu:

- `cloud_bundle_2A202602625.zip`: code mới, dữ liệu gốc, metadata, scripts chấm và mẫu bảng.
- `submission_2A202602625/code/lab.ipynb`: notebook chạy Part 0–4.

Không cần GPU trên máy cá nhân. Notebook cloud dùng GPU của dịch vụ.
Không cần push GitHub để dùng gói ZIP này. Nếu sửa code sau đó, tạo lại gói bằng:

```powershell
.\.venv\Scripts\python.exe submission_2A202602625/code/make_cloud_bundle.py
```

## Google Colab

1. Mở [Colab](https://colab.research.google.com/), chọn **Upload notebook**, upload
   `submission_2A202602625/code/lab.ipynb`.
2. Chọn **Runtime → Change runtime type → GPU** (T4 nếu có). Kết nối runtime.
3. Ở ô Bootstrap, giữ `USE_DRIVE = True`. Điền `STUDENT_NAME`. Mặc định
   `MODE = "full"` trên cloud, `RUN_GROUPS = None` chạy cả bảy chủ đề.
4. Chạy ô Bootstrap. Khi hiện bộ chọn file, upload **một** file
   `cloud_bundle_2A202602625.zip`. Cho phép mount Google Drive khi Colab hỏi.
5. Chạy ô môi trường. Phải thấy `device=cuda` và tên GPU. Nếu không,
   bật GPU rồi restart runtime; chế độ full sẽ dừng để tránh chạy dài bằng CPU.
6. Chạy các ô từ Part 0 đến Part 4, hoặc chọn **Runtime → Run all**.
7. Artifact và checkpoint lưu tại:
   `/content/drive/MyDrive/K4_Track4_Day1/submission_2A202602625/`.
   Dữ liệu được giải nén/nạp ở `/content/k4_lab/`, rồi đưa lên GPU một lần.
8. Đọc `REPORT.md`, bổ sung đối chiếu dự đoán/cơ chế và kiểm tra hình. Tải notebook
   đang mở qua **File → Download → Download .ipynb** để giữ output cloud.
9. Ô cuối tạo ZIP artifact trong Drive. Thay `code/lab.ipynb` trong gói nộp bằng
   bản vừa tải có output, rồi đóng gói lại như mục “Trước khi nộp”.

Nếu runtime bị ngắt: mở lại notebook, chọn GPU, chạy Bootstrap, upload lại ZIP
nếu VM mới chưa có gói, mount **cùng Drive**, giữ cấu hình rồi Run all.
Lượt hoàn tất được tái dùng; lượt dở tiếp tục từ epoch đã lưu cuối cùng.
Các bước trong epoch đang chạy khi bị ngắt sẽ được chạy lại. Không cần huấn luyện
lại các thí nghiệm đã hoàn tất. Không đổi code/config trong lúc khôi phục.

Nếu không dùng Drive, `USE_DRIVE=False` ghi vào VM Colab; tải kết quả/checkpoint
về trước khi runtime bị xoá. GPU và thời lượng miễn phí phụ thuộc hạn mức/khả năng
cấp tài nguyên, không có ETA cố định. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Kaggle

1. Tạo một **Dataset riêng tư**, upload `cloud_bundle_2A202602625.zip`.
   Kaggle có thể tự giải nén ZIP; notebook hỗ trợ cả ZIP và folder đã giải nén.
2. Tạo Notebook và import `lab.ipynb`. **Add Input** dataset vừa tạo.
3. Trong **Settings / Session options**, chọn **Accelerator → GPU**.
   Internet chỉ cần nếu thiếu thư viện phải cài thêm; gói đã chứa dữ liệu/code.
4. Điền `STUDENT_NAME`, giữ `MODE="full"` và `RUN_GROUPS=None` ở ô Bootstrap.
   Notebook tự tìm gói trong `/kaggle/input/`. Nếu không tìm được, đặt
   `BUNDLE_PATH` bằng đường dẫn ZIP hoặc folder gốc chứa `data/`, `scripts/`,
   `templates/`, `submission_2A202602625/` trong panel Input.
5. Chạy Bootstrap và ô môi trường, xác nhận `device=cuda`.
6. Có thể chạy thử `MODE="smoke"` trước; sau đó đổi lại `full`.
   Dùng **Save Version → Save & Run All** cho lượt chạy toàn bộ.
   Đây là phiên mới chạy từ đầu, không kế thừa biến/file của phiên interactive.
7. Artifact full nằm tại:
   `/kaggle/working/k4_lab/submission_2A202602625/`.
   ZIP nộp nằm tại `/kaggle/working/k4_lab/submission_2A202602625.zip`.
8. Sau phiên hoàn tất, tải ZIP qua Outputs và tải/export notebook có output.
   Thay notebook trong gói nộp bằng bản có output trước khi gửi bài.

Checkpoint trong `/kaggle/working` chỉ khôi phục nếu file còn tồn tại hoặc bạn
đã lưu/tải chúng. Muốn tiếp tục ở phiên mới: lưu kết quả phiên trước thành Dataset
(bao gồm `checkpoints/`, `results/`, `figures/`), Add Input dataset đó rồi **thêm một
ô sau Bootstrap, trước Part 2** để copy cache:

```python
from pathlib import Path
import shutil

# Sửa đường dẫn theo Input thực tế; folder này chứa checkpoints/, results/, figures/.
previous = Path("/kaggle/input/my-previous-run/submission_2A202602625")
for name in ("checkpoints", "results", "figures"):
    if (previous / name).exists():
        shutil.copytree(previous / name, Path(OUT_DIR) / name, dirs_exist_ok=True)
# Nếu lần trước đã chốt/đánh giá eval, khôi phục các manifest và artifact đó luôn.
for path in previous.glob("*.json"):
    shutil.copy2(path, Path(OUT_DIR) / path.name)
for name in ("predictions_eval.csv", "baseline_predictions_eval.csv"):
    if (previous / name).exists():
        shutil.copy2(previous / name, Path(OUT_DIR) / name)
```

Giữ cùng code, MODE, seed, RUN_GROUPS và hyperparameter. Khi signature khác,
pipeline sẽ báo lỗi thay vì âm thầm dùng checkpoint khác thí nghiệm.
Nếu đã xem eval, dùng lại selection cũ; không chọn cấu hình mới theo điểm eval.
Hướng dẫn UI và Save & Run All: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

## Notebook sẽ chạy những gì?

| Phần | Công việc |
|---|---|
| Part 0 | Chia train/eval đúng metadata; val 20% seed 42; chỉ fit chuẩn hoá bằng train |
| Part 1 | Kiểm tra 47 879 tham số/shape, loss bước 0, quá khớp 20 mẫu, gradient |
| Part 2 | Tune SGDM lr 0.01/0.03/0.1 bằng val trong 3 epoch; baseline 20 epoch × 3 seed |
| Part 3 | CE/MSE; SGD/SGDM/Adam/AdamW nhiều lr; M-wide; dropout 0.3; clipping thường/stress; FP16/BF16; zeros/normal/Xavier |
| Part 4 | Chốt model bằng val; best-val-loss checkpoint; chấm baseline/final bằng evaluate.py; bảng, ảnh, báo cáo, ZIP |

Mặc định có 25 lượt cấu hình: 3 tuning ngắn, 3 baseline và 19 thí nghiệm.
Các lượt không hỗ trợ phần cứng (ví dụ BF16 trên T4) ghi `skipped`, lý do và ảnh.
Tổng số lượt chạy GPU thực tế có thể ít hơn. Không chạy nhiều thí nghiệm song song
trên cùng GPU để số đo thời gian/bộ nhớ có thể so sánh.

Để giảm thời gian, đặt `RUN_GROUPS` một lần trước khi chạy, ví dụ:

```python
RUN_GROUPS = ["optimizer", "dropout", "amp", "init"]
```

Baseline vẫn đầy đủ. Bỏ chủ đề sẽ giảm độ phủ theo rubric. Sửa RUN_GROUPS sau khi
đã chốt final sẽ không được dùng để chọn lại. Smoke dùng 1 024 train, 512 val,
2 epoch mỗi lượt (tuning 1 epoch), ghi riêng `_smoke/`; không chấm eval chính thức.

Metric summary lấy tại epoch có **val loss nhỏ nhất**, không lấy epoch F1 cao nhất.
Sau đó chọn final bằng F1 tại các snapshot này, seed 1 cố định. Không dùng dữ liệu
eval để chọn lr/model/epoch. Train-loss là eval-mode trên 50k mẫu cố định seed 42;
val dùng toàn bộ. CSV eval luôn dự đoán đủ 116 203 mẫu.

FP16 dùng autocast và GradScaler, unscale trước đo/cắt gradient. BF16 cần GPU hỗ
trợ; baseline vẫn FP32. MSE tính trên logits và one-hot, mean trên B×7, không hệ số
1/2; không so trị số loss MSE trực tiếp với CE.
[PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

## Trước khi nộp

1. Tải và giải nén ZIP artifact; đọc `REPORT.md`, bổ sung họ tên và nhận xét cụ thể
   về kết quả khớp/khác dự đoán. Code tạo bản báo cáo dựa trên số đo; cần bạn rà lại
   cách diễn giải, không suy luận chắc chắn nguyên nhân chỉ từ một seed.
2. Mở `experiments.xlsx` bằng Excel/LibreOffice để tính lại công thức. Giữ nguyên
   Legend, Experiments, Seeds, Summary. Có 1 PNG riêng cho mỗi dòng thí nghiệm;
   các ảnh `compare_*`, `confusion_final` và Part 1 là ảnh bổ sung.
3. Thay `submission_2A202602625/code/lab.ipynb` bằng notebook cloud vừa tải có output.
   Kiểm tra đã thực hiện Run All thành công trên cloud. Bản `validation_smoke.ipynb`
   là bằng chứng kiểm tra CPU, không thay cho notebook full nộp bài.
4. Tạo ZIP nộp sạch trên máy local (dùng `.venv` đã có hoặc Python có requirements):

   ```powershell
   .\.venv\Scripts\python.exe submission_2A202602625/code/reporting.py --notebook "C:/path/to/executed_lab.ipynb"
   ```

   Chạy từ gốc repo, sau khi đã copy artifact cloud vào `submission_2A202602625/`.
   CLI không train, không chấm lại eval. ZIP loại checkpoint, dữ liệu và `_smoke/`.
5. Cần có `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`,
   `figures/`, `code/` và notebook có output. Điểm bảng/báo cáo phải khớp JSON script gốc.
   Không nộp cloud_bundle (nó chứa dữ liệu), dùng ZIP `submission_2A202602625.zip`.

## Tình trạng kiểm thử local

Toàn bộ notebook smoke đã được chạy trên CPU, và pipeline có kiểm thử metric
CE/MSE, clipping, kiến trúc/He, Adam/AdamW wd=0, resume, cache, divergence,
AMP không hỗ trợ, mở rộng bảng và CSV qua script chấm chính thức.
Chưa chạy 20 epoch toàn bộ dữ liệu trên GPU tại máy local; chưa có điểm final
để cam kết. Cần chạy full trên cloud để có kết quả nộp và số đo AMP thực tế.
