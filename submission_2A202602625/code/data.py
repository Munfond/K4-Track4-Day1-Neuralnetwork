"""Nạp, tách và chuẩn hoá dữ liệu CoverType cho các thí nghiệm.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval chỉ dùng để chấm điểm cuối. Không fit thống kê trên eval, chọn cấu hình
hay dừng sớm bằng eval; chỉ áp dụng phép chuẩn hoá đã fit từ train.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import hashlib
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_dir = Path(processed_dir)
    with np.load(processed_dir / "train.npz") as train, np.load(processed_dir / "eval.npz") as eval_set:
        X_train_full, y_train_full = train["X"], train["y"]
        X_eval, y_eval, eval_row_id = eval_set["X"], eval_set["y"], eval_set["row_id"]

    for X, y in ((X_train_full, y_train_full), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64
        assert np.all((0 <= y) & (y <= 6))
    assert eval_row_id.shape == (len(X_eval),) and eval_row_id.dtype == np.int64
    assert np.unique(eval_row_id).size == len(eval_row_id)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    numeric = X_tr[:, :N_NUMERIC]
    mean = numeric.mean(axis=0, dtype=np.float64)
    std = numeric.std(axis=0, dtype=np.float64)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_standardized = X.copy()
    safe_std = np.where(std == 0, 1.0, std)
    X_standardized[:, :N_NUMERIC] = (X_standardized[:, :N_NUMERIC] - mean) / safe_std
    return X_standardized


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    arrays = {
        "X_tr": apply_standardizer(X_tr, mean, std),
        "y_tr": y_tr,
        "X_val": apply_standardizer(X_val, mean, std),
        "y_val": y_val,
        "X_eval": apply_standardizer(X_eval, mean, std),
        "y_eval": y_eval,
    }
    data = {
        name: torch.as_tensor(array, device=device) for name, array in arrays.items()
    }
    data["eval_row_id"] = eval_row_id
    digest = hashlib.sha256()
    for array in (arrays["X_tr"], arrays["y_tr"], arrays["X_val"], arrays["y_val"]):
        digest.update(np.ascontiguousarray(array).tobytes())
    data["signature"] = digest.hexdigest()

    for split in ("tr", "val", "eval"):
        X, y = data[f"X_{split}"], data[f"y_{split}"]
        print(f"{split}: X {tuple(X.shape)} {X.dtype}, y {tuple(y.shape)} {y.dtype}")
    majority_class = int(np.bincount(y_tr, minlength=7).argmax())
    majority_acc = float(np.mean(y_val == majority_class))
    print(f"Lớp đa số từ train: {majority_class}; accuracy trên val: {majority_acc:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")
    indices = (torch.randperm(len(X), generator=generator, device=X.device)
               if shuffle else torch.arange(len(X), device=X.device))
    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]
