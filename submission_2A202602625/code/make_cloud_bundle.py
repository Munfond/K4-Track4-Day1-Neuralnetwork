"""Đóng gói input cloud từ code local đã cập nhật, không cần push GitHub."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def make_bundle():
    target = ROOT / "cloud_bundle_2A202602625.zip"
    paths = [ROOT / name for name in ("data/covtype.csv.gz", "data/split_metadata.csv", "data/README.md",
        "scripts/split_data.py", "scripts/evaluate.py", "templates/experiment_table_template.xlsx",
        "templates/REPORT_TEMPLATE.md", "README.md", "GUIDE.md", "RUBRIC.md", "CLOUD_RUN_GUIDE.md")]
    code = ROOT / "submission_2A202602625/code"
    paths += [p for p in code.iterdir() if p.is_file() and p.suffix in {".py", ".txt", ".ipynb"}
              and p.name != "validation_smoke.ipynb"]
    # Bằng chứng Part 1 local, notebook cloud sẽ ghi số đo mới khi chạy.
    for name in ("results/part1_health.json", "figures/part1_overfit20.png"):
        p = ROOT / "submission_2A202602625" / name
        if p.exists():
            paths.append(p)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    print(f"{target} ({target.stat().st_size / 2**20:.1f} MiB)")
    return target


if __name__ == "__main__":
    make_bundle()
