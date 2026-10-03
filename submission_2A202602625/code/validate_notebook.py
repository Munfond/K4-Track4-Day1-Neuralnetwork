"""Chạy các cell trong cùng kernel IPython CPU và lưu output kiểm tra smoke.

    python validate_notebook.py

Không chạy thí nghiệm toàn bộ CPU. Runtime cloud dùng notebook trực tiếp.
"""
import os
from pathlib import Path
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

HERE = Path(__file__).resolve().parent


def main():
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    shell = InteractiveShell.instance()
    shell.enable_gui = lambda gui: None
    shell.run_line_magic("matplotlib", "inline")
    n = nbformat.read(HERE / "lab.ipynb", as_version=4)
    for index, cell in enumerate(n.cells):
        if cell.cell_type != "code":
            continue
        print(f"Running cell {index}", flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
        cell.outputs = outputs
        cell.execution_count = shell.execution_count - 1
        print(captured.stdout[-3000:], flush=True)
        if result.error_in_exec or result.error_before_exec:
            nbformat.write(n, HERE / "validation_smoke.ipynb")
            raise RuntimeError(str(result.error_in_exec or result.error_before_exec))
    nbformat.write(n, HERE / "validation_smoke.ipynb")
    print("CPU smoke notebook complete", flush=True)


if __name__ == "__main__":
    main()
