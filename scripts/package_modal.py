"""Build a source ZIP and a self-contained Modal notebook without bundling credentials."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "qwen3-1.7b-a10080-e02-b16-ga2-v2"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    paths = [ROOT / name for name in ("README.md", "VALIDATION.md", "pyproject.toml", ".gitignore")]
    for folder in ("src", "configs", "scripts", "tests"):
        paths += sorted(path for path in (ROOT / folder).rglob("*")
                        if path.is_file() and path.suffix in (".py", ".json")
                        and "__pycache__" not in path.parts)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.writestr(str(Path("continue-pretrainedllms") / path.relative_to(ROOT)), path.read_bytes())
    content = payload.getvalue()
    encoded = base64.b64encode(content).decode()
    checksum = hashlib.sha256(content).hexdigest()
    bootstrap = f'''# Code nguồn đóng gói trong notebook; không chứa token hoặc checkpoint.
import base64, hashlib, io, os, zipfile
from pathlib import Path
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {checksum!r}
work = Path.cwd() / "cpt_workspace"
work.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        target = (work / item.filename).resolve()
        if not target.is_relative_to(work.resolve()):
            raise ValueError("Unsafe archive path")
    archive.extractall(work)
os.chdir(work / "continue-pretrainedllms")
print("Project:", Path.cwd())
'''
    invoke = '''import subprocess
def modal_command(*arguments):
    subprocess.run(["modal", *arguments], check=True)
'''
    cells = [
        cell("markdown", "# Qwen3-1.7B CPT trên Modal A100 80GB\n\n"
             "Dùng notebook kernel **CPU**; launcher cấp riêng một A100 80GB. "
             "Tạo Secret `hf-token` chứa `HF_TOKEN` trong cùng Modal environment trước khi chạy. "
             "Full fine-tuning: 2 epoch, batch 16, accumulation 2, context 2048. "
             "Chỉ bắt đầu một job cho mỗi run ID.\n"),
        cell("code", bootstrap),
        cell("code", "%pip install 'modal>=1,<2'\n"),
        cell("markdown", "## Đăng nhập Modal CLI\n\nChạy một lần nếu CLI chưa được xác thực; "
             "mở liên kết được in ra và đăng nhập đúng workspace/environment chứa Secret.\n"),
        cell("code", invoke + '\nmodal_command("setup")\n'),
        cell("markdown", "## Kiểm tra CPU trước khi thuê GPU\n\nKhông tải corpus hoặc weights Qwen. "
             "Nếu cell lỗi, xử lý lỗi trước khi chạy tiếp.\n"),
        cell("code", 'modal_command("run", "scripts/modal_full_ft.py", "--check-only")\n'),
        cell("markdown", "## Chuẩn bị corpus trên CPU\n\nTải 253 shard từ `iraprp/chayqwen`, "
             "ghim model/data revision và lưu packed Arrow trên Volume. Có thể mất nhiều giờ ở lần đầu.\n"),
        cell("code", 'modal_command("run", "scripts/modal_full_ft.py", "--prepare-only")\n'),
        cell("markdown", "## Train hoặc resume một phiên\n\n"
             "Một phiên dừng có lưu checkpoint sau khoảng 18 giờ để chừa thời gian trong giới hạn 24 giờ. "
             "`--detach` giữ job remote nếu notebook mất kết nối; có thể dừng cell sau khi Modal xác nhận job đã khởi chạy. "
             "Chờ phiên trước kết thúc trước khi chạy lại. Giữ nguyên notebook/code/config để resume.\n"),
        cell("code", f'RUN_ID = {RUN_ID!r}\nmodal_command("run", "--detach", "scripts/modal_full_ft.py", "--run-id", RUN_ID)\n'),
        cell("markdown", "## Xem trạng thái và ETA\n\nCó thể chạy từ notebook khác hoặc sau khi phiên kết thúc. "
             "ETA lấy từ tốc độ đo được, gồm overhead xảy ra giữa các lần log.\n"),
        cell("code", f'RUN_ID = {RUN_ID!r}\nmodal_command("run", "scripts/modal_full_ft.py", "--status-only", "--run-id", RUN_ID)\n'),
        cell("markdown", "## Tải hồ sơ thí nghiệm\n\nGói này chứa log, thống kê, cấu hình và snapshot source; "
             "không chứa model weights. Tải `audit_bundle.zip` về máy qua Files panel.\n"),
        cell("code", 'modal_command("run", "scripts/modal_full_ft.py", "--export-only", "--run-id", RUN_ID)\n'
             'modal_command("volume", "get", "qwen3-1-7b-full-ft-runs", f"/{RUN_ID}/audit_bundle.zip", "./audit_bundle.zip")\n'),
    ]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }, "nbformat": 4, "nbformat_minor": 4}
    notebook_path = ROOT / "notebooks" / "modal_a10080.ipynb"
    notebook_path.parent.mkdir(exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    bundle_path = destination / "continue-pretrainedllms-modal-a10080.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [*paths, notebook_path]:
            archive.writestr(str(Path("continue-pretrainedllms") / path.relative_to(ROOT)), path.read_bytes())
    (destination / "SHA256SUMS.txt").write_text(
        f"{hashlib.sha256(bundle_path.read_bytes()).hexdigest()}  {bundle_path.name}\n"
        f"{hashlib.sha256(notebook_path.read_bytes()).hexdigest()}  {notebook_path.name}\n"
    )
    print(bundle_path)
    print(notebook_path)


if __name__ == "__main__":
    main()
