"""Build a source ZIP and a self-contained Modal notebook without bundling credentials."""

import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    bootstrap = f'''# Source code is embedded; no GitHub download or Modal CLI is needed.
import base64, hashlib, io, tempfile, zipfile
from pathlib import Path
ROOT = Path("/mnt/qwen3-cpt")
assert ROOT.is_dir(), "Attach Volume qwen3-cpt at /mnt/qwen3-cpt first"
PROJECT = ROOT / "project-volume-data-v1"
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {checksum!r}
if not (PROJECT / "scripts/modal_notebook.py").is_file():
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        work = Path(directory).resolve()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for item in archive.infolist():
                if not (work / item.filename).resolve().is_relative_to(work):
                    raise ValueError("Unsafe archive path")
            archive.extractall(work)
        (work / "continue-pretrainedllms").rename(PROJECT)
print("Project:", PROJECT)
'''
    readme = (ROOT / "README.md").read_text()
    sections = re.findall(r"(## Cell [^\n]+)\n(.*?)(?=\n## Cell |\Z)", readme, re.S)
    if len(sections) != 8:
        raise ValueError("README must contain the eight direct Modal Notebook cells")
    cells = [cell("markdown", "# Qwen3-1.7B — trực tiếp trên Modal Notebook\n\n"
                  "Chọn 1 A100 80GB; gắn Volume lưu code/cache/log tại `/mnt/qwen3-cpt`. "
                  "Gắn Volume chứa dataset và điền `DATA_PATH` ở Cell 3. Không tải lại dataset. "
                  "Code đã được nhúng trong notebook; "
                  "không cần code ZIP, GitHub, `modal setup` hoặc `modal run`. "
                  "Chạy Cell 1–5 theo thứ tự. Giữ kernel chạy khi train.\n")]
    for index, (heading, section) in enumerate(sections):
        match = re.search(r"```python\n(.*?)```", section, re.S)
        if match is None:
            raise ValueError(f"Missing Python cell under {heading}")
        code = bootstrap if index == 0 else match.group(1)
        description = "Code nguồn được nhúng và giải nén vào Volume.\n" if index == 0 else section[:match.start()]
        cells.append(cell("markdown", heading + "\n\n" + description))
        cells.append(cell("code", code))
        tail = section[match.end():].strip()
        if tail:
            cells.append(cell("markdown", tail + "\n"))
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
