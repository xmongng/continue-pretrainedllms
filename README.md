# Qwen3-1.7B — chạy trực tiếp trên Modal Notebook

Mở **https://modal.com/notebooks**, tạo notebook và chọn **1 GPU A100 80GB**.
Trong sidebar, gắn Volume lưu code/cache/log tại **`/mnt/qwen3-cpt`**.
Gắn Volume chứa dữ liệu của bạn vào notebook nếu đó là Volume khác.
Secret `HF_TOKEN` chỉ cần nếu bạn dùng model cần đăng nhập Hugging Face.

Copy mỗi block dưới đây vào **một code cell**, chạy từ Cell 1 đến Cell 5.
Không cần `modal setup`, `modal run`, hoặc một notebook CPU điều khiển GPU khác.
Nếu import `notebooks/modal_a10080.ipynb` của bản này, các cell đã được tạo sẵn.

## Cell 1 — Lấy code vào Volume

Cell này tải code từ GitHub theo commit cố định để các lần resume dùng cùng
phiên bản. Khi restart notebook, code trên Volume được dùng lại. Nếu repository
private hoặc không tải được URL, dùng notebook chứa sẵn code; hoặc để
`CODE_URL = ""` và upload `continue-pretrainedllms-modal-a10080.zip` vào Files panel,
cùng thư mục hiện tại của notebook.

```python
from pathlib import Path
import io, tempfile, urllib.request, zipfile

ROOT = Path("/mnt/qwen3-cpt")
assert ROOT.is_dir(), "Gắn Volume qwen3-cpt tại /mnt/qwen3-cpt trước"
PROJECT = ROOT / "project-volume-data-v1"
CODE_URL = "https://github.com/xmongng/continue-pretrainedllms/archive/8d037ccd2cb1855bc42c98b25f31fc83e8c2100e.zip"

if not (PROJECT / "scripts/modal_notebook.py").is_file():
    payload = (
        urllib.request.urlopen(CODE_URL, timeout=120).read()
        if CODE_URL else Path("continue-pretrainedllms-modal-a10080.zip").read_bytes()
    )
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        work = Path(directory).resolve()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for item in archive.infolist():
                if not (work / item.filename).resolve().is_relative_to(work):
                    raise ValueError("Đường dẫn ZIP không hợp lệ")
            archive.extractall(work)
        candidates = list(work.glob("*/scripts/modal_notebook.py"))
        assert len(candidates) == 1, "Cần ZIP chứa scripts/modal_notebook.py của bản mới"
        candidates[0].parent.parent.rename(PROJECT)
print("Code:", PROJECT)
```

## Cell 2 — Cài dependencies

```python
%pip install torch==2.8.0 transformers==4.57.6 datasets==4.4.1 accelerate==1.10.1 huggingface_hub==0.36.0 safetensors filelock hf_xet
```

## Cell 3 — Chỉ định folder dữ liệu đã có trên Volume

**Chỉ sửa `DATA_PATH` thành đường dẫn thư mục chứa các shard của bạn.**
Không tải lại hoặc sao chép dataset từ Hugging Face. Các file `.jsonl.gz`/`.jsonl`
phải có trường văn bản `full_text`. Metadata nguồn có thể nằm lồng trong record và
không được dùng làm nội dung huấn luyện. Code tokenize/packing rồi lưu cache;
nếu artifact khớp thì dùng lại. Model revision, file hashes và split được ghi
trong hồ sơ thí nghiệm. Validation/test tự tách từ corpus, mỗi bên 0,1% số nhóm;
chưa dùng valid set riêng bên ngoài.

```python
import subprocess, sys

DATA_PATH = "/mnt/qwen3-cpt/data"  # THAY bằng folder dữ liệu trên Volume của bạn
RUN_ID = "qwen3-1.7b-notebook-a10080-e02-b16-ga2-v1"
SCRIPT = PROJECT / "scripts/modal_notebook.py"
OUTPUT = ROOT / "runs" / RUN_ID

def run(*arguments):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--volume", str(ROOT), "--run-id", RUN_ID, *arguments],
        cwd=PROJECT, check=True,
    )

run("--prepare", "--data-path", DATA_PATH)
```

Dữ liệu giữ nguyên tại `DATA_PATH`. Cache ở `ROOT / "cache"` và `ROOT / "hf-cache"`.
Tokenizer/model có thể vẫn cần tải từ Hugging Face nếu chưa có cache;
cell chuẩn bị không tải weights model. CPU làm tokenize/packing; nếu notebook đã cấp
GPU thì tài nguyên GPU vẫn tính phí trong thời gian chuẩn bị.

## Cell 4 — Chạy check

Kiểm tra packing, số step, loss/gradient, checkpoint/resume và audit bằng model
nhỏ local. Không bắt đầu training nếu cell này báo lỗi hoặc test bị skip.

```python
run("--check")
```

## Cell 5 — Training / resume trên A100 80GB

**2 epoch · context 2048 · batch 16 · accumulation 2 · BF16 · LR 2e-5 ·
warmup 1% · cosine về 0 · weight decay 0.1 · Adam betas (0.9, 0.95).**

```python
run("--train")
```

Giữ kernel chạy. Function train nằm ngay trong notebook này; không có GPU job
thứ hai hoặc `--detach`. Mỗi phiên yêu cầu dừng tại ranh giới optimizer step sau
khoảng 18 giờ, rồi evaluate/save. Chạy lại Cell 5 khi phiên trước đã kết thúc để
resume checkpoint hoàn chỉnh mới nhất. Giữ nguyên code/config, `DATA_PATH`, `RUN_ID` và Volume.
Nếu restart kernel, chạy lại Cell 1–4 trước Cell 5. Không chạy hai notebook train
cùng `RUN_ID`. Chưa benchmark batch 16/VRAM trên A100 80GB trong bản bàn giao.

## Cell 6 — Xem kết quả sau phiên training

```python
import json

print("Output:", OUTPUT)
for name in ("run.json", "summary.json"):
    path = OUTPUT / name
    if path.exists():
        value = json.loads(path.read_text())
        if name == "run.json":
            value = {k: value.get(k) for k in ("status", "resumed_from", "audit_session_id", "error")}
        print(name, json.dumps(value, indent=2, ensure_ascii=False))
```

Trong lúc Cell 5 đang chạy, xem log trực tiếp ở output của cell. Log bền vững nằm
ở `OUTPUT / "metrics.jsonl"` và `OUTPUT / "audit/sessions"`: loss, LR, gradient
norm, token throughput/ETA, VRAM/GPU utilization, thời gian lấy batch/train/eval/save,
config, source hashes, package versions và thống kê dữ liệu. Thời gian host chỉ là
chỉ báo; không thay thế GPU profiler. `metrics.csv` được xuất khi kết thúc phiên.

## Cell 7 — Đóng gói log để tải về

```python
run("--export")
print("Tải file này trong Files panel:", OUTPUT / "audit_bundle.zip")
```

ZIP chứa log/cấu hình/provenance/source snapshot và checksum; không chứa model
weights. Best checkpoint nằm ở đường dẫn `best_checkpoint` trong `summary.json`.
Dữ liệu và kết quả vẫn nằm trên Volume sau khi tắt kernel.

## Cell 8 — Test cuối cùng (tùy chọn)

Chỉ chạy khi run đã `completed` và đã chọn model bằng validation. Chỉ ghi test
assessment một lần. Sau đó chạy lại Cell 7 để tải cả kết quả test.

```python
run("--test")
```

**Trạng thái kiểm thử:** xem `VALIDATION.md`. Chưa chạy thử GPU; Cell 4 là bước
kiểm tra model nhỏ trước training. Exact-text duplicates được gom cùng split;
chưa bảo đảm chống rò rỉ giữa các phiên bản gần trùng của cùng bài báo.
