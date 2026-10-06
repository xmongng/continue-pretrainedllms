# continue-pretrainedllms

Full fine-tuning cho Qwen3 1.7B Base, với split cố định, baseline, token accounting
và checkpoint có thể resume. Xem [report.md](report.md) để hiểu từng module,
định nghĩa metrics, artifact được lưu và giới hạn đã kiểm thử.

## Bắt đầu

Dùng Python 3.10+ từ thư mục này:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
llm-train --config configs/smoke.json
```

Đặt corpus đã clean/dedup tại `data/raw/dataset.zip`, hoặc sửa `data.source`.
Đặt `text_column`, `json_field`, `group_column` và `source_column` theo schema thật
trong `configs/qwen3_1_7b_base_common.json`. Với code, group theo repository nếu có ID.

## Thí nghiệm

```bash
CUDA_VISIBLE_DEVICES=0 llm-train --config configs/qwen3_1_7b_base_e01.json
CUDA_VISIBLE_DEVICES=0 llm-train --config configs/qwen3_1_7b_base_e02.json
CUDA_VISIBLE_DEVICES=0 llm-train --config configs/qwen3_1_7b_base_e03.json
llm-compare --root outputs/qwen3-1.7b-base
```

Ba run độc lập cùng base model và artifact dữ liệu, tổng khoảng 6 epoch compute.
Mỗi run có scheduler riêng. Không dùng checkpoint của e01 để resume thành e02.

```bash
llm-train --config configs/qwen3_1_7b_base_e02.json --resume-from-checkpoint
```

Sau khi chọn run bằng validation, mới đánh giá test; ví dụ chọn e02:

```bash
llm-evaluate --config configs/qwen3_1_7b_base_e02.json --split test
```

Best checkpoint được ghi trong `summary.json`; load trực tiếp bằng Transformers.
Không tạo thêm bản sao weights trong `final/`. Không ghi đè run đã có:
dùng checkpoint để resume hoặc chọn output directory mới.

## Kiểm tra

```bash
make check
```

Runner được kiểm thử trên một thiết bị. DDP/FSDP/DeepSpeed chưa hỗ trợ.
Các bài test dùng model tiny local, không tải weights Qwen3 hoặc cần GPU.
