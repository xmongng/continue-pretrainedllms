# Trạng thái kiểm tra bản bàn giao

Đã chạy tại workspace bằng Python 3.11:

- Ruff: `ruff check src scripts tests` — đạt.
- Python AST và JSON config parsing — đạt.
- Unittest: 6 bài stdlib đạt (packing, step budget, progress/ETA, system log/CSV,
  provenance/source hashes, và audit ZIP integrity).
- 3 bài ML được skip vì workspace chưa có torch/transformers/datasets/accelerate.
  Thử cài vào virtualenv tạm không thành công vì không phân giải được PyPI.
- Chưa chạy Modal image build, chưa thuê GPU, chưa đo VRAM/throughput hoặc thời gian
  hoàn thành. Không có kết quả benchmark A100 trong bản bàn giao.

Chạy bước bắt buộc trước khi train:

```bash
modal run scripts/modal_full_ft.py --check-only
```

Lệnh này chạy CPU trong image được ghim package. Nó kiểm tra thêm loss/gradient
chunked so với phép tính tham chiếu, phát hiện gradient không hữu hạn, và tính nhất
quán của checkpoint/resume so với train liên tục bằng model tiny local. Nếu có
bài bị skip hoặc lỗi, lệnh trả exit code khác 0. Không tải corpus hoặc weights Qwen.

Các thay đổi ảnh hưởng training loop:

- Cross entropy tính theo chunk 512 token và recompute intermediate khi backward,
  nhằm tránh full-batch shifted/softmax FP32 intermediate. Full model logits vẫn
  được tạo; chưa có số đo peak VRAM hoặc tốc độ thực tế.
- Gradient finiteness được gom kết quả trước khi đọc scalar về host để giảm số
  lần đồng bộ CPU/GPU. FP16 vẫn để GradScaler xử lý overflow như trước.
- Thêm audit/timing/host-GPU sampling; các số đo host wall không thay thế CUDA profiler.

Chạy lại check sau khi chỉnh code. Đổi code/config khi resume bị contract từ chối;
dùng run ID mới nếu đổi thí nghiệm.
