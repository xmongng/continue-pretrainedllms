# Experiments

Dùng configs e01/e02/e03 để so sánh ngân sách 1/2/3 epochs từ cùng base model.
Mỗi run giữ baseline, metrics JSONL, run contract, summary và tối đa hai full
checkpoints (best và latest). `llm-compare` in bảng CSV để xem kết quả.

Resume chỉ trong cùng run, cùng recipe, package versions và code.
Không sửa num_train_epochs để kéo dài lịch cosine đã hoàn thành.

Dùng validation để chọn checkpoint/run; test đánh giá cuối sau khi chốt lựa chọn.
Domain/general benchmark cần bộ đánh giá thực riêng; perplexity không chứng minh
năng lực downstream hoặc loại trừ catastrophic forgetting.

Xem [report.md](../report.md) để biết cấu trúc output và lệnh đầy đủ.
