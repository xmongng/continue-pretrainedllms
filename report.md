# Báo cáo pipeline continued pretraining

## 1. Mục tiêu và phạm vi

Tiếp tục pretrain `Qwen/Qwen3-1.7B-Base` trên corpus đã clean/dedup bằng
causal next-token loss; cập nhật toàn bộ trọng số. Code dùng AutoModelForCausalLM,
có thể đổi decoder-only model tương thích Transformers. Chưa tuyên bố mọi kiến trúc
đều đã được kiểm thử: tokenizer phải có EOS và là fast tokenizer.

Runner hiện được kiểm thử cho **một thiết bị**, CPU để test và CUDA để huấn luyện.
Nếu máy có nhiều GPU, chọn một bằng `CUDA_VISIBLE_DEVICES=0`. DDP, FSDP, DeepSpeed
chưa nằm trong phạm vi đã xác minh; runner từ chối thay vì cho chạy với accounting sai.
Full fine-tune giữ master weights và Adam states FP32; BF16/FP16 dùng autocast.
Điều này cần nhiều VRAM hơn inference hoặc LoRA. Chưa đo VRAM thực tế của Qwen3.

## 2. Trình tự thực hiện

1. Đọc config và kiểm tra giá trị, giải quyết đường dẫn theo thư mục project.
2. Tìm JSON/JSONL hoặc giải nén ZIP, kiểm tra cấu trúc file.
3. Hash nội dung file và tokenizer để xác định phiên bản dataset artifact.
4. Chia train/validation/test trước tokenize, lưu manifest từng document.
5. Tokenize, thêm EOS, thống kê và pack; lưu Arrow artifact dùng chung cho ba run.
6. Đo baseline validation của base model trước optimizer update đầu tiên.
7. Train, log metrics, evaluate và lưu checkpoint theo step và cuối epoch.
8. Nạp checkpoint có validation loss thấp nhất để đánh giá cuối run.
9. So sánh các run bằng validation; chỉ chạy test cho model đã chọn.

## 3. Vai trò từng file

| File | Trách nhiệm |
|---|---|
| `config.py` | Load JSON, kế thừa config chung, validate đường dẫn/tỷ lệ/precision |
| `data/dataset.py` | Tìm file, giải nén ZIP an toàn; collator padding/masking |
| `data/artifacts.py` | Fingerprint, chia nhóm, manifest, tokenize, pack, thống kê, cache |
| `training/monitoring.py` | Loss theo token, metrics gọn, callbacks logging/checkpoint |
| `training/train.py` | Model/tokenizer, baseline, train, kiểm tra resume, summary/failure |
| `evaluation/evaluate.py` | Đánh giá checkpoint đã chọn trên validation/test |
| `training/compare.py` | In bảng CSV so sánh các run ra terminal |
| `tests/test_cpt.py` | Regression tests dữ liệu, padding/EOS, weighted loss, save/resume |

`prepare.py` cũ đã được loại bỏ vì clean/dedup là bước upstream đã hoàn tất.

## 4. Dữ liệu và chia split

Mặc định `data.source = data/raw/dataset.zip`, trường nội dung `text`.
ZIP có thể chứa JSON arrays hoặc JSONL trong thư mục con. Với JSON wrapper như
`{"data": [...]}`, đặt `json_field: "data"`. Schema thực tế của bạn chưa được kiểm tra.
Các JSON có schema khác nhau được đọc theo từng file; file metadata cùng đuôi JSON
không được tự động phân biệt với corpus, vì vậy archive chỉ nên chứa các shard dữ liệu.

Tỷ lệ mặc định 98% / 1% / 1%, seed 42. Thuật toán xếp hạng group bằng SHA-256
với seed và chọn số group cho mỗi split; ít nhất một group mỗi split.
Nếu `group_column` là tên cột repository/source ID, mọi document cùng group nằm
trong một split. **Nên đặt group_column cho corpus code có repository ID.**
Nếu chưa có schema, để null: group là hash nội dung text, giúp exact duplicates
cùng split nhưng không phát hiện near duplicates hoặc quan hệ cùng repository.
Không strip text nên giữ nguyên indentation của code.

Tỷ lệ document/token thực tế có thể khác 98/1/1 khi group có kích thước khác nhau;
được ghi rõ trong stats. Không cân bằng lại âm thầm. Dữ liệu dưới ba group bị từ chối.

Artifact được định danh bằng hash nội dung file + settings chia split/context +
tokenizer; các run 1/2/3 epochs tái sử dụng cùng artifact. Thay corpus hoặc tokenizer
sẽ tạo artifact mới. Hash mọi file cần đọc toàn corpus; lần chạy sau vẫn xác minh
nội dung trước dùng cache, ưu tiên tính đúng đắn hơn bỏ qua kiểm tra.

## 5. Tokenization, packing và loss

Không dùng chat template; mỗi document được tokenize nguyên văn rồi thêm một EOS.
Pack liên tục trong từng split, có một token ngữ cảnh chồng giữa hai block.
Với stream N token, tổng target token hợp lệ đúng N−1. Token đầu tiên không có
ngữ cảnh để dự đoán; mọi token còn lại là target đúng một lần. Block cuối ngắn
được giữ; không bỏ token dư ở mỗi batch map như phiên bản trước.

EOS là dấu phân cách, **không phải attention barrier**: model có thể nhìn các
document trước đó trong cùng block. Context không truyền qua block ngoài một token
chồng. Đây là lựa chọn packing được ghi nhận, không phải document-isolated attention.

Collator pad bên phải; labels của padding và vị trí đầu sequence là −100.
EOS thật vẫn được tính loss kể cả khi PAD dùng chung token ID với EOS.
Custom loss dịch logits/labels một lần. Loss train được chuẩn hóa theo tổng target
token của accumulation window. Evaluation cộng NLL và target count của tất cả
sequence rồi chia; perplexity = exp(mean NLL). Không trung bình ngang các sequence
có độ dài khác nhau. Chỉ lưu hai số mỗi sequence trong evaluation, không giữ logits.

## 6. Ba thí nghiệm 1/2/3 epochs

Config chung dùng để kế thừa: `configs/qwen3_1_7b_base_common.json`.

| Config chạy | Ngân sách | Output |
|---|---:|---|
| `qwen3_1_7b_base_e01.json` | 1 epoch | `outputs/qwen3-1.7b-base/cpt-e01-seed42/` |
| `qwen3_1_7b_base_e02.json` | 2 epochs | `outputs/qwen3-1.7b-base/cpt-e02-seed42/` |
| `qwen3_1_7b_base_e03.json` | 3 epochs | `outputs/qwen3-1.7b-base/cpt-e03-seed42/` |

Cả ba bắt đầu từ cùng base model, cùng seed/dataset/context/batch/LR.
Mỗi run dùng cosine scheduler trên toàn ngân sách riêng: epoch 1 của run 3 epochs
không tương đương run 1 epoch độc lập vì lịch LR khác. Tổng chi phí cả ba là khoảng
6 epoch dữ liệu. Ba config là so sánh ngân sách, không khẳng định 3 epochs tốt hơn.
Nên pin `model.revision` thành commit trên Hugging Face trước chạy dài; `run.json`
ghi commit thực tế mà model đã resolve và từ chối resume nếu thay đổi.

Context ban đầu 1024, batch 1, accumulation 16, LR 2e-5, warmup 3%, cosine,
gradient checkpointing. Đây là điểm khởi đầu để đo tài nguyên và chất lượng,
không phải hyperparameter tối ưu đã xác nhận. Thử 2048/4096 bằng experiment mới
sau khi xác nhận VRAM. Không suy ra số epoch tối ưu từ kích thước GB.

## 7. Những file được lưu và lý do

```text
data/cache/prepared/<artifact-id>/
  identity.json          # source hashes, tokenizer hash, recipe
  split_manifest.jsonl   # file index + row + text hash + group/source + split + tokens
  dataset_stats.json     # documents/groups/tokens/length distribution/source shares
  dataset/               # packed Arrow train/validation/test, dùng lại giữa các run

outputs/qwen3-1.7b-base/cpt-e01-seed42/
  run.json               # resolved config, versions, code hashes, hardware, status/error
  baseline.json          # base-model validation loss/perplexity
  metrics.jsonl          # live training/evaluation history
  summary.json           # best checkpoint, validation, token budget consumed, completion
  checkpoint-<step>/     # tối đa 2: best và latest (có thể trùng)
    model.safetensors    # hoặc nhiều shard + index do Transformers quản lý
    config.json          # kiến trúc model
    tokenizer* ...       # tokenizer/special token configuration
    optimizer.pt         # Adam state
    scheduler.pt         # LR scheduler
    scaler.pt            # chỉ khi FP16 có gradient scaler
    rng_state.pth        # Python/NumPy/Torch/device RNG
    trainer_state.json   # global step, epoch, best metric, Trainer history
    training_args.bin    # Trainer configuration
    accounting.json      # token/elapsed và mốc byte metrics để resume chính xác
    COMPLETE             # marker ghi cuối cùng sau checkpoint thành công
```

Không tạo thêm `best/` hay `final/` sao chép weights. `summary.json` trỏ tới best
checkpoint có thể dùng trực tiếp với AutoModelForCausalLM.from_pretrained.
Optimizer và tokenizer được giữ cùng checkpoint để resume không phụ thuộc bản export.
Các file Trainer quản lý có chút metadata trùng metrics JSONL nhưng phục vụ restore;
không xóa chúng để tiết kiệm vài KB và làm mất chức năng resume.

`data/cache` còn có ZIP đã giải nén và Arrow loader cache để tránh xử lý nguồn lại;
đó là cache vận hành, không phải thêm bộ kết quả. Scratch tokenize/packing trong
artifact được xóa khi build thành công. Có thể dọn cache nguồn thủ công sau thí nghiệm,
nhưng giữ artifact đang được các run tham chiếu.

## 8. Theo dõi, lỗi và resume

`metrics.jsonl`: step, epoch, train loss, validation loss/perplexity/target tokens,
learning rate, grad norm, cumulative target_tokens_seen, elapsed và throughput theo
wall clock (bao gồm overhead evaluate/checkpoint). GPU allocated/peak bytes có khi
dùng CUDA. CPU không có GPU metrics. Trainer tổng kết runtime theo session;
không dùng `training_session_metrics.train_loss` để so sánh sau resume vì Trainer
không khôi phục running loss accumulator. Dùng per-step logs và validation weighted loss.

Checkpoint định kỳ mỗi 100 optimizer steps, thêm checkpoint ở cuối epoch và cuối
ngân sách. Lưu tối đa hai checkpoint: best theo validation và latest. Nếu lỗi xảy ra
giữa hai checkpoint, công việc sau checkpoint gần nhất phải chạy lại; chưa có
emergency checkpoint được bảo đảm trong trường hợp SIGKILL/OOM.

`run.json` ghi traceback cho lỗi training/evaluation và lỗi khởi tạo từ CLI. Nếu chỉ
khởi tạo thất bại (ví dụ sai `text_column`), có thể sửa config rồi chạy lại cùng output;
lịch sử các lần khởi tạo lỗi vẫn nằm trong `initialization_failures` của `run.json`.
NaN/Inf loss làm dừng sớm. Các lỗi hệ thống cắt process ngay có thể không ghi traceback;
run còn status running và checkpoint có COMPLETE gần nhất vẫn là điểm khôi phục.

Resume tự động chỉ chọn checkpoint có COMPLETE. Kiểm tra optimizer/scheduler/RNG/
scaler nếu dùng FP16, config, artifact, model revision/local weights, package versions,
device và code hash. Đổi epochs/LR/data giữa chừng bị từ chối để tránh âm thầm đổi
lịch scheduler. Để thử ngân sách khác, dùng config e01/e02/e03 với run riêng.
Metrics sau checkpoint được cắt đúng mốc byte đã lưu khi checkpoint hoàn tất, kể cả
khi log sau đó có cùng số step; do đó không đếm lặp công việc mất sau sự cố.
Baseline giữ nguyên, không đo lại với checkpoint đã train.
Run đã có trạng thái `completed` bị từ chối resume để tránh sửa lịch sử hoàn tất.

## 9. Lệnh chạy

Từ thư mục `Code-AI/llm-training`, Python 3.10+:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
llm-train --config configs/smoke.json

# Đặt ZIP tại data/raw/dataset.zip và sửa text_column/group_column nếu cần.
CUDA_VISIBLE_DEVICES=0 llm-train --config configs/qwen3_1_7b_base_e01.json
CUDA_VISIBLE_DEVICES=0 llm-train --config configs/qwen3_1_7b_base_e02.json
CUDA_VISIBLE_DEVICES=0 llm-train --config configs/qwen3_1_7b_base_e03.json

llm-train --config configs/qwen3_1_7b_base_e02.json --resume-from-checkpoint
llm-compare --root outputs/qwen3-1.7b-base

# Ví dụ SAU KHI chọn e02 bằng validation, chỉ khi đó đánh giá test:
llm-evaluate --config configs/qwen3_1_7b_base_e02.json --split test
```

Test không chạy tự động trong training hoặc baseline, không dùng để chọn epochs.
Lệnh test tạo `test_metrics.json` và từ chối ghi đè nếu đã có. Khi đánh giá lại
validation bằng CLI mới có `validation_metrics.json`; train thông thường đã lưu
validation cuối trong summary nên không cần chạy lệnh này nếu không có nhu cầu.

## 10. Giới hạn và kiểm chứng

Đã có regression tests offline bằng model GPT-2 rất nhỏ tạo tại local: group isolation,
manifest/cache reuse, bảo toàn target token khi pack, PAD=EOS mask, weighted loss,
checkpoint/resume so sánh trọng số với run không ngắt, và từ chối config đổi khi resume.
Các test archive/config kiểm tra ZIP traversal và cấu hình không hợp lệ.

Chưa chạy Qwen3 1.7B trên dữ liệu thực của bạn; ZIP thật chưa có trong workspace.
Không suy diễn chất lượng Qwen3 từ loss model toy. Hardware thực tế cần được đo trước
chạy dài. Resume được kiểm thử trên CPU; CUDA vẫn có sai khác số học phụ thuộc kernel.

Loader dùng Arrow trên đĩa, nhưng group index và length histogram ở RAM; không phải
pipeline streaming có RAM cố định cho hàng tỷ document. Tokenization hiện một tiến trình
để thống kê/manifest và packing nhất quán. Đây có thể là bottleneck trên corpus lớn;
artifact dùng lại nên không tokenize lại cho mỗi epoch/run.

Chưa có benchmark domain/general do chưa có task, harness và bộ câu hỏi cố định.
Loss/perplexity không đủ kết luận catastrophic forgetting hay năng lực domain; không
ghi score giả hoặc tạo file benchmark rỗng. Trước công bố model, cần bổ sung benchmark
thực và so sánh với baseline cùng prompt/decoding settings.

Tham khảo API: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
Model: https://huggingface.co/Qwen/Qwen3-1.7B-Base
