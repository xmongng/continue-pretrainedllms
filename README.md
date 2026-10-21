# Continued pretraining Qwen3-1.7B trên Modal A100 80GB

Full fine-tuning toàn bộ parameter, dùng next-token prediction trên trường
`full_paper` của corpus psychology. Dữ liệu được tokenize và packing trước khi
train, rồi lưu Arrow trên Modal Volume. Không dùng LoRA hoặc quantization.

## 1. Cách nhanh nhất: mở notebook trên modal.com

Gói phát hành gồm:

- `notebooks/modal_a10080.ipynb`: notebook chứa sẵn toàn bộ source code của bản phát hành.
- `dist/continue-pretrainedllms-modal-a10080.zip`: code, configs, tests, README và notebook.

Mở [Modal Notebooks](https://modal.com/notebooks), upload
`modal_a10080.ipynb`, chọn **kernel CPU**. Launcher sẽ cấp riêng một A100 80GB
cho training; không cần cấp GPU cho notebook điều khiển.

Trong [Modal Secrets](https://modal.com/secrets), tạo Secret tên **`hf-token`**,
có key **`HF_TOKEN`** là Hugging Face token đủ quyền đọc dataset
`iraprp/chayqwen`. Secret phải nằm trong cùng Modal environment với job.
Không dán token vào code, notebook hoặc log.

Chạy các cell theo thứ tự. Cell đầu giải nén source đã đóng gói và kiểm tra SHA256;
không cần upload code ZIP riêng. CLI có thể yêu cầu `modal setup` để xác thực
workspace. Phần còn lại của README cung cấp từng lệnh tương ứng để copy chạy
trong terminal; trong notebook có thể thêm `!` trước các lệnh shell.

## 2. Nếu dùng ZIP và terminal

Giải nén ZIP, mở terminal tại thư mục chứa `pyproject.toml`:

```bash
cd continue-pretrainedllms
python -m pip install 'modal>=1,<2'
modal setup
```

Máy điều khiển chỉ cần Python 3.10+ và Modal CLI. PyTorch, Transformers và các
package train được cài trong image remote, không cần GPU trên máy điều khiển.

Kiểm tra pipeline trên **CPU của Modal**, trước khi thuê GPU:

```bash
modal run scripts/modal_full_ft.py --check-only
```

Bài kiểm tra tạo model/tokenizer tiny ngay tại chỗ, không tải weights Qwen hoặc
corpus. Kiểm tra packing giữ đủ token, tính số step, log/ETA, loss và gradient
so với cross entropy tham chiếu, phát hiện gradient không hữu hạn, và train →
checkpoint → resume so với train liên tục. Nếu lệnh lỗi, xử lý lỗi trước khi train.

## 3. Chuẩn bị corpus trên CPU

```bash
modal run scripts/modal_full_ft.py --prepare-only
```

Lệnh này:

1. Ghim commit dataset và model vào Volume để các phiên dùng cùng revision.
2. Tải đúng 253 shard `data/train-*.jsonl[.gz]` từ `iraprp/chayqwen`.
3. Kiểm tra text không rỗng, chia train/validation/test theo nhóm xác định bằng seed 42.
4. Tokenize không truncate, thêm EOS mỗi document; packing dài 2048 với một token
   overlap giữa các block để mỗi target token được học đúng một lần trong mỗi epoch.
5. Lưu packed Arrow, thống kê, source hashes và split manifest.

Lần đầu có thể mất nhiều giờ và nhiều dung lượng cache. Các lần sau tái sử dụng
artifact nếu recipe/data phù hợp. Training vẫn kiểm tra source hash và artifact
identity trước khi tiếp tục.

Hai Volume được tạo tự động:

| Volume | Mount trong function | Nội dung |
|---|---|---|
| `qwen3-1-7b-corpus` | `/data` | Corpus, model/tokenizer cache, packed Arrow, provenance |
| `qwen3-1-7b-full-ft-runs` | `/runs` | Checkpoint, metrics và hồ sơ thí nghiệm |

Validation/test hiện **được tách từ corpus**, mỗi bên **0,1% số nhóm**. Chưa có
đường dẫn đọc valid set riêng bên ngoài. Tỷ lệ document/token thực tế có thể khác
0,1%; xem `dataset_stats.json`. Không dùng test để chọn model.

`group_column=null`: text trùng nguyên văn thuộc cùng split, nhưng không loại bỏ
bản ghi trùng và không bảo đảm gom các phiên bản gần trùng của cùng bài báo.
Cần audit nhóm bài nếu dùng kết quả để báo cáo chất lượng khoa học.

## 4. Train hoặc resume trên một A100 80GB

```bash
modal run --detach scripts/modal_full_ft.py
```

Lệnh chọn đúng **`gpu="A100-80GB"`**. `--detach` giữ function remote chạy khi
terminal/notebook mất kết nối. Có thể xem log tại liên kết App trong output Modal.
Chỉ chạy **một job cho mỗi run ID**; không bấm lại khi phiên trước còn đang train.

Run mặc định:

```text
qwen3-1.7b-a10080-e02-b16-ga2-v2
```

Output:

```text
/runs/qwen3-1.7b-a10080-e02-b16-ga2-v2/
```

Mỗi phiên yêu cầu dừng sau khoảng **18 giờ** tính từ lúc vào function, tại ranh giới
optimizer step. Sau đó còn evaluation/save/commit; đây không phải cam kết thời gian
kết thúc chính xác. Function có timeout 24 giờ. Nếu timeout hoặc mất máy trước khi
save xong, chỉ checkpoint có marker `COMPLETE` mới được chọn để resume.

Khi phiên đã kết thúc với `status="stopped"`, chạy lại **đúng lệnh trên** để resume
checkpoint hoàn chỉnh mới nhất. Khi `status="completed"`, launcher trả summary,
không train lại. Không thay code/config/packages/data giữa các phiên của cùng run.

Nếu muốn giữ CLI hoạt động và tự nối nhiều phiên liên tiếp:

```bash
modal run scripts/modal_full_ft.py --sessions 10
```

Giữ terminal/notebook điều khiển hoạt động với cách này. Không dựa vào `--detach`
để tiếp tục vòng lặp nhiều phiên nếu tiến trình local đã chết. Tổng số phiên thực tế
phụ thuộc tốc độ đo được và các lần bị gián đoạn.

Chọn run mới khi đổi hyperparameter hoặc source code:

```bash
modal run --detach scripts/modal_full_ft.py --run-id experiment-02
```

Không dùng `--run-id` mới chỉ để resume: nó tạo một run khác từ base model.
Bản này dùng tên mặc định mới để tránh đụng checkpoint của các config trước đó.

## 5. Cấu hình đang sử dụng

File: `configs/qwen3_1_7b_modal_a100_full_ft.json`, kế thừa
`configs/qwen3_1_7b_base_common.json`.

| Tham số | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base`, commit được ghim khi prepare |
| GPU | 1 × A100 80GB |
| Epoch | 2 |
| Context / packed sequence length | 2048 |
| Train batch mỗi GPU | 16 sequences |
| Gradient accumulation | 2 |
| Batch hiệu dụng | 32 sequences / optimizer step |
| Target token ở một full optimizer step | 32 × 2047 = 65.504 |
| Precision | BF16 autocast, weights/gradient/Adam states FP32 |
| Attention | SDPA |
| Optimizer | Fused AdamW |
| Adam betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Peak learning rate | `2e-5` |
| LR schedule | Warmup tuyến tính 1%, rồi cosine về **0** |
| Weight decay / gradient clipping | `0.1` / `1.0` |
| Gradient checkpointing | Có, `use_reentrant=false` |
| Evaluation / save | Mỗi 1.000 optimizer steps; thêm tại kết thúc epoch/run |
| Checkpoint retention | Tối đa 2 theo Trainer, giữ best/latest khi khác nhau |
| Logging | Mỗi 10 optimizer steps, thêm log evaluation |
| GPU/host sampling | Khoảng mỗi 30 giây |
| DataLoader workers | 0; dữ liệu đã tokenize/packing |
| Reporting | JSONL/CSV trên Volume; chưa bật TensorBoard/W&B |

Core package trong image được ghim: Python 3.11, torch 2.8.0,
transformers 4.57.6, datasets 4.4.1, accelerate 1.10.1, huggingface_hub 0.36.0.
Version toàn bộ package thực tế nằm trong `packages.txt` của từng phiên. Các package
phụ thuộc gián tiếp chưa được khóa hoàn toàn; resume có kiểm tra core package versions.

Không đặt `max_steps` cho run chính. Trainer tính số optimizer steps từ số packed
sequence thực tế, batch và accumulation. `training_plan.json` ghi dự kiến;
`trainer_plan.json` ghi tổng step thực tế do Trainer xác định khi bắt đầu train.
Với khoảng 2,6 tỷ token trước split, kỳ vọng gần 79.400 steps cho 2 epoch;
con số chính xác phải lấy từ artifact, không lấy từ ước lượng corpus.

Batch 16/context 2048 chưa được benchmark VRAM trên A100 80GB trong workspace này.
Loss tính theo chunk để giảm intermediate FP32; full output logits vẫn được tạo.
Gradient checkpointing và chunked loss có chi phí tính lại. Không có cam kết hoàn
thành trong 1 ngày hoặc cam kết batch này chắc chắn vừa GPU. Theo dõi VRAM ở các
step đầu; nếu cần đổi batch, chọn run mới và kiểm tra lại batch hiệu dụng.

## 6. Theo dõi trạng thái, loss, throughput và ETA

Lệnh đọc trạng thái dùng CPU, không cấp GPU train mới:

```bash
modal run scripts/modal_full_ft.py --status-only
```

Với tên run riêng:

```bash
modal run scripts/modal_full_ft.py --status-only --run-id experiment-02
```

Các record chính:

- `loss`, `learning_rate`, `grad_norm`: log của Trainer khi có dữ liệu tương ứng.
- `eval_loss`, `eval_perplexity`: token-weighted validation NLL và `exp(NLL)`.
- `target_tokens_seen`: số target token đã đưa qua training, không gồm padding
  hoặc token context đầu block; không phải số token duy nhất của corpus.
- `target_tokens_per_second_wall`: throughput tích lũy; có overhead trong thời gian
  train. Baseline trước train và khởi tạo không nằm trong timer train.
- `interval_target_tokens_per_second`, `eta_seconds`: tốc độ/ETA từ khoảng giữa
  các lần tiến độ optimizer tăng; có thể dao động khi interval chứa eval/save.
- `gpu_allocated_bytes`, `gpu_peak_allocated_bytes`: VRAM allocator của PyTorch.
- `system_metrics.jsonl`: GPU utilization, memory utilization, VRAM, power,
  temperature qua `nvidia-smi`; load CPU, `/proc/self/status`, I/O và bộ nhớ host.

Nếu `nvidia-smi` không chạy được, record ghi lỗi sampling; không tự suy diễn GPU
utilization bằng 0. Peak VRAM được reset ở đầu mỗi phiên train.

`phase_totals` tách thời gian host lấy batch, training step, evaluation, ghi checkpoint
và commit Volume. `batch_fetch_host` gồm lấy batch, chuyển device qua Accelerate và
đếm target; **không phải** thời gian đọc đĩa thuần. `training_step_host` là host wall
time của forward/backward; CUDA bất đồng bộ nên không phải GPU kernel timing chính
xác, và không bao gồm toàn bộ optimizer/gradient check. Các phase không cộng thành
tổng runtime. Muốn quy kết bottleneck cụ thể vẫn cần profiler.

## 7. Hồ sơ thí nghiệm được lưu ở đâu?

```text
/runs/<run-id>/
├── run.json                         # contract config/data/model/code/packages, status/errors
├── baseline.json                    # validation của base model trước train
├── metrics.jsonl                    # lịch sử chính, truncate đúng checkpoint khi resume
├── summary.json                     # completed/stopped, best checkpoint, token/time/validation
├── test_metrics.json                # chỉ xuất hiện sau lệnh test riêng
├── checkpoint-<step>/               # weights, optimizer, scheduler, RNG, Trainer state
│   ├── accounting.json              # token/time/log byte boundary để resume
│   └── COMPLETE                     # được ghi sau khi checkpoint đã lưu
└── audit/
    ├── metrics.csv                  # export từ metrics.jsonl khi phiên kết thúc
    ├── initialization_failures/     # lỗi trước khi dựng xong Trainer, nếu có
    └── sessions/<UTC>-<id>/          # một thư mục riêng cho từng phiên hoặc assessment
        ├── config.json              # config đã kế thừa/resolve, có pinned provenance
        ├── contract.json
        ├── environment.json         # Python/platform/CPU/GPU, không dump toàn bộ env
        ├── packages.txt
        ├── initialization_timings.json # train sessions: tokenizer/data/model/Trainer setup
        ├── model_runtime.json       # train sessions: dtype, parameters, CUDA/cuDNN, attention
        ├── dataset_stats.json       # document/group/token/sequence và phân bố nguồn
        ├── data_identity.json       # recipe, source hashes, tokenizer fingerprint
        ├── dataset_reference.json   # đường dẫn + SHA256 split manifest trên corpus Volume
        ├── training_plan.json       # tính từ artifact và config
        ├── trainer_plan.json        # train sessions: số step thực tế của Trainer
        ├── source/                  # snapshot Python/config/launcher cho training sessions
        ├── source_sha256.json
        ├── events.jsonl             # UTC start/end/failure, eval/save/commit durations
        ├── progress.jsonl           # loss/LR/throughput/ETA/phase totals theo phiên
        ├── system_metrics.jsonl
        └── timing_totals.json
```

`audit/sessions` giữ cả các phiên bị rollback; dùng `session_id` và `resumed_from`
để phân biệt công việc đã bị bỏ sau checkpoint. Không cộng token qua các session
một cách trực tiếp. `metrics.jsonl` là lịch sử chính đã căn lại khi resume.
CSV chỉ được làm mới khi kết thúc phiên/assessment; trong lúc train đọc JSONL.
UTC timestamp và schema version có trong record audit; seed không bảo đảm GPU
bitwise deterministic. Split manifest đầy đủ nằm trong corpus Volume, không nhân
bản file lớn vào từng session.

Log được đóng file sau mỗi lần ghi, commit cùng checkpoint và định kỳ khi log
nếu đã qua 5 phút. Modal còn có background commit. Hard kill vẫn có thể mất những
record chưa được persist; không bảo đảm chống mọi kiểu lỗi phần cứng.

## 8. Tải log và provenance về máy

Tạo ZIP hồ sơ, chạy trên CPU:

```bash
modal run scripts/modal_full_ft.py --export-only
```

Tải ZIP:

```bash
modal volume get qwen3-1-7b-full-ft-runs /qwen3-1.7b-a10080-e02-b16-ga2-v2/audit_bundle.zip ./audit_bundle.zip
```

ZIP gồm root metrics/summary và toàn bộ `audit/`, kèm SHA256 từng file trong
`bundle_manifest.json`. Không chứa model weights/optimizer state. Nếu export khi
job còn chạy, đây chỉ là snapshot các log đang có và dòng cuối có thể đang ghi.
Không chạy hai lệnh export đồng thời cho cùng run.

Để tải best checkpoint, lấy đường dẫn trong `summary.json`, bỏ tiền tố `/runs`
khi dùng `modal volume get`. Ví dụ lấy `checkpoint-1000` **chỉ khi summary chọn
đúng checkpoint này**:

```bash
modal volume get qwen3-1-7b-full-ft-runs /qwen3-1.7b-a10080-e02-b16-ga2-v2/checkpoint-1000 ./best-checkpoint
```

Weights tốt nhất nằm trực tiếp trong checkpoint được chọn; không có thư mục
`final/` hoặc bản sao weights thứ hai. Giữ checkpoint đầy đủ nếu còn cần resume.

## 9. Đánh giá test sau khi chọn model bằng validation

Chỉ chạy sau khi `status="completed"` và đã quyết định chọn run:

```bash
modal run scripts/modal_full_ft.py --test-only
```

Lệnh cấp A100 80GB để đánh giá best checkpoint trên test split, lưu
`test_metrics.json` và một audit session riêng. Không cập nhật weights. Mỗi run
chỉ ghi test assessment một lần; các lần sau bị từ chối để giữ kết quả cuối.
Sau test, chạy lại bước export để tải cả kết quả test.

## 10. Kiểm tra và đóng gói lại trên máy phát triển

Kiểm tra phần stdlib, các bài model test sẽ skip nếu chưa cài ML dependencies:

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

Cài đầy đủ để chạy cả tiny-model tests tại local:

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

Tạo lại ZIP và notebook **sau mỗi lần sửa source/config/README**:

```bash
python scripts/package_modal.py
```

Notebook chứa snapshot source tại thời điểm đóng gói; sửa file bên ngoài notebook
không tự thay payload cũ. SHA256 của các artifact phát hành nằm trong
`dist/SHA256SUMS.txt`. Chưa có GPU benchmark trong bản bàn giao này; chạy CPU
checks trước và kiểm tra log/VRAM đầu run trên Modal.

## Tài liệu tham chiếu

- [Modal Notebooks](https://modal.com/docs/guide/notebooks)
- [Chọn A100-80GB trên Modal](https://modal.com/docs/guide/gpu)
- [Modal run và detach](https://modal.com/docs/reference/cli/run)
- [Modal Volumes và commit/download](https://modal.com/docs/guide/volumes)
