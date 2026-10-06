# Dataset contract

Nguồn dữ liệu là ZIP, thư mục, JSON/JSONL hoặc JSON/JSONL gzip. Các file phải có
trường text đã clean/dedup; tên trường cấu hình bằng `data.text_column`.
Giữ nguyên whitespace của nội dung. JSON wrapper có thể chỉ định qua `json_field`.

Split train/validation/test theo group trước tokenization. Khai báo `group_column`
để các document cùng repository/source luôn cùng split. Nếu null, group là hash
nguyên văn text. Khai báo `source_column` để thống kê tỷ lệ token theo nguồn;
nếu null, nguồn là tên shard. Không chạy clean/dedup lại.

Manifest, hash recipe, thống kê và packed Arrow dataset được lưu một lần trong
`data/cache/prepared/<id>/`. Tất cả experiment cùng recipe dùng chung artifact.
Chi tiết packing, token accounting và giới hạn bộ nhớ tại [report.md](../report.md).
