# CLMM–TimesNet

[English](README.md) | **Tiếng Việt**

Dự án phát hiện hư hỏng cầu khi mất nguyên kênh cảm biến. Bản đang thử nghiệm dùng TimesNet đóng băng và adapter CLMM học từ dữ liệu healthy trên Z24 và HellBridge. Phép thử dùng một seed, dữ liệu đầy đủ và 5 mask ở mỗi mức mất ngẫu nhiên 10%, 30% và 50%.

Mở [notebook Kaggle một cell](notebooks/analysis/clmm_timesnet_kaggle.ipynb), gắn hai dataset, rồi bật GPU và Internet. Notebook clone GitHub, ghi lại commit và chạy package. Kết quả gồm metric từng mask, bảng tổng hợp, chênh lệch ghép cặp và kiểm tra độ lệch do batch. Lượt đầy đủ trên hai cầu đang chờ kết quả.

- [Mã nguồn model](src/clmm_timesnet/models/)
- [Cấu hình chạy](configs/experiments/clmm_timesnet.yaml)
- [Giao thức đánh giá](docs/vi/experiment-protocol.md)
- [Bố cục thư mục](docs/vi/repository-structure.md)
- [Vị trí lưu các thí nghiệm cũ và runner baseline của bài báo](docs/en/legacy-workflows.md)
