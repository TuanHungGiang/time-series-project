# Analysis notebooks

[CLMM–TimesNet Kaggle](clmm_timesnet_kaggle.ipynb) runs the packaged two-bridge ablation. It writes the result bundle to Kaggle's working directory. The training and evaluation logic lives in the Python package.

[Targeted baseline benchmark](kaggle_targeted_checkpoints.py) runs one of ten seeds on Z24 and HellBridge from a single Kaggle cell. Set `SEED_INDEX` from 0 to 9. The first shard also fits deterministic PCA-SPE once. All shards use the same healthy-fit split, removal order, test cohort and metric definitions. Z24 selects the anomaly score tail on labeled validation, then calibrates its threshold on healthy validation. For HellBridge, healthy development and test windows share two source recordings.

Save each shard's `weights` directory and report ZIP before the session is discarded. Each model group saves an inference checkpoint immediately after fitting. Reusing the same report and weight directories lets the runner continue from fitted groups. The reports contain per-seed and per-sample scores, split and sensor manifests, input arrays, curve points, confusion counts, model geometry and PNG/PDF figures. Use `MODE = "merge"` and set `REPORT_DIRS` to the ten extracted report directories to build combined mean and sample SD figures.

Đặt `SEED_INDEX` từ 0 đến 9 để chạy từng seed trên hai cầu. Lưu thư mục `weights` và ZIP báo cáo trước khi kết thúc phiên. Bản tổng hợp 10 seed dùng `MODE = "merge"` với `REPORT_DIRS` trỏ đến các thư mục báo cáo đã giải nén.

[Targeted CLMM gain head](kaggle_targeted_gain_head.py) compares native TimesNet, the residual adapter and the gain head on the baseline sensor masks. The cell defaults to the first baseline seed. Set `SEED_INDICES = list(range(10))` to schedule all ten.

[CLMM trên mask của baseline](kaggle_targeted_gain_head.py) so sánh TimesNet gốc, khối hiệu chỉnh và đầu điểm mới trên cùng mask sensor. Cell mặc định chạy seed đầu tiên. Đặt `SEED_INDICES = list(range(10))` để chạy đủ 10 seed.
