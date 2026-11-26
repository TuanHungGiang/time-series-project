# Analysis notebooks

[CLMM–TimesNet Kaggle](clmm_timesnet_kaggle.ipynb) runs the packaged two-bridge ablation. It writes the result bundle to Kaggle's working directory. The training and evaluation logic lives in the Python package.

[Targeted baseline benchmark](kaggle_targeted_checkpoints.py) runs one of ten seeds on Z24 and HellBridge from a single Kaggle cell. Set `SEED_INDEX` from 0 to 9. The first shard also fits deterministic PCA-SPE once. All shards use the same healthy-fit split, removal order, test cohort and metric definitions. Z24 selects the anomaly score tail on labeled validation, then calibrates its threshold on healthy validation. For HellBridge, healthy development and test windows share two source recordings.

Save each shard's `weights` directory and report ZIP before the session is discarded. Each model group saves an inference checkpoint immediately after fitting. Reusing the same report and weight directories lets the runner continue from fitted groups. The reports contain per-seed and per-sample scores, split and sensor manifests, input arrays, curve points, confusion counts, model geometry and PNG/PDF figures. Use `MODE = "merge"` and set `REPORT_DIRS` to the ten extracted report directories to build combined mean and sample SD figures.

Đặt `SEED_INDEX` từ 0 đến 9 để chạy từng seed trên hai cầu. Lưu thư mục `weights` và ZIP báo cáo trước khi kết thúc phiên. Bản tổng hợp 10 seed dùng `MODE = "merge"` với `REPORT_DIRS` trỏ đến các thư mục báo cáo đã giải nén.

[Targeted CLMM gain head](kaggle_targeted_gain_head.py) compares native TimesNet, the residual adapter and the gain head on the baseline sensor masks. The cell defaults to the first baseline seed. Set `SEED_INDICES = list(range(10))` to schedule all ten.

[CLMM trên mask của baseline](kaggle_targeted_gain_head.py) so sánh TimesNet gốc, khối hiệu chỉnh và đầu điểm mới trên cùng mask sensor. Cell mặc định chạy seed đầu tiên. Đặt `SEED_INDICES = list(range(10))` để chạy đủ 10 seed.

[CNN-AE 1D baseline](kaggle_cnn_ae_1d.py) evaluates a multichannel PyTorch port of the Keras convolutional autoencoder with the same targeted protocol. It saves the checkpoint, score exports, clean operating point and all sensor-loss conditions. Set `SEED_INDICES = list(range(10))` to run ten seeds. The notebook displays mean ± sample SD and exports both numeric and formatted tables.

[Baseline CNN-AE 1D](kaggle_cnn_ae_1d.py) dùng bản chuyển kiến trúc CNN-AE của Keras sang PyTorch nhiều kênh. Cell xuất checkpoint, điểm dự đoán và kết quả trên cùng các mask sensor của benchmark. Bảng hiện mean ± SD mẫu và lưu cả CSV số gốc lẫn CSV đã định dạng.

[Centered gain-head ablation](kaggle_centered_gain_head.py) compares the original gain head against a version that subtracts the healthy median for the same mask. The cell can reuse the original CLMM checkpoints and writes paired differences with the clean threshold frozen.

[Gain head trừ trung vị healthy](kaggle_centered_gain_head.py) so sánh hai đầu điểm trên cùng checkpoint và mask. Cell tái sử dụng checkpoint CLMM nếu có và lưu chênh lệch theo từng seed tại ngưỡng clean cố định.
