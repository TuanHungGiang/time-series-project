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

[Original gain-head layer audit](kaggle_old_gain_diagnostics.py) reports the original score on four benchmark masks and logs layer separability, period selection and the adapter shift before and after LayerNorm.

[Chẩn đoán từng lớp của gain head gốc](kaggle_old_gain_diagnostics.py) in điểm phát hiện trên bốn mức mất sensor và ghi độ tách healthy/damage, thay đổi bin FFT và độ dịch qua adapter và LayerNorm.

[Laplacian gain scoring trial](../../src/clmm_timesnet/laplacian_kaggle.py) runs seed 20270901 on Z24 and HellBridge. It evaluates the original CLMM, adapter reconstruction, clipped gain, gain variance and graph Laplacian on matched cumulative masks near 30% and 50% loss, with a clean reconstruction reference. Run the script with `runpy.run_path(..., run_name="__main__")` from a Kaggle cell to display the result tables and ZIP link.

[Thử điểm gain Laplacian](../../src/clmm_timesnet/laplacian_kaggle.py) chạy seed 20270901 trên hai cầu. Script so sánh CLMM gốc, lỗi tái tạo sau hiệu chỉnh, gain chặn trần, phương sai gain và Laplacian trên cùng mask mất gần 30% và 50%, với điểm tái tạo clean làm mốc. Gọi script bằng `runpy.run_path(..., run_name="__main__")` trong Kaggle để hiện bảng và link ZIP.

[Conditional gain scoring trial](../../src/clmm_timesnet/conditional_gain_kaggle.py) compares the reconstruction score after the adapter with marginal and conditional gain scores on the same cumulative masks for both bridges. The runner reuses compatible CLMM checkpoints and exports paired deltas, recall at fixed FPR limits, and results at healthy validation thresholds.

[Thử đầu điểm gain có điều kiện](../../src/clmm_timesnet/conditional_gain_kaggle.py) so sánh điểm tái tạo sau adapter, điểm cộng gain và điểm cộng phần dư gain trên cùng mask cho hai cầu. Runner dùng lại checkpoint CLMM tương thích và xuất chênh lệch ghép cặp, recall tại cùng giới hạn FPR và kết quả tại ngưỡng healthy trên validation.

[Threshold audit across sensor masks](../../src/clmm_timesnet/threshold_masks_kaggle.py) compares frozen clean Q90 and mask-specific Q90 with the same adapted reconstruction scores at clean and matched 30/50% sensor removal. It reuses compatible CLMM checkpoints, exports all mask results and five-mask descriptive summaries, and traces the high-scoring healthy Z24 validation record.

[Kiểm tra ngưỡng trên nhiều mask](../../src/clmm_timesnet/threshold_masks_kaggle.py) so sánh Q90 clean cố định và Q90 theo mask trên cùng điểm tái tạo ở clean và mất khoảng 30/50% sensor. Runner dùng lại checkpoint CLMM tương thích, xuất kết quả từng mask, thống kê mô tả qua năm mask và điểm của bản ghi healthy Z24 đứng đầu ở lần thử trước.

[Z24 01-FVT reconstruction audit](../../src/clmm_timesnet/z24_01_fvt_audit_kaggle.py) traces the high-scoring healthy validation record with the previous targeted masks and a shared checkpoint. It exports per-window and per-sensor errors, matched healthy score scaling, and paired changes from sensor selection, masked inputs, and the adapter.

[Kiểm tra lỗi tái tạo Z24 01-FVT](../../src/clmm_timesnet/z24_01_fvt_audit_kaggle.py) truy vết bản ghi healthy validation có điểm cao với cùng checkpoint và các mask theo thứ tự cũ. Runner lưu lỗi theo cửa sổ và sensor, hệ số chuẩn hóa healthy và chênh lệch do ba bước chọn sensor, che đầu vào và bật adapter.

Run the threshold audit with `--per-sensor` to compare both score heads from the same model predictions on the two bridges. The pilot uses seed 20270901, clean input, the previous targeted mask, and five nested random masks near 30% and 50% sensor loss. The ZIP contains paired metrics, per-sensor healthy references and the Z24 01-FVT score trace.

Chạy kiểm tra ngưỡng với `--per-sensor` để so sánh hai đầu điểm từ cùng các dự đoán tái tạo trên hai cầu. Thử một seed 20270901 ở clean, mask theo thứ tự cũ và năm mask ngẫu nhiên lồng nhau ở mức mất khoảng 30% và 50% sensor. ZIP chứa chênh lệch ghép cặp, mốc healthy theo sensor và log điểm của bản ghi Z24 01-FVT.

[HellBridge spectral scoring pilot](../../src/clmm_timesnet/hellbridge_spectral_kaggle.py) uses one 1024-point TimesNet checkpoint and adapter across all score heads. It contrasts time reconstruction, original gain, observed spectra, and a joint time/spectrum score with clean and exact-mask thresholds at nominal 5% and 10%. The report includes false-positive rates, low-FPR recall, and alarm rates for each damage state.

[Thử đầu điểm phổ trên HellBridge](../../src/clmm_timesnet/hellbridge_spectral_kaggle.py) dùng chung TimesNet 1.024 điểm và adapter cho các đầu điểm. Runner đối chiếu lỗi tái tạo, gain cũ, điểm phổ quan sát và đầu điểm kết hợp, với ngưỡng clean và ngưỡng theo mask ở mức danh nghĩa 5% và 10%. Báo cáo có FPR, recall ở FPR thấp và tỷ lệ cảnh báo cho từng trạng thái hư hỏng.

[Keep-gain HellBridge pilot](../../src/clmm_timesnet/hellbridge_keep_gain_kaggle.py) adds a spectral correction to the original CLMM gain head only when sensors are missing. It reuses compatible native-resolution spectral scores when available and compares both heads on identical windows and masks.

[Thử CLMM giữ gain cũ trên HellBridge](../../src/clmm_timesnet/hellbridge_keep_gain_kaggle.py) thêm điểm phổ vào đầu gain gốc khi mất sensor. Runner dùng lại các điểm của lần thử phổ 1.024 điểm nếu có và so sánh các đầu điểm trên cùng cửa sổ và mask.

[HellBridge low-FPR spectral audit](../../src/clmm_timesnet/hellbridge_low_fpr_kaggle.py) compares two bounded spectral adjustments with the original and unbounded gain heads. It uses the existing 1024-point scores when available and exports low-FPR recall, partial AUC, and changed alarms by damage state and healthy window.

[Thử giới hạn điểm phổ ở vùng FPR thấp](../../src/clmm_timesnet/hellbridge_low_fpr_kaggle.py) so sánh hai cách giới hạn phần phổ với đầu gain gốc và đầu cộng phổ trước đó. Runner dùng lại điểm 1.024 nếu có và xuất recall tại FPR thấp, partial AUC và các thay đổi cảnh báo theo trạng thái hư hỏng và cửa sổ healthy.

Run the spectral pilot with `--segment-consensus` to replace the Welch-averaged spectral discrepancy with a per-sensor median of three overlapping 512-point discrepancies. The report compares both spectral statistics with the same CLMM weights, masks, healthy calibration and bounded signed adjustment.

Chạy thử nghiệm phổ với `--segment-consensus` để lấy trung vị độ lệch phổ của ba đoạn 512 điểm chồng lấn trên từng sensor. Báo cáo so sánh với sai lệch phổ tính sau khi lấy trung bình Welch, trên cùng trọng số CLMM, mask, mốc healthy và công thức giới hạn điểm.

### HellBridge cross-sensor residuals

Run `python -m clmm_timesnet.hellbridge_spectral_kaggle --residual-whitening --baseline-reference experiments/baseline_reference.csv` with `src` on `PYTHONPATH`. The pilot uses seed 20270905 with native 1024-point inputs. `baseline_comparison.csv` includes the current paired heads and historical rows. Each row gives the protocol, masking design, seed count, SD axis, and source. The historical random table was reconstructed from `outputs/paper_sensor_loss_figures/baseline_plot_data.csv`; targeted checkpoints use the listed locally saved summaries, and the ten-seed CNN-AE rows use the displayed table in `notebook73c51fdeb9 (24).ipynb`. Blank fields indicate unavailable metrics.

Chạy chế độ `--residual-whitening` cùng bảng `--baseline-reference` để đối chiếu trên HellBridge. Bảng ghi rõ giao thức, kiểu mất sensor, số seed và nguồn. Dùng `paired_deltas.csv` để so sánh theo đúng từng mask giữa các đầu điểm chạy cùng lượt. Dùng `baseline_comparison.csv` để xem các baseline lịch sử và số liệu cùng lượt chạy.

### One-seed whitening check on both bridges

Run `python -m clmm_timesnet.whitening_two_bridges_kaggle --baseline-reference experiments/baseline_reference.csv` on Kaggle with both inputs attached and `src` on `PYTHONPATH`. The runner uses the same HellBridge score and hyperparameters on Z24 with its existing train/validation/test split. The shared Z24 score direction comes from clean validation. The paired summary uses diagnostic recall at 0% FPR on Z24, where a single false alarm among six healthy test records exceeds 5%, and 5% FPR on HellBridge. Each bridge also reports recall and realized FPR at its healthy-validation threshold. The original baseline reference appears in `baseline_comparison.csv` with the protocol for each row.

Chạy `whitening_two_bridges_kaggle` để kiểm tra một seed trên cả hai cầu. Dùng `paired_summary.csv` cho kết quả chạy cùng split và mask, cùng `baseline_comparison.csv` để đối chiếu các baseline cũ theo giao thức đã lưu.
