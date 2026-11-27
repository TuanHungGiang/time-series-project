# Matched sensor-loss comparison

Compare the post-TimesBlock-2 CLMM-Net with PCA-SPE, LSTM-AE, TimesNet, TranAD, ModernTCN, ScatterAD, and CNN-AE 1D on both bridges. The primary input is the 128-point antialiased version of the 1,024-sample window. Run the same experiment at 1,024 points as a separate resolution sensitivity analysis. Fit each neural model with seeds 20270905, 20270913, and 20270929. PCA-SPE is deterministic.

Run `python -m clmm_timesnet.matched_resolution_gap` with `--z24-root`, `--hell-h5`, `--baseline-dir`, and `--output-dir`. Supply the model sources from `research/mdr_shm_kaggle` at commit `09d81a4466668aa995d76446ae14c1d8e0c216bd`. Use `--points 128 1024` to run both resolutions from a fresh output directory. Outputs record the commits, dataset paths, manifest checksum, seeds, fitted groups and missing physical sensor IDs.

The split, detrending, resampling, fit-based standardization and test mask are shared across methods within each bridge. Baselines retain their model-specific objectives and scores. Z24 chooses score direction from labeled clean validation, with the same native TimesNet direction for the CLMM variants. The healthy validation threshold is either fixed at its complete-input value or recalibrated at the same loss level and mask replicate. Z24 draws masks per recording. HellBridge shares each mask across windows. HellBridge healthy development and test windows share source recordings.

- `paper_table.csv` reports mean ± sample SD after averaging masks within each seed.
- `paired_summary.csv` reports sensor loss against clean input, CLMM-Net against each baseline, and adjacent changes from TimesNet through adapter and score-head ablations.
- `mask_variability.csv` shows variation across masks within each training seed.
- The per-seed folders contain scores, threshold and confusion counts, within-group metrics, recall by damage state, and training and held-out healthy objectives. The adapter file reports training MSE on observed channels. The six-panel figures trace ranking and alarm metrics by sensor loss.

Use the paired seed deltas to check whether the reduction repeats across the three seeds. Compare recall with FPR at the validation threshold and use diagnostic recall to examine the test ROC at its specified FPR limit. Both summaries are conditional on the fixed test cohorts and selected masks.

## Tiếng Việt

Thí nghiệm đối chiếu CLMM-Net, bảy baseline và các bước ablation trên Z24 và HellBridge. Mặc định dùng 128 điểm sau lọc chống alias và giảm mẫu, với ba seed, cùng split và cùng mask lồng nhau ở 10%, 30% và 50%. Chạy thêm 1.024 điểm bằng tùy chọn `--points 128 1024` để kiểm tra ảnh hưởng của độ phân giải đầu vào.

Đọc `paired_summary.csv` để xác định metric nào giảm khi mất sensor, CLMM-Net bù được bao nhiêu, và khối nào tạo ra chênh lệch. Dùng `mask_variability.csv` để xem độ nhạy với từng mask. So recall cùng FPR ở ngưỡng validation. Recall chẩn đoán tại giới hạn FPR là chỉ số đọc từ ROC test. Độ lệch chuẩn qua ba seed mô tả mức biến động trên split và bộ mask cố định.
