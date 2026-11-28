# PCA-SPE sensor-loss audit

Reproduce the full-response 1,024-point run on Z24 and HellBridge with
`python -m clmm_timesnet.pca_sensor_loss_1024 --z24 <input> --hell <h5> --out <folder>`.
Both bridges use healthy-only fit data and thirty nested whole-device mask orders.
Z24 retains the prior validation-selected score tail and exports the conventional
upper-tail sensitivity. HellBridge retains the upper tail.

Use `python -m clmm_timesnet.explain_pca_sensor_loss --report <folder-or-zip> --output <folder> --focus 50`
to reproduce the plotted decisions on saved scores. The analysis checks the AUC,
ranking credit, and alarm metrics against the benchmark before writing outputs.
It reports pairs that lose or gain ranking credit, an example mask selected by
median AUC change, alarm changes along the fixed clean-threshold then
recalibrated-threshold path, and changes by damage state. These diagnostics
describe score and decision changes on the fixed cohorts.

Run `python -m clmm_timesnet.pca_sensor_loss_visuals --report <folder-or-zip> --z24 <input> --out <folder> --focus 50`
to generate eight PNG and PDF figures. The Z24 native files are used for isolated
physical-location removals, waveform examples, and a comparison of shared and
independent masks with a common sensor pool and common pair denominator.
Sites 431 and 441 are retained in the mask-design comparison where available.
All raw-data checks reuse the benchmark's healthy fit and clean score direction.

Run the follow-up with
`python -m clmm_timesnet.pca_observed_sensor_comparison --z24 <input> --hell <h5> --out <folder>`.
All three methods use the original splits, 1,024-point windows, nested masks and
healthy training data. The two alternatives calculate residual energy per
observed dimension using a restricted full basis or an exact subset refit.
The full-basis healthy reference sets the shared score scale within each
measurement group. The original clean Z24 validation tail is fixed across
methods, and the upper-tail analysis is also exported. Thresholds are compared
under both the clean-frozen and mask-matched healthy validation policies.
Reports include paired mask-level deltas, AUC by damage state, and an AUC figure.

Run the Z24 representation comparison with
`python -m clmm_timesnet.pca_z24_spectral_comparison --z24 <input> --out <folder>`.
The original 1,024-point windows feed either the time-domain scorer or a
per-channel Welch spectrum with 32 log-power bands between 0.5 and 30 Hz.
The spectral features are centered per window and channel to retain spectral shape.
Both representations use the same observed-channel PCA-SPE calculation.
The report shows the paired AUC difference and how that difference changes from
the clean condition, along with a heatmap by damage state.

Run the Welch power ablation with
`python -m clmm_timesnet.pca_z24_welch_power_ablation --z24 <input> --out <folder>`.
It compares the original centered 32-band representation with absolute log-band
power and mean log power alone, with time-domain controls. The mean log power is
the exact offset removed by band centering. Both validation classes receive the
original mask schedule, and validation AUC is reported separately from the
exploratory test results. The PCA estimators still use healthy training data.

## Tiếng Việt

Chạy bộ kiểm tra PCA-SPE ở 1.024 điểm trên hai cầu bằng lệnh đầu, hoặc đưa ZIP
kết quả đã có vào lệnh thứ hai để vẽ lại. Hình xuất PNG và PDF kèm CSV.
Phân tích recall và F1 dùng ba điểm đối chiếu: đủ cảm biến với ngưỡng clean,
mất cảm biến với ngưỡng clean, rồi mất cảm biến với ngưỡng hiệu chỉnh.

Lệnh trực quan hóa xuất tám hình PNG và PDF. Phần Z24 kiểm tra bỏ từng vị trí,
so sánh hai cách tạo mask trên cùng nhóm cảm biến và minh họa dạng sóng cùng
phần đóng góp vào lỗi tái tạo của vị trí 431 và 441.

Lệnh phân tích tiếp theo đối chiếu cách điền 0 với phép chiếu chỉ dùng kênh còn đo
và PCA fit lại trên đúng nhóm kênh đó. Ba cách dùng chung split, cửa sổ 1.024
điểm, mask và dữ liệu healthy train. Kết quả gồm chênh lệch theo từng mask,
AUC theo trạng thái hư hỏng và biểu đồ AUC cho hai cầu.

Lệnh Z24 tiếp theo so sánh PCA có mask trên tín hiệu miền thời gian và đặc trưng
phổ Welch 32 dải từ 0,5 đến 30 Hz. Báo cáo ghi cả chênh lệch AUC theo từng mask
và mức thay đổi của chênh lệch so với đầu vào đủ sensor, kèm heatmap theo trạng thái.

Lệnh kiểm tra mức công suất so sánh ba cách biểu diễn từ cùng 32 dải Welch, gồm
hình dạng phổ, phổ log giữ mức công suất và mức công suất riêng. Cùng split và
mask được dùng cho các đối chứng miền thời gian. AUC trên validation và test được
báo cáo riêng.
