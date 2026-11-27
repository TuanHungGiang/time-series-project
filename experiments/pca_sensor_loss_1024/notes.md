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

## Tiếng Việt

Chạy bộ kiểm tra PCA-SPE ở 1.024 điểm trên hai cầu bằng lệnh đầu, hoặc đưa ZIP
kết quả đã có vào lệnh thứ hai để vẽ lại. Hình xuất PNG và PDF kèm CSV.
Phân tích recall và F1 dùng ba điểm đối chiếu: đủ cảm biến với ngưỡng clean,
mất cảm biến với ngưỡng clean, rồi mất cảm biến với ngưỡng hiệu chỉnh.
