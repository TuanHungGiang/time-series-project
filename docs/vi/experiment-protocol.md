# Giao thức thí nghiệm

[English](../en/experiment-protocol.md)

Bản đầy đủ là `clmm_time_frequency`. TimesNet học từ healthy đầy đủ và đóng băng khi huấn luyện adapter. Bảy biến thể lần lượt kiểm tra TimesNet gốc, chu kỳ cố định, residual trên kênh còn đo, điền kênh, dự đoán chéo, attention trên kênh quan sát và loss phổ.

Khi dự đoán chéo, chia sensor còn đo thành bốn nhóm. Lần lượt giấu từng nhóm và dự đoán từ các sensor còn lại, sau đó tính sai số trên số đo thật của nhóm vừa giấu. Bản đầy đủ thêm loss log phổ công suất với trọng số 0.20 khi học. Các biến thể dự đoán chéo dùng cùng residual miền thời gian để chấm điểm.

Z24 chấm theo median cấp record, HellBridge chấm theo cửa sổ. Mỗi biến thể dùng ngưỡng Q90 trên healthy validation clean và giữ cố định ở các mức mất sensor. Báo AUC, AP, BA, recall, FPR, precision, F1, TP và FP. Recall tại FPR ≤ 5% đọc trên ROC test, còn recall triển khai lấy ở ngưỡng validation. Một lượt gồm một seed và năm mask chung ở mỗi mức 10%, 30%, 50%.
