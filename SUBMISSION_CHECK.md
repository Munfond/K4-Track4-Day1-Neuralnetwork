# Kiểm tra bản nộp sau Colab

- Sinh viên: Nguyễn Đức Anh — 2A202602625.
- 25 lượt thí nghiệm hoàn tất; 9 cell code notebook có execution count, không có output lỗi.
- Code và output notebook Colab được giữ; bổ sung nhận xét bằng markdown.
- Evaluator gốc chấm lại hai CSV cho kết quả trùng hoàn toàn JSON tải về.
- Final chọn theo validation: hparam-wide, seed 1, best epoch 20.
- Eval final: macro-F1 0.8762841374; accuracy 0.9197869246.
- Workbook giữ 4 sheet; 25 dòng kết quả; chỉ baseline/final có điểm eval.
- Công thức được tính lại; mean/std seed đúng số đo, không có ô lỗi công thức.
- Report đã có nhận xét loss, optimizer, độ rộng, dropout, clipping, AMP, khởi tạo và lớp khó nhất.
- ZIP không chứa dữ liệu, checkpoint, cache hoặc kết quả smoke.

Không cần huấn luyện lại để nộp gói này. File tải về gốc vẫn ở Downloads.
