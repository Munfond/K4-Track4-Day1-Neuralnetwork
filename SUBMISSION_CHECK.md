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
- Kiểm tra lại: 25 ảnh đúng exp_id, 10 ảnh so sánh, 2 ảnh Part 1/confusion; 37 ảnh mở được từ GitHub public.
- Không có NotImplementedError trong module bài làm; 8 bài kiểm thử đều qua.
- Notebook chạy từ đầu đến cuối ở chế độ CPU smoke trong thư mục chỉ có submission, data và scripts, không cần templates ở gốc. Mẫu bảng đi kèm code, được dùng khi mẫu gốc không có.
- Output nộp vẫn là lượt full trên Tesla T4, 9 cell code đã chạy và không có lỗi. Kiểm tra local không phải lần chạy lại full trên GPU.
- GitHub xác nhận đây là fork public của VinUni-AI20k/K4-Track4-Day1-Neuralnetwork; giảng viên đọc được không cần đăng nhập.

Link cần gửi giảng viên: https://github.com/Munfond/K4-Track4-Day1-Neuralnetwork
Việc gửi link qua kênh giảng viên thông báo chưa được thực hiện trong phiên này. File tải về gốc vẫn ở Downloads.
