# TSP GA Studio (Bản mô tả đầy đủ)

Tài liệu này mô tả chi tiết cách sử dụng, cấu trúc dự án, các chế độ thống kê và cách đọc biểu đồ trong ứng dụng giải TSP bằng Genetic Algorithm (GA) kèm giao diện GUI.

## 1. Tổng quan

Dự án giải bài toán Travelling Salesman Problem (TSP) bằng GA với các điểm nổi bật:

- Hai backend GA: `custom` và `simpleai`.
- Chế độ so sánh BAT (BAT comparison) chạy song song với GA.
- GUI PyQt5 hiển thị route, convergence, overlay nhiều lần chạy.
- Hỗ trợ seed cho **thuật toán** và **thành phố** (city seed).
- Thống kê nhiều lần chạy: best distance, fitness, diversity, tốc độ hội tụ.
- Biểu đồ boxplot để xem độ phân tán của kết quả qua nhiều lần chạy.

## 2. Cấu trúc thư mục chính

```
AI_rp/
│-- tsp_ga_gui.py
│-- run_single.py
│-- run_batch.py
│-- README_vi.md
│-- README.md
│-- tsp_ga_app/
│   │-- gui.py
│   │-- solver.py
│   │-- simpleai_solver.py
│   │-- bat_solver.py
│   │-- problem.py
│   │-- operators.py
│   │-- config.py
│   │-- visualization.py
│-- ga/
│-- core/
│-- experiments/
│-- outputs/
```

## 3. Cài đặt

```bash
pip install numpy matplotlib simpleai pyqt5
```

## 4. Chạy chương trình

### 4.1 GUI

```bash
python tsp_ga_gui.py
```

### 4.2 Chạy một lần (CLI)

```bash
python run_single.py
```

### 4.3 Batch (nhiều cấu hình)

```bash
python run_batch.py
```

## 5. Giải thích seed (rất quan trọng)

- **Seed thuật toán** (Use fixed seed): ảnh hưởng đến selection/crossover/mutation và BAT.
- **Seed thành phố** (Use city seed): ảnh hưởng đến dữ liệu thành phố ngẫu nhiên.

Nếu muốn **mỗi lần chạy giống hệt nhau**, hãy bật cả hai seed.

## 6. Convergence View (bảng hội tụ)

### 6.1 Chọn Metric

Dropdown **Metric** cho phép chọn:

- Best distance
- Avg fitness
- Diversity
- Convergence speed (mức cải thiện mỗi generation)

Nếu metric không có dữ liệu (ví dụ BAT không có avg fitness/diversity), đường tương ứng sẽ được bỏ qua.

### 6.2 Focus Run

Dropdown **Focus run** cho phép chọn một run:

- Chọn **All runs**: tất cả đường đều đậm.
- Chọn **Run N**: run đó đậm, các run khác mờ hơn.

### 6.3 Reset Convergence

Nút **Reset convergence** xoá toàn bộ lịch sử overlay để bắt đầu từ đầu.

### 6.4 Hành vi khi chạy

- Trong lúc đang chạy (playback), convergence chỉ hiển thị **run hiện tại**.
- Khi chạy xong, hệ thống tự chuyển sang overlay toàn bộ các run đã lưu.

## 7. Biểu đồ thống kê (Show metrics)

Nhấn **Show metrics** để mở cửa sổ thống kê.

### 7.1 Dispersion (boxplot)

- Mục đích: xem độ phân tán kết quả qua nhiều lần chạy.
- Cách dùng:
  - Chạy ít nhất **2 run**.
  - Chọn **Dispersion (boxplot)**.
  - Chọn metric (Best distance / Runtime / Avg fitness / Diversity).

Boxplot sẽ hiển thị **một hộp duy nhất** đại diện cho phân bố của tất cả run.
Nếu chỉ có 1 run hoặc metric không có dữ liệu, biểu đồ sẽ báo **No data for selected metric**.

### 7.2 Convergence speed

- Hiển thị tốc độ cải thiện (delta distance) theo từng generation.
- Có thể overlay nhiều run, có cả BAT nếu có dữ liệu.
- Focus run áp dụng giống như convergence view chính.

## 8. Ghi chú về dữ liệu JSON thành phố

- Có thể load file JSON bất kỳ có cấu trúc `cities` hoặc `points`.
- Dữ liệu hợp lệ sẽ hiển thị ngay trên không gian (route plot).
- Nếu chọn “Random cities”, hãy nhấn **Load to space** để tạo dữ liệu mới.

## 9. Đầu ra (outputs)

Khi chạy GUI hoặc batch, thư mục `outputs/` sẽ lưu:

- config.json
- metrics.csv
- best_solution.json
- summary.json
- experiment_log.txt
- convergence.png, final_route.png, evolution.gif (nếu đủ dữ liệu)

## 10. FAQ nhanh

**Q: Vì sao cùng seed nhưng city khác?**  
A: Nếu chỉ bật seed thuật toán, thành phố vẫn ngẫu nhiên. Hãy bật **Use city seed** để cố định city.

**Q: Vì sao boxplot trống?**  
A: Bạn cần chạy ít nhất 2 lần và chọn metric có dữ liệu. Với avg fitness/diversity, chỉ backend custom mới có đầy đủ.

**Q: Convergence không chạy mượt?**  
A: Khi run đang chạy, convergence chỉ hiển thị run hiện tại để tránh giật lag. Khi xong sẽ overlay lại.

---

Nếu cần chỉnh thêm UI, thêm metric mới hoặc xuất báo cáo tự động, cứ báo nhé.