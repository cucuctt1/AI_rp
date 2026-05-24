# BẢNG PHÂN CÔNG NHIỆM VỤ

| Thành viên | Nhiệm vụ |
|---|---|
| Tất Chí Thành | Phụ trách đột biến và cơ chế elitism trong GA, đồng thời kiểm tra tính hợp lệ của lời giải sau mỗi thế hệ và hỗ trợ phần đo lường hiệu suất liên quan đến hội tụ. |
| Nguyễn Minh Thức | Phụ trách giao diện người dùng và luồng tương tác của chương trình, bao gồm nhập tham số, chọn dữ liệu, chạy thuật toán và hiển thị kết quả. |
| Trần Xuân Phát | Phụ trách xử lý dữ liệu đầu vào và hàm khoảng cách, bao gồm đọc JSON, sinh random cities, xây dựng ma trận khoảng cách và tính fitness dựa trên route distance. |
| Vũ Đặng Khánh My | Phụ trách selection, crossover và khởi tạo quần thể trong GA, đảm bảo các toán tử di truyền tạo ra cá thể hợp lệ theo biểu diễn hoán vị. |

# Bộ Giải TSP Bằng Giải Thuật Di Truyền Đã Được Tái Cấu Trúc

## Chạy Chương Trình

```bash
python main.py
```

Lệnh này khởi chạy giao diện Studio PyQt5 hiện tại.

Các entrypoint khác:

```bash
python main.py --cli
python main.py --legacy-gui
python main.py
python -m pytest -q
```

## Cấu Trúc

- `app/` chứa phần triển khai chính đã được làm sạch.
- `app/algorithms/` chứa các helper dùng chung cho bài toán TSP, các toán tử route và core GA engine legacy.
- `app/solvers/` chứa custom GA, SimpleAI GA và BAT solver.
- `app/ui/` chứa Studio UI, các đối tượng worker của UI và GUI legacy.
- `app/experiments/` và `app/reporting/` chứa các runner thí nghiệm, exporter, logging và phần tổng hợp kết quả.
- `core/`, `ga/`, `utils/`, `experiments/` và `tsp_ga_app/` là các compatibility wrapper để giữ các đường import cũ hoạt động.
- `outputs/` là vị trí mặc định duy nhất để lưu các artifact được sinh ra từ bản tái cấu trúc này.

## Hướng Dẫn Theo File

### Các File Ở Thư Mục Gốc

- `main.py` là entrypoint chính của dự án đã tái cấu trúc. Khi không truyền flag, file này mở Studio GUI; `--cli` chạy demo solver không dùng GUI; `--legacy-gui` mở GUI nhỏ cũ hơn.
- `requirements.txt` liệt kê các thư viện Python cần cho ứng dụng đã tái cấu trúc, test, plotting, GUI, SimpleAI solver và xuất GIF.
- `README.md` mô tả cách chạy dự án đã tái cấu trúc, cách tổ chức thư mục và trách nhiệm của từng file.
- `run_single.py` chạy một thí nghiệm mẫu với các thành phố được sinh ra, legacy/core GA engine, quy trình xuất kết quả và metadata dùng cho báo cáo.
- `run_batch.py` chạy một thí nghiệm batch grid search mẫu theo mutation rate, crossover type và selection type.
- `tsp_ga.py` giữ hành vi launcher CLI top-level cũ. Mặc định file này chạy CLI solver và có hỗ trợ `--gui`.
- `tsp_ga_gui.py` giữ launcher GUI legacy top-level cũ.
- `gui_runner.py` giữ đường import/chạy GUI legacy cũ, đồng thời chuyển phần triển khai sang `app.ui.legacy_window`.

### Gói Ứng Dụng Chính

- `app/__init__.py` đánh dấu `app` là gói triển khai canonical của bản tái cấu trúc.
- `app/paths.py` định nghĩa các đường dẫn cục bộ của dự án, đặc biệt là `PROJECT_ROOT` và `OUTPUT_ROOT`
- `app/cli.py` chứa workflow solver không dùng GUI: thiết lập seed, sinh thành phố, tạo ma trận khoảng cách, chọn solver, in summary, tạo animation, vẽ route cuối và vẽ convergence plot.

### Cấu Hình

- `app/config/__init__.py` re-export các thiết lập để import thuận tiện hơn.
- `app/config/settings.py` chứa toàn bộ hằng số mặc định: thiết lập population/generation của GA, mutation/crossover rate, cấu hình SimpleAI, mặc định so sánh BAT, thiết lập animation, random seed, đường dẫn GIF và `DEFAULT_CONFIG` legacy.

### Thuật Toán

- `app/algorithms/__init__.py` cung cấp các helper thuật toán chính và `GAEngine`.
- `app/algorithms/problem.py` chứa các tiện ích chung cho bài toán TSP: trạng thái ma trận khoảng cách đang dùng, sinh thành phố ngẫu nhiên, tạo ma trận khoảng cách Euclid, tính route distance và fitness nghịch đảo theo khoảng cách.
- `app/algorithms/operators.py` chứa các toán tử route/population được dùng trong dự án: tạo population, tournament selection dựa trên khoảng cách, OX1 crossover, inversion mutation, tiến hóa population, tournament/roulette selection dựa trên fitness, PMX/order crossover, swap/reverse/scramble mutation, elitism và local search 2-opt.
- `app/algorithms/core_engine.py` chứa class core GA engine legacy có thể cấu hình, được dùng bởi experiment runners và GUI legacy. Engine này hỗ trợ lựa chọn selection/crossover/mutation operators, elitism, adaptive mutation, local search tùy chọn, metrics, history, runtime và callback progress.

### Solver

- `app/solvers/__init__.py` cung cấp ba solver entrypoint.
- `app/solvers/custom_ga.py` chứa custom GA solver của ứng dụng, được Studio UI và CLI sử dụng. Solver này theo dõi best route, best distance, convergence history, route history, initial best distance, metadata tùy chọn và các live progress payload.
- `app/solvers/simpleai_ga.py` chứa GA solver dựa trên SimpleAI. File này kiểm tra và làm sạch distance matrix, định nghĩa bài toán TSP cho SimpleAI, ghi history thông qua viewer, hỗ trợ hành vi GA manual fallback, elitism/diversity injection tùy chọn, chạy multi-restart và refinement 2-opt tùy chọn.
- `app/solvers/bat.py` chứa metaheuristic TSP lấy cảm hứng từ BAT, được dùng cho các run so sánh trong Studio UI. Solver này triển khai guided movement an toàn với hoán vị, local walk, inversion mutation, cập nhật loudness/pulse, progress payload và convergence history.

### Giao Diện

- `app/ui/__init__.py` cung cấp Studio window và launcher.
- `app/ui/studio_window.py` chứa Studio UI chính bằng PyQt5. File này xây dựng control panel và vùng plot đầy đủ, xử lý dataset, nút run/batch, các chế độ xem convergence, hộp thoại metrics, vẽ route, batch overlay, export sau khi chạy xong và toàn bộ hành vi hiển thị của Studio.
- `app/ui/studio_workers.py` chứa các đối tượng Qt worker để chạy nền. `SolverWorker` chạy solver được chọn và so sánh BAT tùy chọn; `BatchWorker` chạy các batch grid search mà không chặn UI.
- `app/ui/city_io.py` chứa các helper tái sử dụng để parse thành phố từ JSON cho point list, point map, dataset có tên, `cities`, `points` và các dictionary tọa độ.
- `app/ui/legacy_window.py` chứa GUI PyQt5 nhỏ cũ hơn. File này giữ lại các control đơn giản trước đây, progress animation, route/convergence plot, load JSON và hành vi run/reset legacy.

### Thí Nghiệm

- `app/experiments/__init__.py` cung cấp các hàm runner thí nghiệm.
- `app/experiments/runner.py` chạy một thí nghiệm đơn qua `GAEngine`, xuất config/metrics/best solution/summary/population snapshots, cập nhật raw result và summary CSV/JSON, đồng thời lưu plot/GIF nếu có thể.
- `app/experiments/batch_runner.py` chạy các tổ hợp grid search theo parameter grid và số trial, tạo thư mục batch cha, chuyển output của từng trial vào bên trong, thêm batch raw results và lưu biểu đồ cột best distance của batch khi có thể.

### Báo Cáo Và Xuất Kết Quả

- `app/reporting/__init__.py` cung cấp các helper exporter và logging.
- `app/reporting/exporter.py` tạo các thư mục thí nghiệm dưới `/outputs` và ghi config JSON, metrics CSV, best solution JSON, summary JSON, population snapshots, figures và các dòng batch raw result.
- `app/reporting/logger.py` cấu hình logger `GA_TSP` cho console output và file log theo từng thí nghiệm nếu cần.
- `app/reporting/results.py` quản lý result schema và summary reporting. File này chuyển giá trị NumPy sang dữ liệu an toàn cho JSON, đọc/ghi CSV/JSON, ghi dataset metadata, thêm raw results, tính nearest-neighbor baseline, xây dựng các trường optimality và cập nhật summary statistics.

### Trực Quan Hóa

- `app/visualization/__init__.py` cung cấp các helper plotting.
- `app/visualization/plots.py` tạo Matplotlib route plot, convergence plot và animation tiến hóa route. File này cũng có thể lưu evolution GIF vào đường dẫn cục bộ đã cấu hình trong dự án.

### Công Cụ

- `app/tools/__init__.py` đánh dấu package tools.
- `app/tools/gen_data.py` chuyển các dòng CSV theo kiểu TSPLIB thành dataset thành phố dạng JSON. File này xử lý giới hạn kích thước field CSV, tạo slug filename, parse tọa độ và xuất JSON theo từng instance.
- `app/tools/reproduce_report_figures.py` đọc các file CSV báo cáo đã sinh ra và lưu các hình dùng cho báo cáo: best distance theo run, summary mean với confidence interval và số lượng thành phố theo dataset.

### Compatibility Wrapper

Các file này tồn tại để các đường import cũ vẫn hoạt động khi chạy bên trong `new_refract/`. Chúng nên giữ mỏng và chuyển tiếp sang `app.*`.

- `core/__init__.py`, `core/config.py` và `core/ga_engine.py` giữ các import `core.*` cũ cho `DEFAULT_CONFIG` và `GAEngine`.
- `ga/__init__.py`, `ga/selection.py`, `ga/crossover.py`, `ga/mutation.py`, `ga/elitism.py` và `ga/local_search.py` giữ các import toán tử GA cũ.
- `utils/__init__.py`, `utils/exporter.py`, `utils/logger.py` và `utils/results_reporting.py` giữ các import reporting/logging cũ.
- `experiments/__init__.py`, `experiments/runner.py` và `experiments/batch_runner.py` giữ các import experiment runner cũ.
- `tsp_ga_app/__init__.py`, `tsp_ga_app/config.py`, `tsp_ga_app/problem.py`, `tsp_ga_app/operators.py`, `tsp_ga_app/solver.py`, `tsp_ga_app/simpleai_solver.py`, `tsp_ga_app/bat_solver.py`, `tsp_ga_app/visualization.py`, `tsp_ga_app/visualize.py`, `tsp_ga_app/gui.py` và `tsp_ga_app/main.py` giữ API của package ứng dụng cũ trong khi chuyển tiếp sang phần triển khai đã tái cấu trúc.
- `data_gen/__init__.py` và `data_gen/gen_data.py` giữ đường import và đường script chuyển đổi dữ liệu cũ.
- `scripts/__init__.py` và `scripts/reproduce_report_figures.py` giữ đường script tạo figure báo cáo cũ.

### Kiểm Thử

- `tests/conftest.py` bảo đảm `new_refract/` nằm trong `sys.path` khi chạy test.
- `tests/test_selection.py` kiểm tra cả selection legacy dựa trên fitness và tournament selection của app dựa trên khoảng cách đều trả về route hợp lệ.
- `tests/test_crossover.py` kiểm tra PMX, order crossover và OX1 đều tạo ra hoán vị hợp lệ.
- `tests/test_mutation.py` kiểm tra swap, reverse, scramble và inversion mutation đều giữ route hợp lệ.
- `tests/test_elitism.py` kiểm tra elitism của core giữ route có fitness tốt nhất và quá trình evolution của app giữ elite có khoảng cách thấp nhất.
- `tests/test_distance.py` kiểm tra quá trình tạo ma trận khoảng cách Euclid và cyclic route distance.
- `tests/test_route_validity.py` kiểm tra population được sinh ra chứa mỗi thành phố đúng một lần.
- `tests/test_reproducibility.py` kiểm tra cùng seed tạo ra cùng best route và best distance qua `GAEngine`.
- `tests/test_result_schema.py` kiểm tra raw result export chứa các trường bắt buộc.
- `tests/test_dataset_metadata.py` kiểm tra dataset metadata export chứa các trường bắt buộc.
- `tests/test_optimality_gap.py` kiểm tra phép tính gap so với known optimum và nhãn khi không có optimum.
- `tests/test_output_isolation.py` kiểm tra output root mặc định của exporter.

### Output.

- `outputs/.gitkeep` giữ thư mục output tồn tại trong version control.
- `outputs/` là nơi CLI, GUI, experiment runners, báo cáo, summary, logs, figures, population snapshots và GIF được ghi mặc định.

## Tương Thích

```python
from ga.selection import tournament_selection
from core.ga_engine import GAEngine
from tsp_ga_app.problem import compute_distance_matrix
from utils.results_reporting import append_raw_result
```

Các wrapper re-export hàm/class từ `app/`; code mới nên ưu tiên import từ `app.*`.



