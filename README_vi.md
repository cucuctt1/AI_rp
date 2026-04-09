# Huong dan chi tiet (Tieng Viet) - TSP bang Genetic Algorithm

Tai lieu nay mo ta day du cach to chuc code theo nhieu file, cach thuat toan hoat dong, va cach chay/chinh sua.

## 1. Tong quan

Du an giai bai toan Travelling Salesman Problem (TSP) bang Genetic Algorithm (GA) voi cac thanh phan chinh:

- Bieu dien ca the bang hoan vi chi so thanh pho (path representation)
- Selection: Tournament Selection (k = 3)
- Crossover: Order Crossover OX1
- Mutation: Inversion Mutation
- Elitism: giu lai top ELITE_SIZE ca the tot nhat moi the he
- Visualization:
  - Animation qua trinh toi uu bang FuncAnimation
  - Bieu do route cuoi cung
  - Bieu do hoi tu (convergence)

## 2. Cau truc thu muc

```text
AI_rp/
|-- tsp_ga.py
|-- README_vi.md
|-- tsp_ga_app/
|   |-- __init__.py
|   |-- config.py
|   |-- problem.py
|   |-- operators.py
|   |-- solver.py
|   |-- visualization.py
```

Y nghia tung file:

- tsp_ga.py:
  - Entry point gon, chi goi ham main de chay chuong trinh.
- tsp_ga_app/config.py:
  - Tat ca tham so mac dinh cua GA va visualization.
- tsp_ga_app/problem.py:
  - Ham tao city, tinh distance matrix, tinh tong quang duong route, va fitness.
- tsp_ga_app/operators.py:
  - Cac toan tu tien hoa: tao quan the, tournament selection, OX1 crossover, inversion mutation, evolve population.
- tsp_ga_app/solver.py:
  - Vong lap GA chinh, theo doi best theo thoi gian, tra ve du lieu cho animation/convergence.
- tsp_ga_app/visualization.py:
  - Ve route, ve convergence, animate evolution.

## 3. Mapping cac ham bat buoc

Tat ca ham yeu cau van co day du, nhung duoc tach theo module de de bao tri:

- generate_cities: problem.py
- compute_distance_matrix: problem.py
- route_distance: problem.py
- fitness: problem.py
- create_population: operators.py
- tournament_selection: operators.py
- crossover_OX1: operators.py
- mutation_inversion: operators.py
- evolve_population: operators.py
- genetic_algorithm: solver.py
- plot_route: visualization.py
- plot_convergence: visualization.py
- animate_evolution: visualization.py

Ham main nam o tsp_ga_app/main.py va duoc goi qua tsp_ga.py.

## 4. Luong xu ly tong the

1. Tao ngau nhien N thanh pho trong mat phang 2D.
2. Tinh truoc distance matrix doi xung bang khoang cach Euclid.
3. Tao quan the ban dau (cac hoan vi hop le).
4. Lap qua moi generation:
   - Danh gia distance cua tung route
   - Cap nhat best route toan cuc
   - Luu lich su best distance va best route (de ve va animate)
   - Sinh quan the moi bang elitism + selection + crossover + mutation
5. In ket qua cuoi cung (best route, best distance, muc cai thien).
6. Hien thi animation, route cuoi, convergence.

## 5. Chi tiet thuat toan

### 5.1 Bieu dien chromosome

Moi chromosome la mot list chi so thanh pho, vi du:

```text
[0, 3, 1, 4, 2]
```

Route hop le phai la hoan vi day du:

- Khong trung lap
- Khong thieu city
- Do dai bang so thanh pho

### 5.2 Ham muc tieu va fitness

Tong do dai route tinh theo chu trinh kin, co canh quay ve diem dau:

D(route) = sum(dist(route[i], route[i+1])) + dist(route[last], route[0])

Fitness:

F(route) = 1 / D(route)

Distance cang nho thi fitness cang lon.

### 5.3 Tournament Selection

- Chon ngau nhien k = 3 ca the tu quan the
- Ca the co distance nho nhat trong nhom se thang
- Lap lai de lay parent A va parent B

Uu diem:

- De cai dat
- Can bang giua khai pha (exploration) va khai thac (exploitation)

### 5.4 OX1 Crossover (diem de sai nhat)

Quy trinh:

1. Chon 2 vi tri cat left, right.
2. Copy doan parent A [left:right] vao dung vi tri trong child.
3. Duyet parent B theo thu tu goc, bo qua gene da co trong child.
4. Dien cac gene con lai vao child theo vong tron, bat dau tu vi tri right + 1.

Dam bao:

- Child luon la hoan vi hop le
- Khong duplicate
- Khong mat gene

### 5.5 Inversion Mutation

- Chon 2 moc i, j ngau nhien
- Dao nguoc doan con route[i:j]

Vi du:

```text
Truoc: [0, 1, 2, 3, 4, 5]
Dao doan [2:4]
Sau:   [0, 1, 4, 3, 2, 5]
```

Toan tu nay khong lam hu hoan vi.

### 5.6 Elitism

Moi generation giu nguyen top ELITE_SIZE ca the tot nhat.

Loi ich:

- Khong lam mat nghiem tot nhat da tim duoc
- Tang do on dinh hoi tu

## 6. Visualization

### 6.1 Route plot

- Scatter tat ca city
- Noi duong di theo thu tu route
- Dong chu trinh ve city dau
- Hien nhan (index) tung city

### 6.2 Convergence plot

- Truc X: generation
- Truc Y: best distance (best-so-far)
- Duong xu huong giam cho thay tien bo toi uu

### 6.3 Animation

- Dung matplotlib.animation.FuncAnimation
- Moi frame cap nhat:
  - Toa do line route
  - Tieu de: generation hien tai + best distance

Neu backend khong interactive (vi du Agg), animation hien thi se duoc bo qua de tranh canh bao; phan tinh toan van chay binh thuong.

## 7. Cach chay

Tu thu muc AI_rp:

```bash
python tsp_ga.py
```

Neu can cai thu vien:

```bash
pip install numpy matplotlib
```

## 8. Chinh tham so

Tat ca tham so nam trong tsp_ga_app/config.py:

- POP_SIZE = 100
- GENERATIONS = 200
- MUTATION_RATE = 0.1
- CROSSOVER_RATE = 0.8
- ELITE_SIZE = 2
- NUM_CITIES = 20
- TOURNAMENT_SIZE = 3
- ANIMATION_INTERVAL_MS = 80
- RANDOM_SEED = 42
- SAVE_GIF = False
- GIF_PATH = "tsp_ga_evolution.gif"

Goi y:

- Tang POP_SIZE hoac GENERATIONS -> nghiem thuong tot hon, nhung cham hon.
- Tang MUTATION_RATE qua cao co the lam giam do on dinh.
- ELITE_SIZE qua lon co the lam quan the de bi hoi tu som.

## 9. Bao dam tinh dung

Code dang co cac co che bao ve:

- Kiem tra population khong rong
- Kiem tra parent cung do dai khi crossover
- Kiem tra child sau OX1 la hoan vi hop le
- Kiem tra child sau mutation/evolution van la hoan vi hop le
- Luon tinh distance matrix mot lan va tai su dung
- Route distance luon tinh ca canh quay ve diem dau

## 10. Mo rong de xuat

- Thu nghiem nhieu seed de danh gia do on dinh
- So sanh them PMX hoac CX crossover voi OX1
- Them benchmark theo NUM_CITIES lon hon
- Ghi log best distance ra file CSV de phan tich sau
