# Deep Hedging trên CPU — hướng dẫn Windows

Đường chạy `run_cpu.py` dùng simulator, mạng policy, utility OCE, chi phí giao
dịch, soft clipping và các recurrent states của repo gốc. Không cần GPU.

## Cài đặt

Dùng **Python 3.11 64-bit** hoặc **3.12**. Bộ phiên bản cố định trong
`requirements-cpu.txt` không hỗ trợ Python 3.13; có thể cài 3.11 song song.
Mở CMD hoặc PowerShell tại thư mục muốn lưu dự án:

```powershell
git clone https://github.com/thanhbien10a4-ui/DeepHedging.git DeepHedging-CPU
cd DeepHedging-CPU
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Nếu dùng Python 3.12, thay `py -3.11` bằng `py -3.12`. Git có thể yêu cầu
đăng nhập vì repo private. Dùng trực tiếp Python trong `.venv`, không cần
đổi ExecutionPolicy. Linux dùng `python3.11 -m venv .venv` và `.venv/bin/python`.

## Chạy thử rồi huấn luyện

```powershell
.\.venv\Scripts\python.exe run_cpu.py --preset smoke --threads 2 --output runs/smoke
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --threads 4 --output runs/laptop
```

| Preset | Train | Validation | Bước thời gian | Epoch tối đa | Batch |
|---|---:|---:|---:|---:|---:|
| `smoke` | 256 | 128 | 10 | 2 | 64 |
| `laptop` | 4.096 | 1.024 | 20 | 100 | 256 |
| `full` | 10.000 | 2.000 | 20 | 800 | 256 |

Smoke là huấn luyện thật nhưng ngắn, để kiểm tra cài đặt và checkpoint;
không chứng minh hội tụ. Thành công có `target_reached`, 8 Adam steps,
`visible_gpus: []`. Epoch đầu gồm biên dịch đồ thị nên thường lâu hơn.

Mặc định: stochastic world gốc, hedge spot và option, mạng feedforward
3 tầng ẩn × 20 nút, ReLU, initial-delta agent, `exp2`, risk aversion 1,
Adam learning rate 0,001. Preset nhẹ giảm lượng công việc và có thể ảnh
hưởng chất lượng. `full` không có nghĩa chạy toàn bộ notebook trong repo.

## Dừng và chạy tiếp

```powershell
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --threads 4 --output runs/laptop --resume
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --epochs 300 --threads 4 --output runs/laptop --resume
```

`--epochs 300` là tổng mục tiêu, không phải cộng thêm 300. `Ctrl+C` dừng
tiến trình; resume đọc checkpoint hoàn chỉnh gần nhất, gồm model, Adam
moments, iteration và epoch. Mặc định lưu mỗi 5 epoch và tại validation.
Phần chưa checkpoint được chạy lại, lịch sử dư bị loại bỏ.

Giữ nguyên dữ liệu, seed, batch, world, số bước, learning rate và lịch
validation khi resume. Có thể đổi số luồng, tổng epoch, tần suất checkpoint.
Đổi source hoặc phiên bản cốt lõi cũng bị từ chối để tránh resume sai.
Thư mục có dữ liệu không bị ghi đè nếu thiếu `--resume`.

## Tùy chọn và đo tốc độ

```powershell
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --world bs --output runs/bs
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --recurrent 4 --output runs/recurrent
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --eval-every 5 --patience 10 --output runs/early-stop
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --epochs 12 --threads 2 --output runs/benchmark-2
.\.venv\Scripts\python.exe run_cpu.py --preset laptop --epochs 12 --threads 4 --output runs/benchmark-4
.\.venv\Scripts\python.exe run_cpu.py --help
```

Chạy hai benchmark lần lượt. Nhiều luồng hơn không nhất thiết nhanh hơn.
So sánh `median_warm_train_epoch_seconds` và `invocation_wall_seconds`
trong `summary.json`. ETA là phần train còn lại, chưa cộng validation,
checkpoint và biên dịch ban đầu. Không suy ra tốc độ laptop từ CPU khác.

`bs` chuyển sang bài toán Black–Scholes chỉ hedge spot theo simulator gốc.
`--recurrent 4` là 4 classic recurrent states gốc, không phải lớp Keras LSTM.
`--patience 10` dừng sau 10 lần validation không cải thiện; mặc định tắt.
Validation chạy theo lịch và tại epoch cuối. Mở rộng tổng epoch sau một
lần đã kết thúc có thể thêm một điểm validation so với chạy thẳng.

## Đầu ra

| Tệp/thư mục | Ý nghĩa |
|---|---|
| `run.json` | Cấu hình, phiên bản, hash source để kiểm tra resume |
| `metrics.csv` | Loss và thời gian từng epoch |
| `checkpoints/` | 2 trạng thái train gần nhất, gồm Adam |
| `best/` | Các snapshot cải thiện validation, giữ để phục hồi an toàn |
| `best.weights.h5` | Trọng số tốt nhất theo validation để suy luận |
| `summary.json` | Tổng kết, số bước Adam, thời gian, thiết bị |

Không xóa `best/` nếu còn muốn resume. Các snapshot tốt có thể tăng dung
lượng khi chạy lâu. Chỉ `best.weights.h5` không đủ để khôi phục Adam.
Epoch 0 có thể được chọn nếu mọi epoch đã train đều làm validation xấu hơn.

Loss giữ nguyên `-utility - utility0`. Train loss được trung bình trong lúc
batch cập nhật trọng số; validation loss đo với trọng số cố định. Đây không
phải hai loss được đo cùng một thời điểm. Validation chọn mô hình, chưa phải
tập test độc lập để kết luận hiệu quả hedging.

## Đánh giá mô hình đã train trên tập test mới

Sau khi một lần huấn luyện hoàn tất (ví dụ 300 epoch), mở PowerShell trong
thư mục repo và chạy:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe evaluate_cpu.py --run runs/laptop --samples 10000 --seed 20261016 --threads 2
```

Lệnh đọc cấu hình và `best.weights.h5` của lần train, tạo 10.000 đường mô phỏng
mới, rồi so sánh mô hình với không hedge (`unhedged`) và hedge delta bằng
spot (`spot_delta`). Không cần train lại hoặc cài thêm thư viện. Nếu đã đổi seed
khi train, bỏ `--seed` để tự dùng seed train + 2. Seed test không được trùng
seed train hoặc validation. Chỉ chạy sau khi tiến trình train đã kết thúc.

Mỗi lần đánh giá tạo thư mục mới `runs/laptop/evaluations/test-<seed>-<timestamp>/`
gồm `summary.json`, `strategies.csv` và `paths.csv`; không ghi lại trọng số,
Adam hoặc lịch sử train. Có thể dùng `--output runs/my-test` nếu thư mục đó
chưa tồn tại. Kiểm tra source và phiên bản dựa trên `run.json` giúp tránh
nạp trọng số vào bài toán khác. Bản bổ sung evaluator không thay các file
source được hash trong manifest huấn luyện cũ.

| Chỉ số trong `strategies` | Cách đọc |
|---|---|
| `mean_gains` | Trung bình payoff + P&L hedge − chi phí; chưa cộng premium bán option ban đầu |
| `std_gains` | Độ lệch chuẩn kết quả; thấp hơn nghĩa ít phân tán hơn |
| `mean_cost` | Chi phí giao dịch trung bình |
| `loss_var_95` | Phân vị 95% của loss = −gains |
| `loss_cvar_95` | Loss trung bình trong 5% đường đi xấu nhất; thấp hơn tốt hơn theo tiêu chí này |
| `empirical_oce_exp2` | Utility `exp2@1` ước lượng trên gains của từng chiến lược; cao hơn tốt hơn |

Trong `deep_hedging_vs_baseline`, số dương là cải thiện theo chỉ số tương ứng:
chênh lệch mean gains, phần trăm giảm độ lệch chuẩn, mức giảm tail loss hoặc
mức tăng OCE. Không phải các tiêu chí luôn đồng thời cải thiện. Loss không
phải phần trăm độ chính xác, và mean gains âm không tự nó chứng minh hedge kém.

OCE thực nghiệm giải lại **một biến vô hướng** để đo risk statistic cho từng
chiến lược; không cập nhật policy. `frozen_model.utility` giữ nguyên biến OCE
đã học nên có thể khác số này. `frozen_model.loss` giữ định nghĩa loss lúc train.

Delta baseline dùng spot và implied volatility quan sát được, strike 1, lãi
suất 0; chỉ giao dịch spot, cùng giới hạn mỗi lệnh và công thức chi phí với
mô hình. Baseline là tham chiếu đơn giản, không phải chiến lược tối ưu cho
stochastic volatility hay chi phí giao dịch. Giới hạn áp dụng trực tiếp vào
lệnh delta; mạng dùng soft clipping để biểu diễn lệnh trong cùng miền khả thi.

Một đặc điểm của simulator gốc được giữ nguyên: trong chế độ stochastic,
payoff của claim lấy giá **một bước sau** thời điểm thanh toán hedge spot,
do slice `spots[:, :nSteps+2]`; delta baseline tính thêm một `dt` cho kỳ hạn
claim. Chế độ `bs` không có độ lệch này. Không thay simulator của checkpoint
đã train khi đánh giá. Các đường đi có liên hệ qua chuẩn hóa trung bình theo
cross-section trong simulator; báo cáo không giả định chúng độc lập để đưa
khoảng tin cậy. Đây là kiểm tra trên mô phỏng cùng phân phối, chưa phải dữ
liệu thị trường thực hay kiểm tra dưới kịch bản thị trường khác.

Nếu dùng kết quả test để chọn số epoch hoặc đổi cấu hình, tập này trở thành
validation. Sau khi điều chỉnh, giữ một seed chưa dùng cho đánh giá cuối cùng.

## Các thay đổi và giới hạn

- TensorArray thay ghép lịch sử hành động tăng dần, giữ nguyên đầu ra và gradient.
- Vòng train biên dịch, float32, batch và prefetch hữu hạn, giới hạn thread pool.
- Không tính lại toàn bộ train set và chuyển mọi hành động sang NumPy mỗi epoch.
- Validation chia batch, mặc định mỗi 5 epoch; không vẽ biểu đồ liên tục.
- Best model chọn theo validation, khác trainer gốc chọn theo training loss.
- Checkpoint TensorFlow khôi phục cả model và Adam; không dùng cache cũ.

Các kiểm tra tự động được chạy trên Linux. Chưa chạy 800 epoch, chưa kiểm tra
toàn bộ notebook nghiên cứu, SANOS hoặc `under_construction`. Một số notebook
cần thư viện bổ sung. Nếu dùng notebook gốc, đặt `TF_USE_LEGACY_KERAS=1`
trước khi import TensorFlow và khởi động lại kernel nếu đã import trước đó.
Các cảnh báo CUDA khi import bản TensorFlow Linux không tự chứng minh GPU
đang được dùng; kiểm tra `visible_gpus` trong kết quả.

## Kiểm tra kỹ thuật

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Kiểm tra đầu ra và gradient với rollout tham chiếu, recurrent/feedforward,
một/hai công cụ hedge, batch lẻ với trọng số không đều, resume Adam so với
chạy liên tục và chống ghi đè. Xem [CPU_VALIDATION.md](CPU_VALIDATION.md).

Nguồn tương thích:
[TFP 0.25 và TensorFlow 2.18](https://github.com/tensorflow/probability/releases/tag/v0.25.0),
[Keras 2/3](https://keras.io/keras_3/),
[TensorFlow trên Windows](https://www.tensorflow.org/install/pip).
