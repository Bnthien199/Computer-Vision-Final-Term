# PoseFit

Nhận diện **plank, pushup, squat, standing** qua webcam bằng MediaPipe Pose + MLP;
đếm squat/push-up và đo thời gian plank bằng luật hình học theo thời gian.
Demo dùng các artifact đi kèm, **không cần tải dataset và không cần train lại**.
Đây là demo học thuật, chưa chấm kỹ thuật tập đúng/sai. Độ chính xác phân loại
không đồng nghĩa độ chính xác đếm rep; push-up và plank cần kiểm chứng thêm.

## Cài đặt và chạy webcam trên Windows

Cần Windows x64, **Python 3.11 x64** (đã kiểm thử với 3.11.9), Git, webcam và
Internet để cài thư viện. Kiểm tra `py -3.11 --version` trước khi bắt đầu.
Python/MediaPipe phiên bản khác chưa được kiểm thử; runtime yêu cầu legacy
`mp.solutions.pose` của MediaPipe 0.10.21. TensorFlow chạy CPU, không cần CUDA.

Chạy trong PowerShell; các lệnh dưới đây không cần activate môi trường hoặc
đổi ExecutionPolicy:

```powershell
git clone https://github.com/Bnthien199/Computer-Vision-Final-Term.git
cd Computer-Vision-Final-Term
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check

# Kiểm tra checksum, nạp bộ inference, chạy MLP và MediaPipe trên dữ liệu giả
.\.venv\Scripts\python.exe .\check_demo.py

# Đọc thử một frame webcam, không lưu ảnh/video
.\.venv\Scripts\python.exe .\check_demo.py --webcam --camera 0

# Mở cửa sổ demo
.\.venv\Scripts\python.exe .\14_final_posefit.py --camera 0
```

`requirements.txt` liệt kê dependency trực tiếp và pin JAX tương thích với
TensorFlow; `requirements-lock.txt` ghi toàn bộ dependency của môi trường demo
đã kiểm tra trên Windows/Python 3.11. Chỉ cài một gói cung cấp `cv2` là
`opencv-contrib-python`, có GUI; không cài thêm bản headless.

Nếu camera 0 không hoạt động, thử cả hai lệnh webcam/demo với `--camera 1`.
Đóng ứng dụng khác đang giữ camera, bật quyền camera cho desktop apps trong
Windows Settings > Privacy & security > Camera. Smoke check chỉ xác nhận mở và
đọc camera; cần tự quan sát cửa sổ demo để kiểm tra hiển thị và nhận diện thật.
Nếu thiếu DLL lúc import TensorFlow, cài Microsoft Visual C++ Redistributable
2015–2022 x64 từ Microsoft rồi mở PowerShell mới.

Đứng đủ xa để thấy toàn thân; camera nhìn ngang, đủ sáng. Squat cần thấy
vai/hông/gối/mắt cá; push-up cần vai/khuỷu/cổ tay/hông. Plank cần tư thế ổn định
khoảng một giây. Rep cần chu trình trên → dưới → trên. Mất dấu sẽ ngắt tích lũy
thời gian. `Unknown` là ngưỡng từ chối theo confidence, không phải lớp đã học.

Focus cửa sổ: **Q/Esc** thoát, **R** reset rep/timer. Xem tùy chọn:

```powershell
.\.venv\Scripts\python.exe .\14_final_posefit.py --help
.\.venv\Scripts\python.exe .\14_final_posefit.py --camera 1 --confidence 0.65 --width 640
```

Mặc định capture 1280×720, xử lý rộng tối đa 640 và giữ tỷ lệ ảnh; confidence
0.65, EMA 0.20 giây; squat 110°/150°, push-up 100°/150°. Mọi cấu hình mặc định
nằm trong `posefit_runtime.py`; không cần file cấu hình cục bộ hay token.
Demo mặc định không lưu frame, video hoặc trace. Tùy chọn `--plank-trace` ghi
tọa độ pose/xác suất vào thư mục mới dưới `runs/`; giữ riêng dữ liệu chẩn đoán.

## Bộ inference và quyền chia sẻ

Luồng chạy: `14_final_posefit.py` → `posefit_runtime.py` → MediaPipe 33 landmark
xyz → 99 đặc trưng → scaler → MLP → encoder/EMA → counter/timer.

| Artifact đi kèm | Vai trò |
| --- | --- |
| `models/mlp_posefit_final.keras` | MLP, đầu vào 99, đầu ra 4 |
| `training_data/scaler.pkl` | StandardScaler đã fit, 99 đặc trưng |
| `training_data/label_encoder.pkl` | Thứ tự plank, pushup, squat, standing |

Chủ dự án đã xác nhận có quyền công khai cả ba artifact trong lần chuẩn bị
phát hành này. Đây là bundle có sẵn của dự án, được giữ nguyên byte, không
train lại và không thay bằng checkpoint `best`. SHA256 nằm trong
[`artifacts.sha256.json`](artifacts.sha256.json); `check_demo.py` kiểm tra trước
khi nạp. Chỉ nạp pickle/Keras từ nguồn bạn tin cậy. Không có bước tải artifact
riêng; model Pose full của MediaPipe được cung cấp trong wheel dependency,
không sao chép vào repo. Các dependency giữ license riêng của nhà phát hành.
Repo chưa có license chung cấp quyền tái sử dụng; liên hệ chủ repo nếu cần
quyền phân phối/sửa đổi ngoài phạm vi sử dụng được cho phép.

## Train và evaluate (tùy chọn, cần dataset riêng)

Các bước này **không nằm trong quy trình chạy webcam**. Không có video, ảnh,
metadata, feature CSV hay mảng train/test trong repo. Người chạy phải tự có
dataset hợp pháp và quyền sử dụng phù hợp. Không upload video chưa xác minh
quyền chia sẻ, kể cả RepCount-A; `external_datasets/` luôn bị ignore.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m pip check
```

Pipeline `04_create_metadata.py` tìm video gym theo lớp dưới
`archive/verified_data/` và ảnh standing dưới `standing_raw/`. Chuẩn bị dữ liệu
riêng theo schema mà script kiểm tra, sau đó chạy từ thư mục repo. Chọn tên run
mới cho mỗi lần; script từ chối ghi đè đầu ra đã có:

```powershell
.\.venv\Scripts\python.exe .\04_create_metadata.py --output runs/exp01/metadata.csv --seed 42
.\.venv\Scripts\python.exe .\05_extract_landmarks.py --metadata runs/exp01/metadata.csv --output runs/exp01/features.csv
.\.venv\Scripts\python.exe .\08_prepare_training.py --input runs/exp01/features.csv --output runs/exp01/prepared
.\.venv\Scripts\python.exe .\10_train_mlp.py --data-dir runs/exp01/prepared --output runs/exp01/mlp --epochs 100 --seed 42
.\.venv\Scripts\python.exe .\14_final_posefit.py --model runs/exp01/mlp/mlp_posefit_final.keras --data-dir runs/exp01/mlp

# Đánh giá bundle đi kèm trên features/video riêng; không fit/train lại
.\.venv\Scripts\python.exe .\evaluate_posefit.py features --features runs/exp01/features.csv --split val --output runs/eval_val_01
.\.venv\Scripts\python.exe .\evaluate_posefit.py videos --metadata runs/exp01/metadata.csv --split test --output runs/eval_video_01
```

Model mới phải đi cùng scaler/encoder của nó. Metadata có nhãn bài tập không
cung cấp ground truth rep/timer; kết quả đếm trên video chỉ là chẩn đoán nếu
chưa có annotation riêng. Không suy accuracy webcam từ accuracy feature.

## Kiểm thử và giới hạn

Xem [`VALIDATION.md`](VALIDATION.md) cho kết quả kiểm tra phát hành. Smoke check
dùng landmark giả và frame trống, không đo chất lượng nhận diện. Camera và GUI
phải được kiểm tra thủ công trên máy chạy demo.
Trong phiên kiểm thử phát hành, camera 0 và 1 đều không mở được, nên chưa xác
nhận webcam/GUI thực tế; các kiểm tra inference không dùng camera đã pass.

Các regression test dùng dữ liệu tổng hợp, không cần video huấn luyện:

```powershell
.\.venv\Scripts\python.exe .\test_posefit_runtime.py
.\.venv\Scripts\python.exe .\test_posefit_plank.py
.\.venv\Scripts\python.exe .\test_posefit_continuity.py
.\.venv\Scripts\python.exe -m unittest test_posefit_plank_classification test_posefit_plank_handoff test_posefit_video_runtime
# Test pipeline/evaluation cần requirements-training.txt
.\.venv\Scripts\python.exe -m unittest test_posefit_pipeline test_posefit_evaluation
```

Hai kiểm tra lịch sử cần dataset/trace cục bộ được đánh dấu skip khi thiếu dữ liệu;
không cần tải chúng để chạy các regression test tổng hợp còn lại.
