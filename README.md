# ViSign2Voice - Multimodal Recognition and Translation of Vietnamese Sign Language to Text and Speech

## Giới thiệu
Dự án Nhận dạng và Dịch thuật Ngôn ngữ Ký hiệu tiếng Việt sang Văn bản và Giọng nói. Hệ thống sử dụng MediaPipe để trích xuất đặc trưng hình thể (pose, hands) và mô hình Deep Learning (Keras) để phân loại các từ vựng ngôn ngữ ký hiệu theo thời gian thực.

## Cấu trúc thư mục

- `data/`: Chứa dữ liệu (bị bỏ qua bởi git). Chia thành `raw/` (dữ liệu thô), `processed/` (dữ liệu đã trích xuất đặc trưng), và `interim/`.
- `models/`: Chứa mã nguồn định nghĩa kiến trúc các mô hình Deep Learning (GCN, LSTM, Transformer...).
- `saved_models/`: Nơi lưu trữ các trọng số (weights) của mô hình sau khi huấn luyện (`.keras`, `.h5`).
- `notebooks/`: Các file Jupyter Notebook dùng để thử nghiệm, visualize dữ liệu.
- `src/`: Các script tiền xử lý dữ liệu và data loaders.
- `utils/`: Các hàm tiện ích (vẽ bounding box, vẽ landmarks, tính toán metrics).
- `demo/`: Ứng dụng chạy theo thời gian thực (real-time inference).
- `logs/`: Chứa nhật ký huấn luyện (TensorBoard).
- `config.py`: File cấu hình chung (đường dẫn, hyperparameters).
- `train.py`: Script huấn luyện mô hình.
- `evaluate.py`: Script đánh giá mô hình.

## Hướng dẫn cài đặt

1. Clone repository này về máy:
   ```bash
   git clone <URL_CUA_REPO>
   cd ViSign2Voice
   ```

2. Tạo môi trường ảo (khuyến nghị):
   ```bash
   python -m venv venv
   # Kích hoạt môi trường (Windows)
   venv\Scripts\activate
   # (Hoặc MacOS/Linux)
   source venv/bin/activate
   ```

3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Chạy Demo Thực tế
```bash
python demo/realtime_demo.py
```
