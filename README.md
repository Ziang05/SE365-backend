# FinDec AI - Backend (SE365-backend)

Đây là máy chủ xử lý AI (Backend) cho hệ thống Trích xuất sự kiện tài chính từ tin tức CafeF. Nó phụ trách việc cào dữ liệu (crawl) và chạy mô hình AI (PhoBERT, BARTpho) để phân tích bài báo.

## 🛠 Yêu cầu hệ thống
- **Python** 3.10 trở lên.
- Nên có **GPU NVIDIA** (VD: RTX 3060, 4060...) để chạy AI nhanh hơn (tốn khoảng 15s/bài thay vì 5 phút/bài trên CPU).

## 🚀 Hướng dẫn cài đặt & chạy nhanh (Local)

### Bước 1: Cài đặt thư viện
Mở terminal (PowerShell/CMD) tại thư mục `SE365-backend` và chạy lệnh sau:

```bash
pip install -r requirements.txt
```

*(Lưu ý: Nếu máy bạn có GPU NVIDIA, hãy cài thêm PyTorch hỗ trợ CUDA để chạy cực nhanh bằng lệnh: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`)*

### Bước 2: Chạy Server
Tiếp tục chạy lệnh sau để khởi động backend:

```bash
python -m uvicorn main:app --port 8000
```
- Khi terminal hiện dòng `[FinDec] All 4 models ready on cuda` (hoặc cpu), nghĩa là AI đã tải xong và sẵn sàng nhận yêu cầu từ web.
- Server sẽ chạy tại địa chỉ: `http://localhost:8000`.

---
**💡 Mẹo nhỏ về file `.env`:**
Project đã được cấu hình sẵn để chạy ngay trên máy cá nhân của bạn. Bạn **KHÔNG CẦN** phải tạo file `.env` (từ file `.env.example`). Hệ thống sẽ tự động dùng các giá trị mặc định chuẩn xác nhất cho localhost. Bạn chỉ cần quan tâm đến file `.env` khi nào muốn đưa project lên server thật (production).
