# CafeF Crawler Backend

FastAPI server điều phối crawler tin tức chứng khoán từ [CafeF](https://cafef.vn).

## Yêu cầu

- Python ≥ 3.10
- pip

## Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd SE365-backend

# Tạo virtual environment (khuyến nghị)
python -m venv .venv
.venv\Scripts\activate   # Windows
# hoặc: source .venv/bin/activate  (Linux/Mac)

# Cài dependencies
pip install -r requirements.txt
```

## Cấu hình

Copy file `.env.example` thành `.env` rồi chỉnh sửa:

```bash
copy .env.example .env
```

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `ALLOWED_ORIGINS` | CORS origins (phân cách bằng `,`) | `http://localhost:5173` |
| `DATA_DIR` | Thư mục lưu file CSV/JSONL output | `./data/raw` |
| `PORT` | Cổng server | `8000` |

## Chạy

```bash
uvicorn main:app --reload --port 8000
```

Server sẽ chạy tại `http://localhost:8000`.

Tài liệu API tự động: `http://localhost:8000/docs`

## API Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/health` | Health check |
| `POST` | `/crawl` | Bắt đầu crawl job mới |
| `GET` | `/crawl/status/{job_id}` | Poll trạng thái job |
| `GET` | `/crawl/result/{job_id}` | Lấy kết quả sau khi done |

### Ví dụ: Bắt đầu crawl

```bash
curl -X POST http://localhost:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{"source": "category", "max_articles": 20, "include_content": true}'
```

### Ví dụ: Kiểm tra trạng thái

```bash
curl http://localhost:8000/crawl/status/<job_id>
```

## Output

Sau khi crawl xong, dữ liệu được lưu tại `data/raw/`:
- `cafef_news.csv` — dữ liệu dạng bảng
- `cafef_news.jsonl` — mỗi dòng là 1 JSON object

## Cấu trúc project

```
SE365-backend/
├── main.py                  # FastAPI server
├── cafef_news_crawler.py    # Logic crawler CafeF
├── requirements.txt         # Python dependencies
├── .env.example             # Template biến môi trường
├── .gitignore
└── data/
    └── raw/                 # Output CSV/JSONL (tự tạo khi crawl)
```
