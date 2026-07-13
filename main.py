"""
main.py — FastAPI server điều phối CafeF news crawler + FinDec model inference.

Chạy local:
    uvicorn main:app --reload --port 8000

Biến môi trường:
    ALLOWED_ORIGINS  — danh sách CORS, phân cách bằng dấu phẩy
                       Mặc định: http://localhost:5173,http://127.0.0.1:5173
    PORT             — cổng server (mặc định 8000)
    DATA_DIR         — thư mục lưu CSV/JSONL (mặc định ./data/raw)
    SKIP_INFERENCE   — set "1" để bỏ qua bước inference (dev/test nhanh)
"""

from __future__ import annotations

import argparse
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import hàm crawl từ file crawler cùng thư mục
from cafef_news_crawler import (
    NewsArticle,
    build_parser,
    crawl,
    write_csv,
    write_jsonl,
)
from pathlib import Path

# Import model inference (lazy-loaded singleton)
from model_inference import get_inference

SKIP_INFERENCE = os.getenv("SKIP_INFERENCE", "0") == "1"

# ─────────────────────────────────────────────
# Config từ biến môi trường
# ─────────────────────────────────────────────

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

DATA_DIR = os.getenv("DATA_DIR", str(Path(__file__).parent / "data" / "raw"))

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="CafeF Crawler + FinDec Inference API",
    description="API điều khiển crawler CafeF và pipeline phân tích sự kiện tài chính.",
    version="2.0.0",
)


@app.on_event("startup")
async def _preload_models() -> None:
    """Preload FinDec models khi server khởi động (chạy trong thread riêng)."""
    if SKIP_INFERENCE:
        print("[Startup] SKIP_INFERENCE=1 → bỏ qua preload models.")
        return

    def _load() -> None:
        try:
            get_inference().load_models()
        except Exception as exc:  # noqa: BLE001
            print(f"[Startup] Model preload FAILED: {exc}")

    threading.Thread(target=_load, daemon=True, name="model-preload").start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# In-memory job store  (đủ cho demo single-server)
# ─────────────────────────────────────────────

# job_id → {status, progress, articles, error, ...}
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class CrawlConfig(BaseModel):
    source: str = "category"          # "rss" | "category" | "both"
    max_articles: int = 20            # 0 = không giới hạn
    include_content: bool = True      # Có fetch trang chi tiết không
    start_date: Optional[str] = None  # "YYYY-MM-DD" hoặc null
    end_date: Optional[str] = None    # "YYYY-MM-DD" hoặc null


class StartCrawlResponse(BaseModel):
    job_id: str
    message: str


class ProgressInfo(BaseModel):
    current: int
    total: int


class CrawlStatusResponse(BaseModel):
    job_id: str
    status: str           # pending | running | analyzing | done | error
    progress: ProgressInfo
    articles_count: int
    error: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]


class ArticleOut(BaseModel):
    article_id: str
    source: str
    category: str
    title: str
    summary: str
    url: str
    published_date: str
    usable_from_date: str
    author: str
    tags: str
    content_length: int
    crawled_at: str
    crawl_error: str


class EventContextOut(BaseModel):
    who: Optional[str] = None
    what: Optional[str] = None
    when: Optional[str] = None
    where: Optional[str] = None
    why: Optional[str] = None
    how: Optional[str] = None
    tense: Optional[str] = None
    result: Optional[str] = None


class FinancialEventOut(BaseModel):
    id: str
    main_topic: str
    event_type: str
    title: str
    entities_involved: list[str]
    context: dict
    attributes: dict
    evidence_text: str
    confidence: float


class CrawlResultResponse(BaseModel):
    job_id: str
    articles: list[ArticleOut]
    events: list[FinancialEventOut]
    total: int
    events_total: int


# ─────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────

def _article_to_dict(article: NewsArticle) -> dict:
    return {
        "article_id": article.article_id,
        "source": article.source,
        "category": article.category,
        "title": article.title,
        "summary": article.summary,
        "content": article.content,          # cần cho inference pipeline
        "url": article.url,
        "published_date": article.published_date,
        "usable_from_date": article.usable_from_date,
        "author": article.author,
        "tags": article.tags,
        "content_length": article.content_length,
        "crawled_at": article.crawled_at,
        "crawl_error": article.crawl_error,
    }


def _run_crawl_job(job_id: str, config: CrawlConfig) -> None:
    """Chạy crawl + inference trong background thread; cập nhật _jobs liên tục."""

    # Build args từ argparse parser (tái dụng toàn bộ logic CLI)
    argv: list[str] = ["--source", config.source]
    if config.max_articles > 0:
        argv += ["--max-articles", str(config.max_articles)]
    if config.include_content:
        argv += ["--include-content"]
    else:
        argv += ["--no-include-content"]
    if config.start_date:
        argv += ["--start-date", config.start_date]
    if config.end_date:
        argv += ["--end-date", config.end_date]

    parser = build_parser()
    args = parser.parse_args(argv)

    # Đánh dấu running
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        # ── Phase 1: Crawl ──────────────────────────────────────
        articles = _crawl_with_progress(job_id, args)

        # Ghi output files
        out_dir = Path(DATA_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "cafef_news.csv", articles, append=True)
        write_jsonl(out_dir / "cafef_news.jsonl", articles, append=True)

        articles_dicts = [_article_to_dict(a) for a in articles]

        with _jobs_lock:
            _jobs[job_id]["articles"] = articles_dicts
            _jobs[job_id]["progress"] = {"current": len(articles), "total": len(articles)}
            _jobs[job_id]["articles_count"] = len(articles)

        # ── Phase 2: Model Inference ────────────────────────────
        events: list[dict] = []
        if not SKIP_INFERENCE and articles:
            with _jobs_lock:
                _jobs[job_id]["status"] = "analyzing"

            try:
                inference = get_inference()
                events = inference.predict(articles_dicts)
                # Lưu events ra JSONL
                import json as _json
                events_path = out_dir / "cafef_events.jsonl"
                with events_path.open("a", encoding="utf-8") as ef:
                    for ev in events:
                        ef.write(_json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception as inf_exc:  # noqa: BLE001
                print(f"[Inference] WARNING: inference failed for job {job_id}: {inf_exc}")
                # Không crash job — trả về events rỗng

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["events"] = events
            _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()

    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()


def _crawl_with_progress(job_id: str, args: argparse.Namespace) -> list[NewsArticle]:
    """
    Wrapper quanh crawl() để cập nhật progress vào job store.
    Vì crawl() là blocking, ta patch time.sleep để inject progress update.
    """
    import cafef_news_crawler as crawler_module
    import time as _time

    original_sleep = _time.sleep
    _article_counter = {"count": 0, "total": getattr(args, "max_articles", 0)}

    def _patched_sleep(secs: float) -> None:
        _article_counter["count"] += 1
        total = _article_counter["total"] if _article_counter["total"] > 0 else "?"
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = {
                    "current": _article_counter["count"],
                    "total": total if isinstance(total, int) else 0,
                }
                _jobs[job_id]["articles_count"] = _article_counter["count"]
        original_sleep(secs)

    # Patch sleep toàn module để đếm progress
    crawler_module.time.sleep = _patched_sleep  # type: ignore[attr-defined]
    try:
        return crawl(args)
    finally:
        # Restore sleep gốc
        crawler_module.time.sleep = original_sleep  # type: ignore[attr-defined]


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Kiểm tra backend đang chạy."""
    return {"status": "ok", "service": "cafef-crawler"}


@app.post("/crawl", response_model=StartCrawlResponse)
def start_crawl(config: CrawlConfig):
    """Bắt đầu crawl job mới. Trả về job_id ngay lập tức."""
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": {"current": 0, "total": config.max_articles},
            "articles": [],
            "articles_count": 0,
            "events": [],
            "error": None,
            "started_at": None,
            "finished_at": None,
        }

    thread = threading.Thread(
        target=_run_crawl_job,
        args=(job_id, config),
        daemon=True,
        name=f"crawl-{job_id[:8]}",
    )
    thread.start()

    return StartCrawlResponse(
        job_id=job_id,
        message=f"Crawl job {job_id} started.",
    )


@app.get("/crawl/status/{job_id}", response_model=CrawlStatusResponse)
def get_crawl_status(job_id: str):
    """Poll trạng thái crawl job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    progress = job["progress"]
    return CrawlStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=ProgressInfo(
            current=progress.get("current", 0),
            total=progress.get("total", 0),
        ),
        articles_count=job["articles_count"],
        error=job["error"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
    )


@app.get("/crawl/result/{job_id}", response_model=CrawlResultResponse)
def get_crawl_result(job_id: str):
    """Lấy danh sách articles và events sau khi job hoàn tất."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    if job["status"] not in {"done", "error"}:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} chưa hoàn tất (status: {job['status']}).",
        )

    articles_data = job.get("articles", [])
    articles = [ArticleOut(**a) for a in articles_data]

    events_data = job.get("events", [])
    events = [FinancialEventOut(**e) for e in events_data]

    return CrawlResultResponse(
        job_id=job_id,
        articles=articles,
        events=events,
        total=len(articles),
        events_total=len(events),
    )
