#!/usr/bin/env python3
"""Crawl tin tuc chung khoan CafeF de tao du lieu cho cac feature tin tuc.

File nay gom 4 nhom viec chinh:
1. Lay danh sach bai viet tu RSS hoac trang danh muc CafeF.
2. Vao tung trang chi tiet de lay tieu de, tom tat, noi dung, tag, tac gia.
3. Tinh ngay du lieu duoc phep su dung.
4. Ghi ket qua ra CSV va JSONL cho cac buoc xu ly tiep theo.

Crawler duoc viet theo huong lich su: co User-Agent ro rang, co retry,
co delay giua cac request, va khong co gang vuot qua bat ky chan truy cap nao.
"""

from __future__ import annotations  # Cho phep dung type hint moi ma van an toan o runtime.

import argparse  # Doc tham so dong lenh khi chay script.
import csv  # Ghi va doc file CSV.
import json  # Ghi file JSONL tung dong JSON.
import re  # Tim va tach chuoi bang regular expression.
import time  # Sleep giua cac request de crawl lich su.
import xml.etree.ElementTree as ET  # Parse XML cua RSS feed.
from dataclasses import asdict, dataclass  # Khai bao object du lieu va doi sang dict.
from datetime import date, datetime, timedelta, time as dtime, timezone  # Xu ly ngay gio.
from email.utils import parsedate_to_datetime  # Parse ngay dang RFC trong RSS.
from pathlib import Path  # Lam viec voi duong dan file theo kieu object.
from typing import Iterable  # Type hint cho cac ham nhan danh sach/iterator.
from urllib.error import HTTPError, URLError  # Bat loi HTTP/network khi fetch.
from urllib.parse import urljoin, urlparse, urlunparse  # Chuan hoa va tach URL.
from urllib.request import Request, urlopen  # Tao request HTTP va mo URL.

from bs4 import BeautifulSoup  # Parse HTML de trich du lieu trong trang CafeF.


BASE_URL = "https://cafef.vn"  # Domain goc cua CafeF, dung de bien link tuong doi thanh tuyet doi.
DEFAULT_CATEGORY_URL = "https://cafef.vn/thi-truong-chung-khoan.chn"  # Trang danh muc mac dinh.
DEFAULT_RSS_URL = "https://cafef.vn/thi-truong-chung-khoan.rss"  # RSS feed mac dinh.
DEFAULT_TIMELINE_ZONE_ID = "18831"  # ZoneId cua chuyen muc thi truong chung khoan tren CafeF.
DEFAULT_AUTO_TIMELINE_PAGE_LIMIT = 50000  # Bao ve khi auto bam "xem them" theo timeline.
VN_TZ = timezone(timedelta(hours=7))  # Mui gio Viet Nam, CafeF dang timestamp theo gio nay.

FIELDNAMES = [
    # Thu tu cot khi ghi CSV, dong thoi la schema chinh cua dataset raw.
    "article_id",  # ID bai viet tach tu URL CafeF.
    "source",  # Ten nguon chuan hoa, o day luon la cafef.
    "raw_source",  # Noi phat hien bai viet: rss, category, hoac detail.
    "category",  # Chuyen muc bai viet.
    "title",  # Tieu de bai viet.
    "summary",  # Sap o/tom tat ngan.
    "content",  # Noi dung day du sau khi vao trang chi tiet.
    "url",  # URL bai viet da chuan hoa.
    "published_at",  # Thoi diem xuat ban co kem timezone.
    "published_date",  # Ngay xuat ban, tien dung de join voi du lieu gia.
    "usable_from_date",  # Ngay an toan de dua tin vao feature tranh leakage.
    "author",  # Tac gia neu CafeF co hien thi.
    "tags",  # Tag cua bai viet, noi bang dau |.
    "image_url",  # Anh dai dien/og:image.
    "content_length",  # Do dai noi dung, dung kiem tra chat luong crawl.
    "crawled_at",  # Thoi diem script crawl bai viet.
    "crawl_error",  # Loi khi lay trang chi tiet neu co.
]


@dataclass
class NewsArticle:
    """Mot record bai viet trong dataset raw."""

    article_id: str = ""  # ID bai viet lay tu duoi URL.
    source: str = "cafef"  # Nguon du lieu da chuan hoa.
    raw_source: str = ""  # Noi crawler tim thay bai viet luc dau.
    category: str = ""  # Chuyen muc tren CafeF.
    title: str = ""  # Tieu de.
    summary: str = ""  # Tom tat/sapo.
    content: str = ""  # Noi dung chinh cua bai viet.
    url: str = ""  # URL bai viet.
    published_at: str = ""  # Timestamp xuat ban da format ISO.
    published_date: str = ""  # Ngay xuat ban dang YYYY-MM-DD.
    usable_from_date: str = ""  # Ngay duoc phep dung de train/feature.
    author: str = ""  # Ten tac gia.
    tags: str = ""  # Danh sach tag dang chuoi phan tach bang |.
    image_url: str = ""  # URL anh dai dien.
    content_length: int = 0  # So ky tu noi dung sau khi lam sach.
    crawled_at: str = ""  # Timestamp script crawl.
    crawl_error: str = ""  # Thong bao loi neu fetch/parse detail that bai.

    def to_row(self) -> dict[str, str | int]:
        """Doi dataclass thanh dict de ghi CSV/JSONL."""
        return asdict(self)


def clean_text(value: str | None) -> str:
    """Lam sach text: None/empty thanh chuoi rong, nhieu khoang trang thanh mot dau cach."""
    if not value:
        # Tra ve rong de cac ham phia sau khong phai xu ly None.
        return ""
    # Gop moi loai whitespace (xuong dong, tab, nhieu space) thanh 1 space va cat dau/cuoi.
    return re.sub(r"\s+", " ", value).strip()


def normalize_url(url: str, base_url: str = BASE_URL) -> str:
    """Chuyen URL ve dang tuyet doi va bo query/fragment de de dedupe."""
    # Ghep link tuong doi voi domain/trang hien tai.
    absolute = urljoin(base_url, url)
    # Tach URL thanh cac phan scheme, domain, path, query...
    parsed = urlparse(absolute)
    # Giu scheme/domain/path, bo params/query/fragment de URL on dinh hon.
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def is_cafef_article_url(url: str) -> bool:
    """Kiem tra URL co phai bai viet CafeF that su hay khong."""
    parsed = urlparse(url)
    return (
        # Chi chap nhan domain CafeF.
        parsed.netloc.endswith("cafef.vn")
        # Bai viet CafeF thuong ket thuc bang .chn.
        and parsed.path.endswith(".chn")
        # Bai viet chi tiet co ID so o cuoi path, vi du -188260527....chn.
        and re.search(r"-\d+\.chn$", parsed.path) is not None
    )


def extract_article_id(url: str) -> str:
    """Tach ID so cua bai viet tu URL CafeF."""
    # Lay day so nam ngay truoc duoi .chn.
    match = re.search(r"-(\d+)\.chn$", urlparse(url).path)
    # Neu URL khong dung mau thi tra ve rong de khong lam crash crawler.
    return match.group(1) if match else ""


def fetch_url(url: str, timeout: int = 25, retries: int = 3) -> str:
    """Tai noi dung URL voi header giong trinh duyet va retry khi loi tam thoi."""
    headers = {
        # CafeF co the phan biet request thieu User-Agent, nen dung UA pho bien.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        # Chap nhan HTML/XML vi ham nay dung cho ca RSS va trang bai viet.
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        # Uu tien tieng Viet, fallback tieng Anh neu server can.
        "Accept-Language": "vi,en-US;q=0.8,en;q=0.6",
    }
    # Luu loi cuoi cung de thong bao ro neu tat ca lan thu deu that bai.
    last_error: Exception | None = None
    # Thu toi da retries lan; attempt bat dau tu 0 de tinh backoff don gian.
    for attempt in range(retries):
        try:
            # Tao request kem header.
            request = Request(url, headers=headers)
            # Mo URL trong timeout gioi han de crawler khong treo vo han.
            with urlopen(request, timeout=timeout) as response:
                # Lay charset tu response, neu thieu thi dung utf-8.
                charset = response.headers.get_content_charset() or "utf-8"
                # Decode noi dung; ky tu loi duoc thay bang replacement char.
                return response.read().decode(charset, errors="replace")
        except HTTPError as exc:
            # HTTPError co status code, dung de quyet dinh co retry hay khong.
            last_error = exc
            if exc.code < 500 and exc.code not in {408, 429}:
                # Loi 4xx khong phai timeout/rate-limit thuong khong giai quyet bang retry.
                break
        except (URLError, TimeoutError) as exc:
            # Loi mang/timeout thuong co the tam thoi nen cho retry.
            last_error = exc
        if attempt < retries - 1:
            # Backoff tuyen tinh nhe: lan sau cho lau hon lan truoc.
            time.sleep(1.5 * (attempt + 1))
    # Neu het retry van loi, nem RuntimeError de caller ghi vao crawl_error.
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_datetime(value: str | None) -> datetime | None:
    """Parse nhieu dinh dang ngay gio CafeF/RSS ve datetime co timezone VN."""
    # Lam sach input truoc khi parse.
    value = clean_text(value)
    if not value:
        # Khong co timestamp thi tra ve None, caller se ghi chuoi rong.
        return None

    try:
        # Thu truoc dinh dang email/RSS, vi pubDate RSS thuong nam o dang nay.
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            # Neu timestamp khong co timezone, gia dinh la gio Viet Nam.
            parsed = parsed.replace(tzinfo=VN_TZ)
        # Dua ve VN_TZ de cac phep tinh ngay dong nhat.
        return parsed.astimezone(VN_TZ)
    except (TypeError, ValueError, IndexError, OverflowError):
        # Neu khong parse duoc bang RFC/email format thi thu format khac.
        pass

    # ISO format co the ket thuc bang Z, doi thanh +00:00 cho fromisoformat.
    normalized = value.replace("Z", "+00:00")
    try:
        # Thu parse ISO 8601, hay gap trong meta article:published_time.
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            # Gia dinh gio VN neu metadata khong dinh kem timezone.
            parsed = parsed.replace(tzinfo=VN_TZ)
        # Chuan hoa ve gio VN.
        return parsed.astimezone(VN_TZ)
    except ValueError:
        # Neu van khong dung ISO, tiep tuc thu regex CafeF hien thi.
        pass

    # Bat cac chuoi kieu dd/mm/yyyy hh:mm hoac dd-mm-yyyy hh:mm.
    match = re.search(
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{4}).*?(\d{1,2}):(\d{2})", value
    )
    if match:
        # Chuyen tung nhom regex thanh so nguyen.
        day, month, year, hour, minute = map(int, match.groups())
        # Tao datetime theo gio VN.
        return datetime(year, month, day, hour, minute, tzinfo=VN_TZ)
    # Khong nhan dien duoc thi tra ve None.
    return None


def format_dt(dt: datetime | None) -> str:
    """Format datetime thanh ISO string theo phut, hoac rong neu thieu."""
    if not dt:
        # Giu schema output on dinh bang chuoi rong thay vi None.
        return ""
    # Doi ve gio VN va bo giay de output gon hon.
    return dt.astimezone(VN_TZ).isoformat(timespec="minutes")


def next_weekday(value: date) -> date:
    """Dich ngay len ngay lam viec tiep theo neu roi vao Thu bay/Chu nhat."""
    while value.weekday() >= 5:
        # weekday: 5 la Thu bay, 6 la Chu nhat.
        value += timedelta(days=1)
    # Tra ve ngay dau tien khong phai cuoi tuan.
    return value


def usable_from_date(dt: datetime | None, cutoff: dtime = dtime(14, 45)) -> str:
    """Approximate when news should become usable for daily stock features.

    CafeF timestamps are Vietnam local time. If an article is published after
    the stock market close, shift it to the next weekday to reduce leakage when
    merging with EOD price data. This does not account for exchange holidays.
    """

    if not dt:
        # Khong co ngay xuat ban thi khong tinh duoc ngay usable.
        return ""
    # Bao dam dang tinh theo gio VN.
    local_dt = dt.astimezone(VN_TZ)
    # Mac dinh tin dung duoc trong chinh ngay xuat ban.
    usable = local_dt.date()
    if usable.weekday() >= 5:
        # Tin ra cuoi tuan thi day sang ngay lam viec tiep theo.
        usable = next_weekday(usable)
    elif local_dt.time() >= cutoff:
        # Tin ra sau gio dong cua thi chi dung tu ngay giao dich ke tiep.
        usable = next_weekday(usable + timedelta(days=1))
    # Ghi ra YYYY-MM-DD de join voi du lieu daily.
    return usable.isoformat()


def parse_date_arg(value: str | None) -> date | None:
    """Parse CLI date YYYY-MM-DD."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ngay khong hop le: {value!r}. Dung dinh dang YYYY-MM-DD."
        ) from exc


def validate_date_range(start_date: date | None, end_date: date | None) -> None:
    """Bao loi neu khoang ngay bi nguoc."""
    if start_date and end_date and end_date < start_date:
        raise ValueError("--end-date phai lon hon hoac bang --start-date")


def article_published_date(article: NewsArticle) -> date | None:
    """Lay published_date cua article ve date object."""
    if not article.published_date:
        return None
    try:
        return date.fromisoformat(article.published_date)
    except ValueError:
        return None


def in_date_range(
    article: NewsArticle,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    """Kiem tra bai viet co nam trong khoang published_date hay khong."""
    if not start_date and not end_date:
        return True

    published = article_published_date(article)
    if not published:
        return False
    if start_date and published < start_date:
        return False
    if end_date and published > end_date:
        return False
    return True


def page_is_before_start(
    articles: list[NewsArticle],
    start_date: date | None,
) -> bool:
    """True neu toan bo trang co ngay dang cu hon start_date."""
    if not start_date or not articles:
        return False
    dates = [article_published_date(article) for article in articles]
    return all(published is not None and published < start_date for published in dates)


def seed_may_be_in_date_range(
    article: NewsArticle,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    """Loc seed som, nhung giu bai thieu date de detail co co hoi bo sung."""
    if not start_date and not end_date:
        return True
    return article_published_date(article) is None or in_date_range(article, start_date, end_date)


def build_timeline_url(zone_id: str, page: int) -> str:
    """URL phan trang timeline cua CafeF category."""
    return f"{BASE_URL}/timelinelist/{zone_id}/{page}.chn"


def parse_timeline_page(args: argparse.Namespace, page: int) -> list[NewsArticle]:
    """Fetch va parse mot trang timeline CafeF."""
    timeline_url = build_timeline_url(args.timeline_zone_id, page)
    return parse_category_items(
        fetch_url(timeline_url, timeout=args.timeout),
        args.category_url,
    )


def page_date_bounds(articles: list[NewsArticle]) -> tuple[date | None, date | None]:
    """Tra ve (newest_date, oldest_date) cua mot trang timeline."""
    dates = [
        published
        for article in articles
        if (published := article_published_date(article)) is not None
    ]
    if not dates:
        return None, None
    return max(dates), min(dates)


def find_timeline_start_page(args: argparse.Namespace) -> int | None:
    """Tim page timeline dau tien co the chua bai <= end_date.

    CafeF timeline sap xep moi -> cu. Neu end_date nam rat xa hien tai, binary
    search giup nhay thang toi vung page can crawl thay vi bam "xem them" tu page 2.
    """
    if not args.end_date or args.timeline_pages > 0:
        return 2

    low = 2
    high = max(args.auto_timeline_page_limit, 2)
    best: int | None = None

    while low <= high:
        mid = (low + high) // 2
        try:
            page_articles = parse_timeline_page(args, mid)
        except RuntimeError:
            high = mid - 1
            continue
        if not page_articles:
            high = mid - 1
            continue

        _, oldest = page_date_bounds(page_articles)
        if oldest is None:
            return 2
        if oldest > args.end_date:
            low = mid + 1
        else:
            best = mid
            high = mid - 1

    return best


def count_candidate_articles(
    articles: list[NewsArticle],
    start_date: date | None,
    end_date: date | None,
) -> int:
    """Dem so seed co the nam trong range sau dedupe."""
    return len(
        [
            article
            for article in dedupe_articles(articles)
            if seed_may_be_in_date_range(article, start_date, end_date)
        ]
    )


def should_fetch_more_timeline(args: argparse.Namespace, seeds: list[NewsArticle]) -> bool:
    """Quyet dinh co can auto bam 'xem them' tren timeline CafeF khong."""
    if args.timeline_pages > 0:
        return True
    if args.max_articles > 0:
        return (
            count_candidate_articles(seeds, args.start_date, args.end_date)
            < args.max_articles
        )
    if args.start_date:
        return True
    return False


def parse_description_html(description: str) -> tuple[str, str]:
    """Tach tom tat va anh dai dien tu HTML nam trong truong description cua RSS."""
    # RSS description co the chua HTML nho, nen parse bang BeautifulSoup.
    soup = BeautifulSoup(description or "", "html.parser")
    # Mac dinh khong co anh.
    image = ""
    # Lay anh dau tien neu RSS nhung img vao description.
    img = soup.find("img")
    if img and img.get("src"):
        # Chuan hoa URL anh de luu duoc truc tiep.
        image = normalize_url(img["src"])
    # Xoa cac phan khong phai text truoc khi lay tom tat.
    for element in soup.select("img, script, style"):
        element.decompose()
    # Tra ve text da lam sach va URL anh.
    return clean_text(soup.get_text(" ", strip=True)), image


def parse_rss_items(rss_xml: str) -> list[NewsArticle]:
    """Parse RSS XML thanh danh sach NewsArticle seed."""
    # Chuyen chuoi XML thanh cay element.
    root = ET.fromstring(rss_xml)
    # RSS hop le thuong co node channel chua cac item.
    channel = root.find("channel")
    if channel is None:
        # Khong co channel thi xem nhu feed rong.
        return []

    # Danh sach bai viet lay duoc tu feed.
    articles: list[NewsArticle] = []
    # Lap qua tung item trong RSS.
    for item in channel.findall("item"):
        # Lay va lam sach title bai viet.
        title = clean_text(item.findtext("title"))
        # Chuan hoa link bai viet.
        link = normalize_url(item.findtext("link") or "")
        if not link or not is_cafef_article_url(link):
            # Bo qua item khong co link hoac khong phai bai chi tiet CafeF.
            continue

        # RSS description thuong gom sapo va anh.
        summary, image_url = parse_description_html(item.findtext("description") or "")
        # Parse thoi diem xuat ban tu pubDate.
        published_dt = parse_datetime(item.findtext("pubDate"))
        # Tao record seed: chua co content day du, se merge voi detail sau.
        articles.append(
            NewsArticle(
                article_id=extract_article_id(link),  # ID tu URL.
                raw_source="rss",  # Danh dau bai nay duoc tim tu RSS.
                category=clean_text(channel.findtext("title")).replace(" | cafef", ""),  # Ten channel.
                title=title,  # Tieu de RSS.
                summary=summary,  # Tom tat tu description.
                url=link,  # URL bai viet.
                published_at=format_dt(published_dt),  # Timestamp ISO.
                published_date=published_dt.date().isoformat() if published_dt else "",  # Ngay xuat ban.
                usable_from_date=usable_from_date(published_dt),  # Ngay an toan cho feature.
                image_url=image_url,  # Anh dai dien neu co.
            )
        )
    # Tra ve cac seed lay duoc.
    return articles


def parse_category_items(html_doc: str, category_url: str) -> list[NewsArticle]:
    """Parse trang danh muc CafeF thanh danh sach NewsArticle seed."""
    # Parse HTML cua trang danh muc.
    soup = BeautifulSoup(html_doc, "html.parser")
    # Cac selector tuong ung voi vi tri bai noi bat va bai trong list tren CafeF.
    containers = soup.select(".firstitem, .cate-hl-row2 .big, .tlitem.box-category-item")
    # Noi luu cac bai viet hop le.
    articles: list[NewsArticle] = []
    # Lap qua tung khoi bai viet tren trang danh muc.
    for container in containers:
        # Link co the nam trong h1/h2/h3 hoac anh dai dien.
        link_element = container.select_one("h1 a[href], h2 a[href], h3 a[href], a.avatar[href]")
        if not link_element:
            # Khoi khong co link thi bo qua.
            continue
        # Chuan hoa URL theo category_url vi link tren page co the la relative.
        url = normalize_url(link_element.get("href") or "", category_url)
        if not is_cafef_article_url(url):
            # Bo qua link khong phai bai viet chi tiet.
            continue

        # Uu tien lay title tu heading thay vi avatar title.
        title_element = container.select_one("h1 a[href], h2 a[href], h3 a[href]")
        # Neu co heading thi lay text heading, neu khong thi fallback sang attr title cua link.
        title = clean_text(
            title_element.get_text(" ", strip=True)
            if title_element
            else link_element.get("title") or ""
        )
        # Lay sapo neu khoi danh muc co class .sapo.
        summary = clean_text(
            container.select_one(".sapo").get_text(" ", strip=True)
            if container.select_one(".sapo")
            else ""
        )
        # CafeF co the dat timestamp trong title cua .time-ago hoac data-time cua .time.
        time_element = container.select_one(".time-ago[title], .time[data-time]")
        # Chuoi ngay gio thô chua parse.
        published_raw = ""
        if time_element:
            # Uu tien title, neu khong co thi lay data-time.
            published_raw = time_element.get("title") or time_element.get("data-time") or ""
        # Parse thoi gian ve datetime.
        published_dt = parse_datetime(published_raw)
        # Lay anh dai dien dau tien trong container.
        image = container.select_one("img[src]")

        # Tao record seed tu trang danh muc.
        articles.append(
            NewsArticle(
                article_id=extract_article_id(url),  # ID tu URL.
                raw_source="category",  # Danh dau lay tu trang danh muc.
                category="Thi truong chung khoan",  # Ten danh muc crawl.
                title=title,  # Tieu de bai.
                summary=summary,  # Sapo neu co.
                url=url,  # URL bai viet.
                published_at=format_dt(published_dt),  # Timestamp ISO neu parse duoc.
                published_date=published_dt.date().isoformat() if published_dt else "",  # Ngay xuat ban.
                usable_from_date=usable_from_date(published_dt),  # Ngay feature duoc phep dung.
                image_url=normalize_url(image["src"]) if image and image.get("src") else "",  # Anh dai dien.
            )
        )
    # Tra ve cac seed tu trang danh muc.
    return articles


def get_meta(soup: BeautifulSoup, key: str) -> str:
    """Lay noi dung meta tag theo property hoac name."""
    # Meta OpenGraph dung property, meta thuong dung name; thu ca hai.
    element = soup.find("meta", attrs={"property": key}) or soup.find(
        "meta", attrs={"name": key}
    )
    # Neu tim thay content thi lam sach, nguoc lai tra ve rong.
    return clean_text(element.get("content")) if element and element.get("content") else ""


def first_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str:
    """Tra ve text dau tien khong rong trong danh sach CSS selector."""
    # Thu tung selector theo thu tu uu tien.
    for selector in selectors:
        # Tim element dau tien khop selector.
        element = soup.select_one(selector)
        if element:
            # Lay text da strip va gop whitespace.
            value = clean_text(element.get_text(" ", strip=True))
            if value:
                # Tra ve ngay khi co gia tri hop le.
                return value
    # Khong selector nao co text.
    return ""


def extract_content(soup: BeautifulSoup) -> str:
    """Trich noi dung chinh cua bai viet CafeF, loai quang cao/box lien quan."""
    # Selector chinh cua noi dung tren layout CafeF moi.
    content = soup.select_one("#mainContent [data-role='content']")
    # Fallback cho cac layout/cu phien ban khac.
    content = content or soup.select_one("#mainContent .detail-content")
    # Fallback layout co data-role nam truc tiep tren .detail-content.
    content = content or soup.select_one(".detail-content[data-role='content']")
    # Fallback layout cu hon.
    content = content or soup.select_one(".contentdetail")
    if not content:
        # Khong tim thay vung noi dung thi tra ve rong.
        return ""

    # Cac selector can xoa de noi dung sach hon.
    noise_selectors = [
        "script",  # JavaScript khong phai noi dung bai.
        "style",  # CSS khong phai noi dung bai.
        "iframe",  # Embed video/ads.
        "noscript",  # Noi dung fallback trinh duyet.
        "form",  # Form dang ky/tuong tac.
        "figure",  # Anh/figure thuong lam lap text.
        "figcaption",  # Chu thich anh, thuong nhieu noise.
        ".PhotoCMS_Caption",  # Caption anh cua CafeF.
        ".tindnd",  # Box tin doc nhieu/doc nhanh.
        ".tinlienquan",  # Box tin lien quan.
        ".relatednews",  # Box tin lien quan layout khac.
        ".VCSortableInPreviewMode[type='RelatedNewsBox']",  # Box related news tu CMS.
        ".c-banner",  # Banner quang cao.
        ".h-show-pc",  # Noi dung hien rieng desktop, thuong la ads/widget.
        ".h-show-mobile",  # Noi dung hien rieng mobile, thuong la ads/widget.
        ".adsbygoogle",  # Quang cao Google.
    ]
    # Xoa tat ca node noise khoi vung content.
    for element in content.select(",".join(noise_selectors)):
        element.decompose()

    # Noi luu tung doan text sau khi lam sach.
    paragraphs: list[str] = []
    # Set de bo cac doan bi lap.
    seen: set[str] = set()
    # Lay cac the noi dung thuong gap trong bai viet.
    for element in content.find_all(["p", "li", "h2", "h3"], recursive=True):
        # Lay text cua tung paragraph/list/heading.
        text = clean_text(element.get_text(" ", strip=True))
        if not text or text in seen:
            # Bo qua doan rong hoac trung lap.
            continue
        if text.upper() in {"TIN MOI", "TIN LIEN QUAN"}:
            # Bo cac heading noise.
            continue
        # Danh dau da gap doan nay.
        seen.add(text)
        # Giu lai doan hop le.
        paragraphs.append(text)

    if paragraphs:
        # Noi cac doan bang newline de giu cau truc bai viet.
        return "\n".join(paragraphs)
    # Fallback: neu khong tim thay p/li/h2/h3 thi lay toan bo text cua content.
    return clean_text(content.get_text(" ", strip=True))


def parse_article_page(html_doc: str, url: str) -> NewsArticle:
    """Parse trang chi tiet CafeF thanh NewsArticle day du."""
    # Parse HTML chi tiet.
    soup = BeautifulSoup(html_doc, "html.parser")

    # Lay title tu h1 uu tien, sau do fallback sang og:title.
    title = first_text(soup, ["h1.title[data-role='title']", "h1.title", "h1"])
    title = title or get_meta(soup, "og:title")

    # Lay sapo/tom tat tu selector CafeF, fallback sang og:description.
    summary = first_text(soup, ["h2.sapo[data-role='sapo']", ".sapo[data-role='sapo']"])
    summary = summary or get_meta(soup, "og:description")

    # Lay thoi gian xuat ban tu meta article:published_time truoc.
    published_dt = parse_datetime(get_meta(soup, "article:published_time"))
    if not published_dt:
        # Neu meta thieu, lay text timestamp hien tren page.
        published_dt = parse_datetime(first_text(soup, [".pdate[data-role='publishdate']", ".pdate"]))

    # Lay danh sach tag tu khu vuc tagdetail.
    tags = [
        clean_text(tag.get_text(" ", strip=True)).strip(" ,")
        for tag in soup.select(".tagdetail .row2 a")
    ]
    # Bo tag rong.
    tags = [tag for tag in tags if tag]

    # Lay chuyen muc neu co tren trang chi tiet.
    category = first_text(soup, ["[data-role='cate-name']", ".category-page__name.cat"])
    # Lay tac gia tu meta hoac selector tren giao dien.
    author = get_meta(soup, "article:author") or first_text(
        soup, ["[data-role='author']", ".dateandcat .author", ".t-contentdetail .author"]
    )
    # Loai dau | thuong dinh kem ten tac gia tren CafeF.
    author = author.replace("|", "").strip()

    # Trich noi dung bai viet da lam sach.
    content = extract_content(soup)
    # Tra ve record chi tiet, se duoc merge voi seed neu can.
    return NewsArticle(
        article_id=extract_article_id(url),  # ID tu URL detail.
        raw_source="detail",  # Danh dau record nay parse tu trang chi tiet.
        category=category,  # Chuyen muc detail.
        title=title,  # Tieu de detail.
        summary=summary,  # Sapo detail.
        content=content,  # Noi dung day du.
        url=url,  # URL detail.
        published_at=format_dt(published_dt),  # Timestamp da format.
        published_date=published_dt.date().isoformat() if published_dt else "",  # Ngay xuat ban.
        usable_from_date=usable_from_date(published_dt),  # Ngay usable.
        author=author,  # Tac gia.
        tags="|".join(tags),  # Noi tag bang | de de ghi CSV.
        image_url=get_meta(soup, "og:image"),  # Anh dai dien tu OpenGraph.
        content_length=len(content),  # So ky tu content.
    )


def merge_detail(seed: NewsArticle, detail: NewsArticle) -> NewsArticle:
    """Gop du lieu detail vao seed, giu gia tri seed neu detail bi rong."""
    # Doi seed sang dict de cap nhat tung field.
    merged = seed.to_row()
    # Doi detail sang dict de lap qua field.
    detail_row = detail.to_row()
    # Field nao detail co gia tri thi thay vao seed.
    for key, value in detail_row.items():
        if value not in {"", 0}:
            merged[key] = value
    # raw_source phai la noi phat hien ban dau, khong phai detail.
    merged["raw_source"] = seed.raw_source
    # Tao lai dataclass tu dict da merge.
    return NewsArticle(**merged)


def dedupe_articles(articles: Iterable[NewsArticle]) -> list[NewsArticle]:
    """Bo cac bai trung URL nhung giu thu tu xuat hien dau tien."""
    # Set URL da gap.
    seen: set[str] = set()
    # Ket qua sau khi loai trung.
    result: list[NewsArticle] = []
    # Duyet tung bai theo thu tu nguon tra ve.
    for article in articles:
        if article.url in seen:
            # Neu URL da gap thi bo qua.
            continue
        # Ghi nhan URL moi.
        seen.add(article.url)
        # Giu bai dau tien ung voi URL nay.
        result.append(article)
    # Tra ve danh sach da dedupe.
    return result


def read_existing_csv(path: Path) -> list[dict[str, str]]:
    """Doc CSV cu neu co, dung khi --append de merge khong trung URL."""
    if not path.exists():
        # File chua ton tai thi xem nhu khong co dong cu.
        return []
    # utf-8-sig xu ly BOM neu Excel/Windows them vao dau file.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        # DictReader tra moi dong thanh dict theo header.
        return list(csv.DictReader(file))


def project_output_rows(rows: list[dict]) -> list[dict]:
    """Giu dung schema CafeF hien tai, bo field cu/khong dung nua."""
    return [{field: row.get(field, "") for field in FIELDNAMES} for row in rows]


def write_csv(path: Path, articles: list[NewsArticle], append: bool = False) -> None:
    """Ghi danh sach bai viet ra CSV, co the merge voi file cu khi append=True."""
    # Doi moi article thanh dict.
    rows = [article.to_row() for article in articles]
    if append and path.exists():
        # Neu append, doc file hien co truoc.
        existing = read_existing_csv(path)
        # Dung URL de tranh ghi trung bai.
        seen_urls = {row.get("url", "") for row in existing}
        # Giu dong cu, them dong moi chua co URL trong file cu.
        rows = existing + [row for row in rows if row.get("url", "") not in seen_urls]
    rows = project_output_rows(rows)

    # Tao thu muc output neu chua co.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi CSV voi BOM utf-8-sig de mo bang Excel de hon.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        # extrasaction ignore giup bo field ngoai schema neu co.
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="ignore")
        # Ghi header truoc.
        writer.writeheader()
        # Ghi toan bo rows.
        writer.writerows(rows)


def write_jsonl(path: Path, articles: list[NewsArticle], append: bool = False) -> None:
    """Ghi danh sach bai viet ra JSONL, moi dong la mot JSON object."""
    # Doi moi article thanh dict.
    rows = [article.to_row() for article in articles]
    if append and path.exists():
        # Doc cac dong JSONL cu neu append.
        existing_rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as file:
            # Moi dong khong rong la mot JSON object.
            for line in file:
                if line.strip():
                    existing_rows.append(json.loads(line))
        # Dedupe bang URL.
        seen_urls = {row.get("url", "") for row in existing_rows}
        # Giu dong cu va them dong moi chua trung URL.
        rows = existing_rows + [row for row in rows if row.get("url", "") not in seen_urls]
    rows = project_output_rows(rows)

    # Tao thu muc output neu chua co.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi JSONL bang UTF-8 thuong.
    with path.open("w", encoding="utf-8") as file:
        # Ghi tung dict thanh mot dong JSON.
        for row in rows:
            # ensure_ascii=False de giu tieng Viet doc duoc.
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def crawl(args: argparse.Namespace) -> list[NewsArticle]:
    """Dieu phoi toan bo qua trinh crawl theo tham so dong lenh."""
    # Seed la cac bai viet moi co metadata tu RSS/category, chua chac co content full.
    seeds: list[NewsArticle] = []
    if args.source in {"rss", "both"}:
        # Lay seed tu RSS neu user chon rss hoac both.
        seeds.extend(parse_rss_items(fetch_url(args.rss_url, timeout=args.timeout)))
    if args.source in {"category", "both"}:
        # Lay seed tu trang danh muc neu user chon category hoac both.
        seeds.extend(
            parse_category_items(
                fetch_url(args.category_url, timeout=args.timeout), args.category_url
            )
        )
        max_timeline_page = (
            args.timeline_pages
            if args.timeline_pages > 0
            else args.auto_timeline_page_limit
        )
        page = find_timeline_start_page(args)
        if page is None:
            print(
                "[warn] Auto timeline limit does not reach end-date. "
                "Increase --auto-timeline-page-limit or set --timeline-pages."
            )
            page = max_timeline_page + 1
        while page <= max_timeline_page and should_fetch_more_timeline(args, seeds):
            # CafeF load them bai category qua /timelinelist/{zoneId}/{page}.chn.
            try:
                page_articles = parse_timeline_page(args, page)
            except RuntimeError as exc:
                print(f"[warn] Timeline page {page} fetch failed: {exc}")
                break
            if not page_articles:
                break
            if page_is_before_start(page_articles, args.start_date):
                break
            seeds.extend(page_articles)
            time.sleep(max(args.timeline_delay, 0))
            page += 1

    # Loai bai trung URL giua RSS va category.
    articles = dedupe_articles(seeds)
    if args.start_date or args.end_date:
        # Loc seed truoc khi fetch detail de tranh tai cac bai ngoai khoang ngay.
        articles = [
            article
            for article in articles
            if seed_may_be_in_date_range(article, args.start_date, args.end_date)
        ]
    if args.max_articles > 0:
        # Gioi han so bai can crawl neu user truyen --max-articles.
        articles = articles[: args.max_articles]

    # Mot timestamp chung cho tat ca bai trong lan crawl nay.
    now = datetime.now(VN_TZ).isoformat(timespec="seconds")
    # Ket qua cuoi cung sau khi fetch detail.
    result: list[NewsArticle] = []
    # Lap qua tung seed de lay them chi tiet.
    for index, seed in enumerate(articles, start=1):
        # Mac dinh dung seed neu khong lay detail hoac detail loi.
        article = seed
        if args.include_content:
            # Neu include_content bat, vao tung trang bai viet de lay full text.
            try:
                # Tai HTML trang chi tiet.
                detail_html = fetch_url(seed.url, timeout=args.timeout)
                # Parse HTML detail thanh NewsArticle.
                detail = parse_article_page(detail_html, seed.url)
                # Merge detail vao seed de khong mat metadata tu discovery.
                article = merge_detail(seed, detail)
            except Exception as exc:  # noqa: BLE001 - keep partial rows for analysis.
                # Ghi loi vao row, van giu bai seed de phan tich sau.
                article.crawl_error = str(exc)
        # Gan timestamp crawl.
        article.crawled_at = now
        # Cap nhat lai do dai content sau khi merge.
        article.content_length = len(article.content)
        if not in_date_range(article, args.start_date, args.end_date):
            # Khi co filter ngay, chi giu bai nam trong published_date yeu cau.
            continue
        # Them bai hop le vao ket qua.
        result.append(article)
        if index < len(articles):
            # Nghi giua cac request de giam tai cho website.
            time.sleep(max(args.delay, 0))
    # Tra ve danh sach bai da xu ly.
    return result


def build_parser() -> argparse.ArgumentParser:
    """Tao parser cho cac tham so dong lenh cua script."""
    # Khoi tao argparse voi mo ta ngan.
    parser = argparse.ArgumentParser(
        description="Crawl CafeF stock-market news into CSV/JSONL."
    )
    # URL trang danh muc CafeF.
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL)
    # URL RSS feed CafeF.
    parser.add_argument("--rss-url", default=DEFAULT_RSS_URL)
    # Nguon discovery: RSS, HTML category, hoac ca hai.
    parser.add_argument(
        "--source",
        choices=["rss", "category", "both"],
        default="category",
        help="Use RSS for discovery, category HTML, or both.",
    )
    parser.add_argument(
        "--timeline-pages",
        type=int,
        default=0,
        help=(
            "So trang timeline category can lay. 0 = auto khi can lay them; "
            "page >=2 dung /timelinelist/{zoneId}/{page}.chn."
        ),
    )
    parser.add_argument(
        "--timeline-zone-id",
        default=DEFAULT_TIMELINE_ZONE_ID,
        help="ZoneId CafeF cua chuyen muc. Mac dinh 18831 = thi truong chung khoan.",
    )
    parser.add_argument(
        "--auto-timeline-page-limit",
        type=int,
        default=DEFAULT_AUTO_TIMELINE_PAGE_LIMIT,
        help="Gioi han page khi --timeline-pages=0 auto xem them.",
    )
    # So bai toi da lay trong mot lan chay; 0 = khong gioi han sau khi discovery.
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="So bai toi da sau discovery; 0 = khong gioi han.",
    )
    # Delay giua cac request detail.
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--timeline-delay",
        type=float,
        default=0.2,
        help="Delay giua cac request timeline xem them.",
    )
    # Timeout moi request.
    parser.add_argument("--timeout", type=int, default=25)
    # Co lay full content tu trang detail hay khong.
    parser.add_argument(
        "--include-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch article detail pages for full text.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date_arg,
        default=None,
        help="Ngay bat dau published_date, dinh dang YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date_arg,
        default=None,
        help="Ngay ket thuc published_date, inclusive, dinh dang YYYY-MM-DD.",
    )
    # Duong dan output CSV.
    parser.add_argument("--output", default="data/raw/cafef_news.csv")
    # Duong dan output JSONL.
    parser.add_argument("--jsonl-output", default="data/raw/cafef_news.jsonl")
    # Co merge voi output cu hay ghi moi.
    parser.add_argument("--append", action="store_true", help="Merge with existing output.")
    # Tra parser cho main su dung.
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point cua script; tra ve exit code 0 neu crawl khong loi."""
    # Tao parser CLI.
    parser = build_parser()
    # Parse tham so tu argv truyen vao hoac sys.argv neu argv=None.
    args = parser.parse_args(argv)
    try:
        validate_date_range(args.start_date, args.end_date)
    except ValueError as exc:
        parser.error(str(exc))
    # Chay crawl theo config da parse.
    articles = crawl(args)
    if args.output:
        # Ghi CSV neu user khong tat output nay.
        write_csv(Path(args.output), articles, append=args.append)
    if args.jsonl_output:
        # Ghi JSONL neu user khong tat output nay.
        write_jsonl(Path(args.jsonl_output), articles, append=args.append)

    # Dem so bai bi loi khi fetch/parse detail.
    error_count = sum(1 for article in articles if article.crawl_error)
    # In tong ket de nguoi chay biet ket qua.
    print(
        f"Crawled {len(articles)} CafeF articles "
        f"({error_count} with errors). Output: {args.output}, {args.jsonl_output}"
    )
    # Neu co loi detail thi exit 1 de pipeline/CI biet co van de.
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    # Khi chay file truc tiep: goi main va dung exit code main tra ve.
    raise SystemExit(main())
