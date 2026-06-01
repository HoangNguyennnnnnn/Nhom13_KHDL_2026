"""Scrape TGDD smartphone products and reviews - v3 (correct selectors).

Verified selectors from live TGDD HTML (2026-05-31):
  - Product cards  : li.item.__cate_42 (smartphones), etc.
  - Product name   : p.product-title  OR  a.main-contain[data-name]
  - Brand          : a.main-contain[data-brand]
  - Price          : .box-price .price  /  .price-old
  - Reviews        : li.par  (inside .comment-list)
  - Review text    : .cmt-txt
  - Review stars   : .cmt-top-star  (class encodes star count)
  - Review date    : .cmt-top-name + sibling time/span
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
import uuid
import sys
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except ImportError:
    USE_WDM = False

# ── Config ────────────────────────────────────────────────────────────────────

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    encoding="utf-8",
    errors="replace",
)
log = logging.getLogger(__name__)

RAW_DIR = Path("data-project/raw")

CATEGORIES = [
    ("smartphone", "https://www.thegioididong.com/dtdd#c=42&o=7&pi=9"),
    ("tablet",     "https://www.thegioididong.com/may-tinh-bang"),
    ("laptop",     "https://www.thegioididong.com/laptop"),
]

SCROLL_SLEEP  = 1.2
PAGE_SLEEP    = 2.0
RETRY_SLEEP   = 3.0
MAX_RETRIES   = 3


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ProductRow:
    Product_ID:        str
    Name:              str
    Brand:             str
    Category:          str
    Original_Price:    str
    Discounted_Price:  str
    Delivery_Options:  str
    Sales_Volume:      str
    Avg_Star_Rating:   str
    Total_Reviews:     str
    Product_URL:       str


@dataclass
class ReviewRow:
    Review_ID:          str
    Product_ID:         str
    Product_Name:       str
    Review_Date:        str
    Star_Rating:        str
    Review_Text:        str
    Helpfulness_Count:  str = "0"
    Support_Contacted:  str = "0"
    Language_Code:      str = "vi"


# ── Driver ────────────────────────────────────────────────────────────────────

def build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    opts.page_load_strategy = "eager"
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1440,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    if USE_WDM:
        svc = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=svc, options=opts)
    return webdriver.Chrome(options=opts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_text(node, selector: str, default: str = "") -> str:
    found = node.select_one(selector)
    return " ".join(found.get_text(" ", strip=True).split()) if found else default


def numeric_text(text: str) -> str:
    return re.sub(r"[^\d]", "", text or "")


def parse_compact_count(text: str) -> str:
    """Parse TGDD compact counters like '243,4k' -> '243400'."""
    s = (text or "").strip().lower()
    if not s:
        return ""

    m = re.search(r"([\d]+(?:[.,]\d+)?)\s*([km])?", s)
    if not m:
        return numeric_text(s)

    raw_num = m.group(1).replace(",", ".")
    suffix = m.group(2)
    try:
        value = float(raw_num)
    except ValueError:
        return numeric_text(s)

    if suffix == "k":
        return str(int(value * 1000))
    if suffix == "m":
        return str(int(value * 1000000))
    return numeric_text(raw_num)


def get_page(driver: webdriver.Chrome, url: str, wait_css: str = "body", timeout: int = 15) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_css))
            )
            return True
        except (TimeoutException, WebDriverException) as exc:
            log.warning(f"    Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    return False


def scroll_catalog(driver: webdriver.Chrome, max_scrolls: int = 80) -> None:
    """Scroll listing page and click 'Xem them' until product cards stop increasing."""
    last_h = 0
    last_count = 0
    stable_rounds = 0

    # Dismiss location popups and overlays safely without deleting the BODY element
    try:
        driver.execute_script("""
            document.querySelectorAll("div[class*='location'], div[class*='popup'], div[id*='location'], div[id*='popup']").forEach(el => {
                if (el.tagName !== 'BODY' && el.textContent && (el.textContent.includes("tỉnh thành") || el.textContent.includes("Địa chỉ"))) {
                    el.remove();
                }
            });
            document.querySelectorAll("div[class*='backdrop'], div[class*='overlay']").forEach(el => {
                if (el.tagName !== 'BODY') el.remove();
            });
            if (document.body) document.body.style.overflow = 'auto';
        """)
    except Exception:
        pass

    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight || (document.body ? document.body.scrollHeight : 0));")
        time.sleep(SCROLL_SLEEP)

        h = driver.execute_script("return document.documentElement.scrollHeight || (document.body ? document.body.scrollHeight : 0)")
        cur_count = len(driver.find_elements(By.CSS_SELECTOR, "li.item"))

        clicked = False
        try:
            # Query ONLY specific show-more classes to prevent overhead
            for btn in driver.find_elements(By.CSS_SELECTOR, ".btn-show-more, .viewmore, [class*='show-more']"):
                btn_text = (btn.get_attribute("textContent") or btn.text or "").strip().lower()
                if "xem thêm" in btn_text or "xem them" in btn_text:
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", btn)
                    clicked = True
                    time.sleep(SCROLL_SLEEP)
                    break
        except Exception:
            pass

        if cur_count > last_count or clicked:
            stable_rounds = 0
            if cur_count > last_count:
                last_count = cur_count
        else:
            stable_rounds += 1

        if stable_rounds >= 6 and h == last_h:
            break

        last_h = h


def load_more_reviews(driver: webdriver.Chrome, max_clicks: int = 20) -> None:
    """Click review "load more" controls repeatedly (selectors vary by TGDD build)."""
    for _ in range(max_clicks):
        clicked = False
        for sel in [
            ".btn-cmt-larger10",
            ".btn-add",
            "button[class*='more']",
            "a[class*='more']",
            "[class*='comment'][class*='more']",
            "[class*='cmt'][class*='more']",
        ]:
            try:
                btn = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                txt = btn.text.strip().lower()
                if txt and ("xem" not in txt and "thêm" not in txt and "them" not in txt and "review" not in txt):
                    continue
                driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(1.2)
                clicked = True
                break
            except Exception:
                continue

        if clicked:
            continue

        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(translate(normalize-space(text()), 'THÊMXEM', 'thêmxem'), 'xem') and (contains(translate(normalize-space(text()), 'THÊM', 'thêm'), 'thêm') or contains(translate(normalize-space(text()), 'THEM', 'them'), 'them'))]"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(1.5)
        except (TimeoutException, NoSuchElementException):
            break
        except Exception:
            break


def expand_category_products(driver: webdriver.Chrome, category: str) -> None:
    """Jump to the TGDD category hash page that exposes the full product set."""
    hash_urls = {
        "smartphone": "https://www.thegioididong.com/dtdd#c=42&o=7&pi=9",
    }
    target_url = hash_urls.get(category)
    if not target_url:
        return

    try:
        driver.get(target_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "li.item"))
        )
        time.sleep(7.0)
    except Exception as exc:
        log.warning(f"  Category hash navigation failed for {category}: {exc}")


def extract_review_page_count(html: str) -> int:
    """Extract total review pages from links like javascript:ratingCmtList(3)."""
    nums = [int(x) for x in re.findall(r"ratingCmtList\((\d+)\)", html or "")]
    return max(nums) if nums else 1


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_catalog(html: str, category: str, limit: int | None, seen: set[str] = None) -> list[tuple[ProductRow, str]]:
    soup = BeautifulSoup(html, "html.parser")

    # All known card selectors for TGDD
    cards = soup.select("li.item")
    log.info(f"  Raw li.item cards: {len(cards)}")

    if seen is None:
        seen = set()
    rows: list[tuple[ProductRow, str]] = []

    for card in cards:
        # URL
        link = card.select_one("a.main-contain, a[href]")
        if not link:
            continue
        href = link.get("href", "")
        if not href:
            continue
        url = href if href.startswith("http") else f"https://www.thegioididong.com{href}"
        if url in seen:
            continue
        seen.add(url)

        # Name — verified selector: p.product-title  OR  data-name attribute
        name = safe_text(card, "p.product-title")
        if not name:
            name = link.get("data-name", "")
        if not name:
            name = safe_text(card, "h3, h4, .name")
        name = name.replace("\n", " ").strip()

        # Brand — from data-brand attribute
        brand = link.get("data-brand", "")
        if not brand and name:
            brand = name.split()[0].title()

        # Product ID — slug from URL
        slug = url.rstrip("/").split("/")[-1]
        product_id = slug or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or str(uuid.uuid4())

        # Prices
        discounted_raw = safe_text(card, "strong.price, .box-price strong, .price")
        original_raw   = safe_text(card, ".price-old, .oldprice, strike") or discounted_raw

        # Rating & Sales — verified HTML structure:
        #   <div class="rating_Compare">
        #     <div class="vote-txt"><i/><b>5</b></div>   <- avg star
        #     <span>• Da ban 179</span>                  <- sales volume
        #   </div>
        avg_rating = ""
        sales_volume = ""
        rc_el = card.select_one(".rating_Compare")
        if rc_el:
            # Avg star rating
            star_b = rc_el.select_one(".vote-txt b")
            if star_b:
                avg_rating = star_b.get_text(strip=True)
            # Sales volume — find span containing 'ban'
            for span in rc_el.select("span"):
                txt = span.get_text(strip=True)
                if "bán" in txt or "ban" in txt.lower():
                    sales_volume = parse_compact_count(txt)
                    break

        # Total_Reviews — not on listing page, will be filled from product page
        total_reviews = ""

        rows.append((
            ProductRow(
                Product_ID=product_id,
                Name=name,
                Brand=brand,
                Category=category,
                Original_Price=numeric_text(original_raw),
                Discounted_Price=numeric_text(discounted_raw),
                Delivery_Options=safe_text(card, ".shiping, .delivery, [class*='ship']") or "standard",
                Sales_Volume=sales_volume,
                Avg_Star_Rating=avg_rating,
                Total_Reviews=total_reviews,
                Product_URL=url,
            ),
            url,
        ))

        if limit and len(rows) >= limit:
            break

    return rows


def parse_reviews(product_id: str, product_name: str, html: str) -> list[ReviewRow]:
    """Parse reviews from TGDD product page.

    Verified HTML structure (2026-05-31):
      li.par
        .cmt-top > p.cmt-top-name          <- reviewer name (SKIP)
        .cmt-intro > .cmt-top-star > i.iconcmt-starbuy  <- stars (count i tags)
        .cmt-content > p.cmt-txt           <- actual review text
        .cmt-command > span.cmtd           <- usage duration
        .support                            <- 'ngay DD/MM/YYYY' date string
    """
    soup = BeautifulSoup(html, "html.parser")

    review_nodes = soup.select("li.par")
    if not review_nodes:
        review_nodes = soup.select(".comment-item, .cmt-item")

    rows: list[ReviewRow] = []
    for node in review_nodes:
        # ── Review text: ONLY .cmt-txt inside .cmt-content ──────────────────
        txt_el = node.select_one(".cmt-content .cmt-txt, .cmt-content p")
        if not txt_el:
            txt_el = node.select_one(".cmt-txt")
        text = " ".join(txt_el.get_text(" ", strip=True).split()) if txt_el else ""

        # ── Star rating: count i.iconcmt-starbuy inside .cmt-top-star ───────
        star_el = node.select_one(".cmt-top-star")
        if star_el:
            filled = len(star_el.select("i.iconcmt-starbuy"))
            star_rating = str(filled) if 1 <= filled <= 5 else ""
        else:
            star_rating = ""

        # ── Date: extract from .support text 'ngay DD/MM/YYYY' ───────────────
        support_el = node.select_one(".support")
        date = ""
        support_text = ""
        if support_el:
            support_text = " ".join(support_el.get_text(" ", strip=True).split())
            dm = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", support_text)
            date = dm.group(1) if dm else ""
        support_contacted = "1" if support_text else "0"

        # Keep support-status notes and keep rating-only reviews (empty text is allowed).
        if support_text:
            if text:
                text = f"{text} || {support_text}"
            else:
                text = support_text

        if not text and not star_rating:
            continue

        # ── Helpfulness count + stable comment id from links like javascript:likeRating(123)
        helpfulness = "0"
        comment_uid = ""
        useful_el = node.select_one(
            "a[href*='likeRating'], a[onclick*='likeRating'], .click-useful, [class*='useful'], [class*='huu-ich']"
        )
        if not useful_el:
            for cand in node.select("a, span"):
                txt = cand.get_text(" ", strip=True).lower()
                if "hữu ích" in txt or "huu ich" in txt:
                    useful_el = cand
                    break
        if useful_el:
            useful_text = useful_el.get_text(" ", strip=True)
            href = useful_el.get("href", "") if hasattr(useful_el, "get") else ""
            onclick = useful_el.get("onclick", "") if hasattr(useful_el, "get") else ""
            id_match = re.search(r"likeRating\((\d+)\)", f"{href} {onclick}")
            if id_match:
                comment_uid = f"tgdd-{id_match.group(1)}"
            # Find numbers inside parentheses or text like "Hữu ích (12)" -> 12
            num_match = re.search(r"\((\d+)\)", useful_text)
            if num_match:
                helpfulness = num_match.group(1)
            else:
                # If no parenthesis but has numbers like "12 Hữu ích" or just "12"
                num_only = numeric_text(useful_text)
                if num_only:
                    helpfulness = num_only

        rows.append(ReviewRow(
            Review_ID=comment_uid or str(uuid.uuid4()),
            Product_ID=product_id,
            Product_Name=product_name,
            Review_Date=date,
            Star_Rating=star_rating,
            Review_Text=text,
            Helpfulness_Count=helpfulness,
            Support_Contacted=support_contacted,
        ))
    return rows


# ── CSV helpers ───────────────────────────────────────────────────────────────

def ensure_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv(path: Path, rows: list, fieldnames: list[str]) -> None:
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            w.writerow(asdict(row))


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape(limit: int | None, headless: bool, max_reviews: int, categories: list) -> None:
    pf = [f.name for f in fields(ProductRow)]
    rf = [f.name for f in fields(ReviewRow)]
    pp = RAW_DIR / "products.csv"
    rp = RAW_DIR / "reviews.csv"
    ensure_csv(pp, pf)
    ensure_csv(rp, rf)

    seen_urls: set[str] = set()
    if pp.exists():
        try:
            with pp.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Product_URL"):
                        seen_urls.add(row["Product_URL"])
            log.info(f"Loaded {len(seen_urls)} already scraped products from products.csv to prevent duplicates.")
        except Exception as e:
            log.warning(f"Could not load existing products.csv: {e}")

    driver = build_driver(headless=headless)
    driver.set_page_load_timeout(50)
    total_p = total_r = 0

    try:
        for cat_name, cat_url in categories:
            log.info(f"=== Category: {cat_name.upper()} ===")

            if not get_page(driver, cat_url):
                log.error(f"Cannot load {cat_url}, skip.")
                continue

            # Direct loading bypasses redundant steps
            # expand_category_products(driver, cat_name)
            log.info("Expanding catalog page by clicking 'Xem thêm' to load 100+ items...")
            scroll_catalog(driver, max_scrolls=20)
            pairs = parse_catalog(driver.page_source, category=cat_name, limit=limit, seen=seen_urls)
            log.info(f"  Parsed {len(pairs)} new products to scrape")

            for i, (product, url) in enumerate(pairs, 1):
                log.info(f"  [{i}/{len(pairs)}] {product.Name[:55]} | {product.Brand}")

                if not get_page(driver, url):
                    log.warning(f"    Skip (page load failed)")
                    append_csv(pp, [product], pf)  # save what we have
                    total_p += 1
                    continue

                page_html = driver.page_source
                page_soup = BeautifulSoup(page_html, "html.parser")

                # Total_Reviews from product page
                # Try: span/div showing total comment count
                total_rev_el = (
                    page_soup.select_one(".total-vote, .count-review, [class*='total-cmt'], [class*='count-cmt']")
                    or page_soup.select_one("[class*='luot-danh-gia'], [class*='so-danh-gia']")
                )
                if total_rev_el:
                    product.Total_Reviews = numeric_text(total_rev_el.get_text())
                else:
                    # Look for text like '3 đánh giá' or '7,3k khách hài lòng' in the rating/comment section specifically
                    found_total = ""
                    # Narrow down search tags to prevent matching headers/footers or dates
                    rating_box = page_soup.select_one(".rating-lst, .box-feedback, .detail-rating, .comment-box, .comment-list")
                    search_area = rating_box if rating_box else page_soup
                    
                    for el in search_area.select("p, span, div, b, a"):
                        txt = el.get_text(strip=True).lower()
                        # Only target elements with length less than 50 characters to avoid huge paragraphs
                        if len(txt) > 50:
                            continue
                        
                        # Check "7,3k khách hài lòng" or similar with k abbreviation
                        # Match: number optionally followed by space then 'k' (as a word boundary or followed by space/non-word like 'khách')
                        # e.g., "7,3k", "7.3 k", but NOT "110 khách" (where 'k' is part of 'khách')
                        if "khách hài lòng" in txt or "khach hai long" in txt:
                            m = re.search(r"([\d.,]+)\s*(k)?\s*(?:khách|khach)", txt)
                            if m:
                                num_str = m.group(1).replace(",", ".")
                                has_k = bool(m.group(2))
                                try:
                                    found_total = str(int(float(num_str) * 1000)) if has_k else numeric_text(num_str)
                                except ValueError:
                                    found_total = numeric_text(num_str)
                                break
                            else:
                                # matches "110 khách hài lòng"
                                m = re.search(r"([\d.,]+)\s+(?:khách|khach)", txt)
                                if m:
                                    found_total = numeric_text(m.group(1))
                                    break
                                
                        if "đánh giá" in txt or "danh gia" in txt:
                            m = re.search(r"([\d.,]+)\s*(k)?\s*(?:đánh|danh)\s*giá?", txt)
                            if m:
                                num_str = m.group(1).replace(",", ".")
                                has_k = bool(m.group(2))
                                try:
                                    found_total = str(int(float(num_str) * 1000)) if has_k else numeric_text(num_str)
                                except ValueError:
                                    found_total = numeric_text(num_str)
                                break
                            else:
                                # matches "3 đánh giá"
                                m = re.search(r"([\d.,]+)\s+(?:đánh|danh)", txt)
                                if m:
                                    found_total = numeric_text(m.group(1))
                                    break
                                    
                    if found_total:
                        product.Total_Reviews = found_total
                    else:
                        # Fallback: count li.par on page
                        total_par = len(page_soup.select("li.par"))
                        if total_par > 0:
                            product.Total_Reviews = str(total_par)

                # Save product row (now with Total_Reviews filled)
                append_csv(pp, [product], pf)
                total_p += 1

                # Always try to crawl dedicated review page first to get full comments.
                review_url = f"{url.rstrip('/')}/danh-gia"
                review_html = ""
                paged_htmls: list[str] = []
                if get_page(driver, review_url):
                    # Some products use "load more", some use ratingCmtList(page) pagination.
                    load_more_reviews(driver, max_clicks=30)
                    time.sleep(0.5)
                    review_html = driver.page_source

                if review_html:
                    paged_htmls.append(review_html)
                    total_pages = extract_review_page_count(review_html)
                    for pg in range(2, total_pages + 1):
                        try:
                            driver.execute_script(f"if (typeof ratingCmtList === 'function') ratingCmtList({pg});")
                            WebDriverWait(driver, 8).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "li.par"))
                            )
                            time.sleep(0.8)
                            paged_htmls.append(driver.page_source)
                        except Exception:
                            break

                # Prefer review-page count (e.g. '18đánh giá').
                if review_html:
                    review_soup = BeautifulSoup(review_html, "html.parser")
                    heading_txt = review_soup.get_text(" ", strip=True).lower()
                    hm = re.search(r"([\d.,]+)\s*(k)?\s*(?:đánh|danh)\s*giá", heading_txt)
                    if hm:
                        n_raw = hm.group(1).replace(",", ".")
                        try:
                            product.Total_Reviews = str(int(float(n_raw) * 1000)) if hm.group(2) else numeric_text(n_raw)
                        except ValueError:
                            product.Total_Reviews = numeric_text(hm.group(1))

                revs: list[ReviewRow] = []
                seen_review_ids: set[str] = set()

                # 1. Parse reviews from dedicated review page (if available)
                if paged_htmls:
                    for src_html in paged_htmls:
                        for rv in parse_reviews(product.Product_ID, product.Name, src_html):
                            if rv.Review_ID in seen_review_ids:
                                continue
                            seen_review_ids.add(rv.Review_ID)
                            revs.append(rv)

                # 2. Parse reviews from the main product details page (fallback / merge)
                for rv in parse_reviews(product.Product_ID, product.Name, page_html):
                    if rv.Review_ID in seen_review_ids:
                        continue
                    seen_review_ids.add(rv.Review_ID)
                    revs.append(rv)

                revs = revs[:max_reviews]
                if revs:
                    append_csv(rp, revs, rf)
                    total_r += len(revs)
                    log.info(f"    +{len(revs)} reviews | total_reviews={product.Total_Reviews} (cumulative: {total_r})")
                else:
                    log.info(f"    No reviews found")

                time.sleep(PAGE_SLEEP)

            log.info(f"  Category done: {total_p} products, {total_r} reviews so far")

    finally:
        driver.quit()

    log.info(f"FINISHED: {total_p} products, {total_r} reviews")
    log.info(f"  -> {pp.resolve()}")
    log.info(f"  -> {rp.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TGDD Scraper v3")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max products per category (default: all)")
    parser.add_argument("--max-reviews", type=int, default=100,
                        help="Max reviews per product (default: 100)")
    parser.add_argument("--headed",      action="store_true",
                        help="Show browser window")
    parser.add_argument("--category",    type=str, default=None,
                        help="Only one category: smartphone|tablet|laptop")
    args = parser.parse_args()

    cats = CATEGORIES
    if args.category:
        cats = [c for c in CATEGORIES if c[0] == args.category]
        if not cats:
            log.error(f"Unknown category '{args.category}'")
            sys.exit(1)

    scrape(
        limit=args.limit,
        headless=not args.headed,
        max_reviews=args.max_reviews,
        categories=cats,
    )


if __name__ == "__main__":
    main()
