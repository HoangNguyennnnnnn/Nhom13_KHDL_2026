"""Scrape TGDD smartphone products and reviews.

The scraper uses Selenium for JavaScript interactions and BeautifulSoup for
HTML parsing. TGDD page markup changes frequently, so selectors are deliberately
layered and tolerant. Run politely and verify compliance with the website terms
before scraping at scale.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


BASE_URL = "https://www.thegioididong.com/dtdd"
RAW_DIR = Path("data-project/raw")
SEED_SLEEP_SECONDS = 1.2


@dataclass
class ProductRow:
    Product_ID: str
    Brand: str
    Original_Price: str
    Discounted_Price: str
    Delivery_Options: str
    Inward_Date: str
    Sales_Volume: str
    Avg_Star_Rating: str
    Total_Reviews: str


@dataclass
class ReviewRow:
    Review_ID: str
    Product_ID: str
    Review_Date: str
    Star_Rating: str
    Review_Text: str
    Language_Code: str = "vi"


def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1400")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def safe_text(node, selector: str, default: str = "") -> str:
    found = node.select_one(selector)
    return " ".join(found.get_text(" ", strip=True).split()) if found else default


def numeric_text(text: str) -> str:
    return re.sub(r"[^\d]", "", text or "")


def parse_display_number(text: str) -> str:
    """Parse TGDD display counts such as '179', '7,3k', or '101k'."""
    if not text:
        return ""
    match = re.search(r"(\d+(?:[,.]\d+)?)\s*([kK]?)", text)
    if not match:
        return ""
    value = float(match.group(1).replace(",", "."))
    if match.group(2).lower() == "k":
        value *= 1000
    return str(int(round(value)))


def extract_sold_count(text: str) -> str:
    match = re.search(r"đã\s*bán\s*([\d.,]+)\s*([kK]?)", text, flags=re.IGNORECASE)
    return parse_display_number("".join(match.groups())) if match else ""


def extract_rating_near_sold(text: str) -> str:
    """Extract the visible star value that appears immediately before 'Đã bán'."""
    sold_match = re.search(r"đã\s*bán", text, flags=re.IGNORECASE)
    if not sold_match:
        return ""
    before_sold = text[: sold_match.start()]
    candidates = re.findall(r"(?<![\d.])([1-5](?:[,.]\d)?)(?![\d.])", before_sold)
    return candidates[-1].replace(",", ".") if candidates else ""


def extract_total_reviews(text: str) -> str:
    match = re.search(r"([\d.,]+)\s*([kK]?)\s*(?:đánh\s*giá|review|nhận\s*xét)", text, flags=re.IGNORECASE)
    return parse_display_number("".join(match.groups())) if match else ""


def scroll_and_expand(driver: webdriver.Chrome, max_clicks: int = 20) -> None:
    last_height = 0
    for _ in range(max_clicks):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SEED_SLEEP_SECONDS)
        height = driver.execute_script("return document.body.scrollHeight")
        buttons = driver.find_elements(
            By.XPATH,
            "//*[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'xem thêm')]",
        )
        clicked = False
        for button in buttons:
            try:
                if button.is_displayed() and button.is_enabled():
                    driver.execute_script("arguments[0].click();", button)
                    clicked = True
                    time.sleep(SEED_SLEEP_SECONDS)
                    break
            except Exception:
                continue
        if height == last_height and not clicked:
            break
        last_height = height


def parse_catalog(html: str, limit: int | None = None) -> list[tuple[ProductRow, str]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("li.item, .listproduct li, .item.__cate_44")
    rows: list[tuple[ProductRow, str]] = []
    for card in cards:
        link = card.select_one("a[href]")
        href = link["href"] if link else ""
        if not href:
            continue
        product_url = href if href.startswith("http") else f"https://www.thegioididong.com{href}"
        name = safe_text(card, "h3") or safe_text(card, ".name")
        product_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or str(uuid.uuid4())
        prices = card.select(".price, strong.price")
        discounted = prices[0].get_text(" ", strip=True) if prices else ""
        original = safe_text(card, ".price-old, .oldprice, strike") or discounted
        card_text = " ".join(card.get_text(" ", strip=True).split())
        rating = safe_text(card, ".rating-total, .item-rating-total, .rating")
        avg_rating = (
            re.search(r"\d+(?:[,.]\d+)?", rating).group(0).replace(",", ".")
            if re.search(r"\d+(?:[,.]\d+)?", rating)
            else extract_rating_near_sold(card_text)
        )
        row = ProductRow(
            Product_ID=product_id,
            Brand=(name.split()[0] if name else "").title(),
            Original_Price=numeric_text(original),
            Discounted_Price=numeric_text(discounted),
            Delivery_Options="",
            Inward_Date="",
            Sales_Volume=extract_sold_count(card_text),
            Avg_Star_Rating=avg_rating,
            Total_Reviews="",
        )
        rows.append((row, product_url))
        if limit and len(rows) >= limit:
            break
    return rows


def enrich_product_from_detail(product: ProductRow, html: str) -> ProductRow:
    """Update fields that may exist on the detail page.

    Delivery options are location/session dependent on TGDD, so the scraper only
    records them when they are explicitly present in the rendered page.
    """
    soup = BeautifulSoup(html, "html.parser")
    detail_text = " ".join(soup.get_text(" ", strip=True).split())
    total_reviews = extract_total_reviews(detail_text)
    if total_reviews:
        product.Total_Reviews = total_reviews
    rating_text = safe_text(soup, ".rating-total, .boxReview-score, .point, .rating")
    rating_match = re.search(r"(?<![\d.])([1-5](?:[,.]\d)?)(?![\d.])", rating_text)
    if rating_match:
        product.Avg_Star_Rating = rating_match.group(1).replace(",", ".")
    delivery_text = safe_text(soup, ".delivery, .shiping, .box-delivery, .delivery-info, .policy")
    if delivery_text and re.search(r"giao|nhận hàng|vận chuyển", delivery_text, flags=re.IGNORECASE):
        product.Delivery_Options = delivery_text
    return product


def parse_reviews(product_id: str, html: str) -> list[ReviewRow]:
    soup = BeautifulSoup(html, "html.parser")
    review_nodes = soup.select(".comment_ask, .comment-item, .ratingLst li, .par")
    rows: list[ReviewRow] = []
    for node in review_nodes:
        text = safe_text(node, ".cmt-txt, .comment-content, .content, p") or node.get_text(" ", strip=True)
        if len(text) < 5:
            continue
        star_text = " ".join([span.get("class", [""])[0] for span in node.select("[class*=icon-star]")])
        star_match = re.search(r"([1-5])", safe_text(node, ".rating, .star") or star_text)
        rows.append(
            ReviewRow(
                Review_ID=str(uuid.uuid4()),
                Product_ID=product_id,
                Review_Date=safe_text(node, ".time, .date"),
                Star_Rating=star_match.group(1) if star_match else "",
                Review_Text=text,
            )
        )
    return rows


def write_csv(path: Path, rows: Iterable[object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def scrape(limit: int | None, headless: bool) -> None:
    driver = build_driver(headless=headless)
    try:
        driver.get(BASE_URL)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        scroll_and_expand(driver)
        product_pairs = parse_catalog(driver.page_source, limit=limit)
        products = [row for row, _ in product_pairs]
        reviews: list[ReviewRow] = []
        for product, url in product_pairs:
            driver.get(url)
            try:
                WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            except TimeoutException:
                continue
            scroll_and_expand(driver, max_clicks=8)
            enrich_product_from_detail(product, driver.page_source)
            reviews.extend(parse_reviews(product.Product_ID, driver.page_source))
    finally:
        driver.quit()

    write_csv(RAW_DIR / "products.csv", products, list(ProductRow.__annotations__))
    write_csv(RAW_DIR / "reviews.csv", reviews, list(ReviewRow.__annotations__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    scrape(limit=args.limit, headless=not args.headed)


if __name__ == "__main__":
    main()
