# 📱 Phân Tích Cảm Xúc Khách Hàng & Dự Báo Doanh Số Thị Trường Điện Thoại (TGDD)

> Đồ án môn **Nhập môn Khoa học Dữ liệu** — Đại học Bách Khoa Hà Nội (HUST), kỳ 20252.
> Hệ thống Khoa học dữ liệu **end-to-end** trên dữ liệu thực tế của **Thế Giới Di Động (thegioididong.com)**.

---

## 📑 Mục lục
1. [Tổng quan & con số nổi bật](#1-tổng-quan--con-số-nổi-bật)
2. [Bài toán & mục tiêu](#2-bài-toán--mục-tiêu)
3. [Mô hình kiến trúc hệ thống](#3-mô-hình-kiến-trúc-hệ-thống)
4. [Luồng xử lý dữ liệu (Data Flow)](#4-luồng-xử-lý-dữ-liệu-data-flow)
5. [Dữ liệu & tiền xử lý chung](#5-dữ-liệu--tiền-xử-lý-chung)
6. [⭐ Các mô hình ML — phương pháp & kết quả](#6--các-mô-hình-ml--phương-pháp--kết-quả)
   - [6.1 Phân loại cảm xúc (Text Classification)](#61-phân-loại-cảm-xúc-text-classification)
   - [6.2 Phân tích cảm xúc theo khía cạnh (ABSA)](#62-phân-tích-cảm-xúc-theo-khía-cạnh-absa)
   - [6.3 Phân cụm sản phẩm (Clustering)](#63-phân-cụm-sản-phẩm-clustering)
   - [6.4 Độ co giãn cầu theo giá (Price Elasticity)](#64-độ-co-giãn-cầu-theo-giá-price-elasticity)
   - [6.5 Dự báo doanh số (XGBoost)](#65-dự-báo-doanh-số-xgboost)
   - [6.6 Gợi ý bán kèm (Apriori)](#66-gợi-ý-bán-kèm-apriori)
7. [Bảng tổng hợp kết quả](#7-bảng-tổng-hợp-kết-quả)
8. [Hệ thống triển khai & cài đặt](#8-hệ-thống-triển-khai--cài-đặt)

---

## 1. Tổng quan & con số nổi bật

Pipeline Khoa học dữ liệu hoàn chỉnh: **thu thập → tiền xử lý → 5 mô hình ML → triển khai dịch vụ**, trả lời 5 câu hỏi kinh doanh: khách hàng nghĩ gì về sản phẩm? phân loại bình luận tự động? thị trường có những phân khúc nào? doanh số nhạy giá ra sao? dự báo doanh số thế nào?

| Hạng mục | Giá trị |
|---|---|
| Sản phẩm / thương hiệu | **187** / **13** |
| Đánh giá / sản phẩm có review | **2.756** / **138** |
| Tập huấn luyện phân loại cảm xúc | **2.075** mẫu (1.324 pos / 751 neg) |
| **Phân loại cảm xúc — Accuracy** | **0.920** |
| **Phân loại cảm xúc — F1 (positive)** | **0.937** |
| **Phân cụm — Silhouette** | **0.498** (k=8) |
| **Dự báo doanh số — R²** | **0.461** (5-fold CV) |

> 📌 **Báo cáo này tập trung vào phương pháp và kết quả tổng quát** (F1, recall, accuracy, silhouette, RMSE/R²…). **Kết quả chi tiết từng sản phẩm** (Pros/Cons cụ thể, phân khúc của từng máy, dự báo từng model) được trình bày trong phần **Demo**.

---

## 2. Bài toán & mục tiêu

| # | Bài toán | Loại bài toán | Kỹ thuật chính | Chỉ số đánh giá |
|---|---|---|---|---|
| 1 | Phân loại cảm xúc bình luận | Phân loại nhị phân (supervised) | TF-IDF + Ensemble | Accuracy, F1, Recall |
| 2 | Cảm xúc theo khía cạnh (ABSA) | NLP — Rule + lexicon | Keyword + cue weighting | Độ phủ, điểm TB/khía cạnh |
| 3 | Phân khúc sản phẩm | Học không giám sát | K-Means + PCA | Silhouette |
| 4 | Độ co giãn cầu theo giá | Hồi quy tuyến tính | Linear Regression/hãng | PED, R² |
| 5 | Dự báo doanh số | Hồi quy phi tuyến | XGBoost + Optuna | RMSE, MAE, R² |

**Mục tiêu kỹ thuật:** xử lý đặc thù **tiếng Việt TMĐT** (teencode, từ ghép) · chạy **100% CPU** · **tái lập** (seed=42) · **chống rò rỉ dữ liệu** · đóng gói thành **API + dashboard**.

---

## 3. Mô hình kiến trúc hệ thống

### 3.1. Kiến trúc phân tầng (5 tầng, ghép nối lỏng qua artefact)

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  TẦNG 5 — TRÌNH BÀY    Next.js :3000  │  Streamlit :8501  │  Swagger /docs        ║
╚═══════════════════════════════╪══════════════════════════════════════════════════╝
                    HTTP/JSON    │
╔═══════════════════════════════▼══════════════════════════════════════════════════╗
║  TẦNG 4 — DỊCH VỤ      FastAPI + Uvicorn :8000  (nạp .pkl vào RAM, trả real-time)  ║
║   /sentiment/predict   /forecast/sales   /products/clusters   /cross_sell  …       ║
╚═══════════════════════════════╪══════════════════════════════════════════════════╝
                  joblib.load    │
╔═══════════════════════════════▼══════════════════════════════════════════════════╗
║  TẦNG 3 — MÔ HÌNH      scikit-learn · XGBoost · Optuna · mlxtend                   ║
║   → rf_classifier.pkl · kmeans_product.pkl · xgboost_model.pkl · *.json            ║
╚═══════════════════════════════╪══════════════════════════════════════════════════╝
                pandas.read      │
╔═══════════════════════════════▼══════════════════════════════════════════════════╗
║  TẦNG 2 — XỬ LÝ        pandas · regex · pyvi/underthesea                           ║
║   parse_number · clean_text(+teencode) · word-segment · extract_specs · labeling   ║
╚═══════════════════════════════╪══════════════════════════════════════════════════╝
                raw/*.csv        │
╔═══════════════════════════════▼══════════════════════════════════════════════════╗
║  TẦNG 1 — THU THẬP     Selenium (Chrome headless) + BeautifulSoup                  ║
║   thegioididong.com → products.csv (187) · reviews.csv (2.756)                     ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

### 3.2. Công nghệ theo tầng
| Tầng | Thành phần | Công nghệ |
|---|---|---|
| Thu thập | `scraper/tgdd_scraper.py` | Selenium, BeautifulSoup |
| Xử lý | `preprocessing/preprocess.py` | pandas, numpy, regex, **pyvi/underthesea** |
| Mô hình | `models/**` | scikit-learn, **XGBoost**, **Optuna**, mlxtend, joblib |
| Dịch vụ | `api/main.py` | **FastAPI**, Uvicorn, Pydantic |
| Trình bày | `dashboard/`, `streamlit_app/` | **Next.js**, **Streamlit**, Plotly |

> **Nguyên tắc:** các tầng không gọi nhau trực tiếp mà giao tiếp qua *artefact* (file) → train lại model không cần sửa API; đổi frontend không đụng ML.

---

## 4. Luồng xử lý dữ liệu (Data Flow)

```
 tgdd_scraper.py
   ├─▶ raw/products.csv ─▶ preprocess.py ─▶ products_clean.csv ─┬─▶ product_cluster.py ─▶ products_clustered.csv
   │                                                            ├─▶ price_optimization.py ─▶ price_elasticity.json
   │                                                            └─▶ train_xgboost.py ─▶ xgboost_model.pkl
   │                                                                     ▲
   └─▶ raw/reviews.csv ──▶ preprocess.py ─▶ reviews_clean.csv ─▶ train_tfidf_rf.py ─▶ reviews_scored.csv ┘
                       └─▶ aspect_analysis.py ─▶ aspect_sentiment.json
```

| Bước | Script | Đầu vào | Đầu ra |
|---|---|---|---|
| 1 | `tgdd_scraper.py` | TGDD website | `raw/products.csv`, `raw/reviews.csv` |
| 2 | `preprocess.py` | `raw/*.csv` | `products_clean.csv`, `reviews_clean.csv` |
| 3 | `train_tfidf_rf.py` | `reviews_clean.csv` | `rf_classifier.pkl`, `reviews_scored.csv` |
| 4 | `aspect_analysis.py` | `raw/reviews.csv` | `aspect_sentiment.json` |
| 5 | `product_cluster.py` | `products_clean.csv` | `products_clustered.csv`, `kmeans_product.pkl` |
| 6 | `price_optimization.py` | `products_clean.csv` | `price_elasticity.json` |
| 7 | `train_xgboost.py` | `products_clean.csv` + `reviews_scored.csv` | `xgboost_model.pkl` |

---

## 5. Dữ liệu & tiền xử lý chung

### 5.1. Dữ liệu thu thập
**187 sản phẩm / 13 hãng · 2.756 review / 138 sản phẩm.** Phân bố sao review: 1★:420 · 2★:333 · 3★:493 · 4★:431 · **5★:1.079** (lệch tích cực ~39% — đặc trưng TMĐT).

`products.csv` (11 trường): `Product_ID, Name, Brand, Category, Original_Price, Discounted_Price, Sales_Volume, Avg_Star_Rating, Total_Reviews, Delivery_Options, Product_URL`.
`reviews.csv` (9 trường): `Review_ID, Product_ID, Product_Name, Review_Date, Star_Rating, Review_Text, Helpfulness_Count, Support_Contacted, Language_Code`.

### 5.2. Tiền xử lý chung (áp dụng trước mọi mô hình)
| Bước | Hàm | Mô tả |
|---|---|---|
| Chuẩn hóa số | `parse_number()` | `"37.990.000đ"`, `"36,99 triệu"`, `"243,4k"` → số thực; phân biệt dấu `.`/`,` kiểu VN; khuyết → median |
| Làm sạch text | `clean_text()` | lowercase → xóa HTML/URL/**emoji**/ký tự đặc biệt → **map teencode** token-by-token |
| Từ điển teencode | `teencode_dict.json` | **81 mục**: `ko/k→không`, `dt/đt→điện thoại`, `sài/xài→dùng`, `đểu→kém`… |
| Tách từ | `segment_vi()` | `pyvi.ViTokenizer` (fallback underthesea) → ghép từ kép bằng `_` |
| Trích cấu hình | `extract_specs_from_name()` | Regex `"8GB/256GB"` → RAM/ROM; luật riêng iPhone, feature phone, pin/cam |
| Gán nhãn | rule theo sao | `≥4★→1`, `≤2★→0`, `=3★→loại` |

**Ví dụ một review chạy qua tiền xử lý (minh họa phương pháp):**
```
Thô:        "Viet danh gia dai thi k up đc"            (2★)
Teencode:   "viet danh gia dai thi không up được"     (k→không, đc→được)
Tách từ:    "viet danh_gia dai thi không up được"
Gán nhãn:   label = 0  (vì 2★ ≤ 2 → tiêu cực)
```
**Kết quả gán nhãn:** 1.510 tích cực · 753 tiêu cực · 493 trung tính (loại) → sau khi bỏ text rỗng còn **2.075 mẫu** huấn luyện.

---

## 6. ⭐ Các mô hình ML — phương pháp & kết quả

> Mỗi mô hình trình bày theo cấu trúc: **Bài toán → Đầu vào → Các bước xử lý → Thuật toán & cấu hình → Đánh giá → Kết quả tổng quát.**

---

### 6.1. Phân loại cảm xúc (Text Classification)

**Bài toán:** phân loại nhị phân một bình luận tiếng Việt → **Tích cực (1) / Tiêu cực (0)**.
**Đầu vào:** `reviews_clean.csv` — **2.075 mẫu** có nhãn, không rỗng (1.324 pos / 751 neg).

**Các bước xử lý của mô hình:**
1. Lọc mẫu có nhãn & text khác rỗng; gán **trọng số mẫu** `weight = 1 + Helpfulness_Count` (review nhiều lượt "hữu ích" → ảnh hưởng lớn hơn).
2. **Đặc trưng A — TF-IDF** từ-cấp, **n-gram (1,2)**, `min_df=2` (loại từ hiếm, giảm nhiễu).
3. **Đặc trưng B — underthesea**: gọi `underthesea.sentiment()` cho mỗi câu → 1 giá trị (pos=1/neg=0/neutral=0.5) làm tín hiệu cảm xúc tổng quát.
4. **Ghép đặc trưng** `hstack([TF-IDF, underthesea])` → ma trận thưa.
5. **StratifiedKFold (5-fold)**: mỗi fold **fit vectorizer riêng trên tập train** (tránh rò rỉ), huấn luyện ensemble với `sample_weight`.
6. **Fit cuối** trên toàn bộ dữ liệu → lưu `tfidf_vectorizer.pkl` + `rf_classifier.pkl`; ghi điểm `Sentiment_Score` ra `reviews_scored.csv` (làm đặc trưng cho mô hình dự báo).

**Thuật toán — Ensemble Soft-Voting:**
```
TF-IDF(1,2) ⊕ underthesea ──▶ ┌─ LogisticRegression(class_weight=balanced) ─┐
                              │                                             ├─▶ Soft Voting ─▶ P(tích cực)
                              └─ CalibratedClassifierCV(LinearSVC) ─────────┘
```
- **Lý do chọn:** LinearSVC mạnh trên text thưa nhưng không cho xác suất → bọc `Calibrated`; LogReg cho xác suất ổn định. Soft-voting hai mô hình → vừa chính xác vừa trả `confidence %` cho API.
- **Cấu hình:** `class_weight=balanced` (chống lệch nhãn), `max_iter=1000`, `seed=42`, **100% CPU**.

> ⚠️ **Lưu ý:** file lưu tên cũ `rf_classifier.pkl` (giữ tương thích API) nhưng **bản chất là Ensemble LogReg + LinearSVC**, KHÔNG còn là Random Forest. Trình bày đúng là *"TF-IDF + Ensemble (LogReg + SVM hiệu chỉnh)"*.

**Đánh giá:** out-of-fold qua 5-fold CV.

**Kết quả tổng quát:**

| Chỉ số | Giá trị |
|---|---|
| **Accuracy** | **0.920** |
| **F1 (positive)** | **0.937** |
| F1 (macro) | 0.913 |
| F1 (weighted) | 0.920 |
| Precision / Recall (tích cực) | 0.938 / 0.937 |
| Precision / Recall (tiêu cực) | 0.888 / 0.891 |

**Ma trận nhầm lẫn (out-of-fold, n=2.075):**
| | Dự đoán Tiêu cực | Dự đoán Tích cực |
|---|---|---|
| **Thật Tiêu cực** | 669 ✅ | 82 |
| **Thật Tích cực** | 84 | 1.240 ✅ |

**Độ ổn định giữa các fold (F1 positive):** 0.937 · 0.943 · 0.949 · 0.913 · 0.944 → **TB 0.937 ± 0.014** (rất ổn định).

#### 🔬 So sánh phương pháp (cùng 5-fold CV, n=2.075) — *vì sao chọn Ensemble + underthesea*
| # | Phương pháp | Accuracy | F1_macro | Recall (tiêu cực) |
|---|---|---|---|---|
| 0 | Baseline (đoán lớp đa số) | 0.638 | 0.390 | 0.000 |
| 1 | TF-IDF + **Naive Bayes** | 0.904 | 0.896 | 0.871 |
| 2 | TF-IDF + **Random Forest** | 0.898 | 0.886 | **0.798** ⬇️ |
| 3 | TF-IDF + Logistic Regression | 0.921 | 0.913 | 0.872 |
| 4 | TF-IDF + LinearSVC | **0.924** | **0.918** | 0.892 |
| 5 | TF-IDF + Ensemble (LogReg+SVC) | 0.921 | 0.914 | 0.871 |
| 6 | TF-IDF (word+char) + Ensemble | 0.919 | 0.912 | 0.866 |
| 7 | **⭐ TF-IDF + underthesea + Ensemble (FINAL)** | 0.920 | 0.913 | **0.891** |

**Kết luận so sánh (nhấn mạnh phương pháp):**
- ✅ Nhóm **mô hình tuyến tính** (LogReg, SVC, Ensemble: ~0.92) **vượt rõ** Naive Bayes (0.904) và **Random Forest (0.898)** trên text thưa.
- ✅ **Random Forest bị loại** vì recall lớp tiêu cực thấp nhất (**0.798**) — bỏ sót nhiều bình luận xấu; đây chính là lý do dự án **chuyển từ RF sang Ensemble** (dù file vẫn tên `rf_classifier.pkl`).
- ✅ LinearSVC đơn lẻ có accuracy cao nhất (0.924) **nhưng không cho xác suất** → Ensemble bọc `Calibrated(SVC)` + LogReg để có `confidence %` cho API, chỉ đánh đổi ~0.4% accuracy.
- ✅ Thêm đặc trưng **underthesea** nâng **recall lớp tiêu cực 0.871 → 0.891** (cân bằng 2 lớp tốt nhất trong các Ensemble) — quan trọng với dữ liệu lệch nhãn.
- ✅ Char n-gram **không giúp ích** (0.919 < 0.921) → giữ word-only cho gọn nhẹ.

> ✅ Mô hình chính xác (92%) và **cân bằng** cả hai lớp (recall tiêu cực 0.891 dù lớp này ít hơn) nhờ `class_weight=balanced` + trọng số "hữu ích" + đặc trưng underthesea xử lý teencode.

---

### 6.2. Phân tích cảm xúc theo khía cạnh (ABSA)

**Bài toán:** với mỗi sản phẩm, chấm điểm hài lòng và rút **Pros/Cons** theo **5 khía cạnh**: Pin&Sạc · Hiệu năng/OS · Camera · Thiết kế · Dịch vụ/Nhân viên.
**Đầu vào:** `raw/reviews.csv` (138 sản phẩm có review).

**Các bước xử lý của mô hình:**
1. Làm sạch + map teencode cho từng review.
2. **Tách mệnh đề** theo dấu câu & liên từ (`. , ; ! ?` và `"nhưng"`).
3. Với mỗi mệnh đề, **khớp từ khóa** của 5 khía cạnh (regex word-boundary).
4. **Luật chống chồng lấn:** mệnh đề chứa "pin/sạc" → không tính *Hiệu năng*; chứa "camera/chụp" → không tính *Thiết kế*.
5. **Quyết định sắc thái:** mặc định theo số sao (≥4→pos, ≤2→neg, =3→0.5), **ghi đè** bằng *cue lexicon* (chỉ có cue tiêu cực→0, chỉ có cue tích cực→1).
6. **Cộng dồn theo trọng số** `weight = 1 + Helpfulness_Count`.
7. Gom theo sản phẩm: `Score = pos_weight / total_weight` (0–1); lọc Pros (sentiment=1) & Cons (sentiment=0).

**Thuật toán:** **Rule-based + lexicon** (5 bộ từ khóa khía cạnh + 2 bộ cue pos/neg), có trọng số theo độ hữu ích. Không cần huấn luyện.
**Đánh giá:** không có nhãn aspect chuẩn → đánh giá bằng **độ phủ** (số SP đề cập) và **điểm trung bình** mỗi khía cạnh (tổng quát toàn thị trường).

**Kết quả tổng quát (138 sản phẩm):**

| Khía cạnh | Số SP đề cập | Điểm TB (0–1) | Nhận định |
|---|---|---|---|
| Thiết kế / Ngoại hình | 67 | **0.77** | ✅ Mạnh nhất toàn thị trường |
| Pin & Sạc | 107 | 0.66 | Khá tốt |
| Camera | 99 | 0.65 | Khá tốt |
| Hiệu năng / OS | 88 | 0.47 | Trung bình |
| Dịch vụ / Nhân viên | 125 | **0.39** | ⚠️ Yếu nhất & bị nhắc nhiều nhất |

> 📌 Pros/Cons cụ thể của từng sản phẩm sẽ hiển thị trong **Demo**.

---

### 6.3. Phân cụm sản phẩm (Clustering)

**Bài toán:** gom 187 sản phẩm thành các **phân khúc thị trường** + bản đồ 2D.
**Đầu vào:** 7 đặc trưng — `RAM, ROM, Battery, Camera_MP, Discounted_Price, Avg_Star_Rating, Total_Reviews`.

**Các bước xử lý của mô hình:**
1. Ép numeric 7 đặc trưng, điền khuyết bằng **median**.
2. **Chuẩn hóa** `StandardScaler` (z-score, để các thang đo đồng đều).
3. **Quét k = 2…9**, mỗi k chạy `KMeans(n_init=20)`, tính **inertia** + **silhouette**.
4. **Chọn k tối ưu theo silhouette lớn nhất**.
5. Fit `KMeans(best_k)` → `cluster_id`.
6. **PCA giảm 2 chiều** → `PC1, PC2` (vẽ bản đồ phân khúc).
7. **Đặt tên cụm** bằng luật giá & RAM (6 nhãn nghiệp vụ tiếng Việt).

**Thuật toán:** **K-Means** (chọn k bằng Silhouette) + **PCA** (trực quan hóa). Seed=42.
**Đánh giá:** **Silhouette score** (chỉ số nội tại đo độ tách cụm).

**Kết quả tổng quát — chọn mô hình theo Silhouette:**

| k | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 |
|---|---|---|---|---|---|---|---|---|
| Silhouette | 0.361 | 0.375 | 0.400 | 0.413 | 0.464 | 0.478 | **0.498** ✅ | 0.475 |

→ **k = 8** (Silhouette = **0.498**). Phân bố quy mô 6 nhóm nghiệp vụ: Cận cao cấp–Hiệu năng (62) · Cao cấp–Siêu phẩm (47) · Tầm trung–Tiết kiệm (33) · Giá rẻ–Phổ thông (17) · Tầm trung–Pin khoẻ (17) · Cận cao cấp–Thiết kế (11).

> 📌 Bản đồ phân khúc và máy cụ thể trong từng cụm sẽ hiển thị trong **Demo**.

---

### 6.4. Độ co giãn cầu theo giá (Price Elasticity)

**Bài toán:** đo mức độ doanh số phản ứng với chiết khấu, **theo từng thương hiệu**.
**Đầu vào:** `products_clean.csv` (giá, sales theo hãng).

**Các bước xử lý của mô hình:**
1. Ép numeric giá & sales; tính `Discount_Rate = (Original − Discounted) / Original`, clip [0,1].
2. Lọc SP hợp lệ (`giá > 0` và `sales > 0`).
3. **Group theo Brand** (chỉ hãng có ≥2 sản phẩm).
4. **Hồi quy tuyến tính** `Sales_Volume = α + β·Discount_Rate` cho từng hãng → `β, α, R²`.
5. Tính **hệ số co giãn** `PED = β · (mean_discount / mean_sales)`.
6. Mô phỏng **uplift doanh số** khi tăng chiết khấu +5% / +10%.

**Thuật toán:** **Linear Regression** (scikit-learn) cho mỗi hãng.
**Đánh giá:** `R²` + dấu/độ lớn của `PED`.

**Kết quả tổng quát (13 hãng):**

| Hãng | n | Giá TB | PED | R² |
|---|---|---|---|---|
| Motorola | 8 | 11,0tr | **+1.56** | 0.49 |
| realme | 24 | 7,9tr | −0.72 | 0.16 |
| Samsung | 21 | 19,2tr | −0.49 | 0.05 |
| vivo | 22 | 12,5tr | −0.38 | 0.13 |
| OPPO | 31 | 15,8tr | −0.11 | 0.03 |

> R² thấp ở nhiều hãng cho thấy **giá không phải yếu tố duy nhất** chi phối doanh số (thương hiệu, thời điểm ra mắt cũng quan trọng) — giới hạn được báo cáo trung thực.

---

### 6.5. Dự báo doanh số (XGBoost)

**Bài toán:** hồi quy dự đoán `Sales_Volume` của sản phẩm từ cấu hình, giá, cảm xúc, lịch.
**Đầu vào (8 đặc trưng):** `Discounted_Price, Discount_Rate, Avg_Star_Rating, Sentiment_Score, Day_of_Week, Is_Weekend, Is_Holiday, Sales_Volume_7d_mean`.

**Các bước xử lý của mô hình:**
1. **Ghép cảm xúc:** lấy `Sentiment_Score` trung bình theo sản phẩm từ `reviews_scored.csv` (đầu ra mục 6.1).
2. Đặt `target = Sales_Volume`.
3. **🔑 Đặc trưng chống rò rỉ:** `Sales_Volume_7d_mean` = **trung bình doanh số của các SP *khác* cùng hãng (leave-one-out)** — không dùng nhãn của chính nó.
4. Sinh **đặc trưng lịch** (`Day_of_Week, Is_Weekend, Is_Holiday`) từ ngày nhập nếu có; nếu không, để trung tính.
5. **Tối ưu siêu tham số bằng Optuna (TPE)** — hàm mục tiêu = **RMSE trung bình KFold**.
6. Lấy bộ tham số tốt nhất → **KFold (5-fold)** đánh giá RMSE/MAE/R².
7. **Fit cuối** trên toàn bộ → lưu `xgboost_model.pkl`.

**Thuật toán:** `XGBRegressor` (`tree_method=hist`, đa nhân) + **Optuna**.
**Không gian tìm kiếm:** `n_estimators∈[50,300]`, `max_depth∈[2,6]`, `learning_rate∈[0.01,0.2]`, `subsample∈[0.7,1]`, `colsample_bytree∈[0.7,1]`, `reg_lambda∈[0.1,10]`.
**An toàn:** từ chối train nếu < 5 mẫu thật (không bịa dữ liệu giả).

**Kết quả tổng quát (5-fold CV):**

| Chỉ số | Trung bình | Khoảng các fold |
|---|---|---|
| **RMSE** | **38.353** | 13.311 – 57.002 |
| **MAE** | **18.869** | 8.549 – 31.260 |
| **R²** | **0.461** | −0.360 – 0.794 |

#### 🔬 So sánh phương pháp (cùng 5-fold CV, n=187) — *vì sao chọn XGBoost + Optuna*
| # | Phương pháp | RMSE | MAE | R² |
|---|---|---|---|---|
| 0 | Baseline (đoán trung bình) | 56.657 | 33.224 | −0.077 |
| 1 | Linear Regression | 52.985 | 28.766 | 0.062 |
| 2 | Random Forest | 47.193 | 21.811 | 0.156 |
| 3 | XGBoost (mặc định) | 42.644 | 20.581 | 0.382 |
| 4 | **⭐ XGBoost + Optuna (FINAL)** | **38.815** | **18.895** | **0.456** |
| 5 | XGBoost + Optuna **(bỏ đặc trưng Sentiment)** | 44.500 | 21.859 | **0.171** ⬇️ |

**Kết luận so sánh (nhấn mạnh phương pháp):**
- ✅ Hiệu năng **tăng dần đều** qua từng cấp độ: Baseline (R²≈0) → Linear (0.06) → RF (0.16) → XGBoost (0.38) → **XGBoost+Optuna (0.46)**.
- ✅ **Optuna** nâng R² từ 0.382 → 0.456 (**+0.074**) so với XGBoost mặc định.
- 🔑 **Đặc trưng cảm xúc rất quan trọng:** bỏ `Sentiment_Score` khiến R² **rớt mạnh 0.456 → 0.171** (giảm hơn một nửa). Điều này chứng minh **mô hình NLP (mục 6.1) tạo giá trị trực tiếp** cho mô hình dự báo — gắn kết toàn pipeline.

> R² dao động mạnh do **cỡ mẫu nhỏ (187 SP)** và dữ liệu **cắt ngang** (không phải chuỗi thời gian thật). Báo cáo trung thực: XGBoost nắm xu hướng chung, cần thêm dữ liệu lịch sử theo ngày để cải thiện.

---

### 6.6. Gợi ý bán kèm (Apriori)

**Bài toán:** khai phá luật **điện thoại → phụ kiện** để gợi ý bán kèm.
**Phương pháp:** `mlxtend.apriori` (`min_support=0.02`, `confidence≥0.3`, `lift≥1.2`) → lọc luật antecedent là smartphone, consequent là accessory.
**Trạng thái:** module đã hiện thực, cần dữ liệu giỏ hàng (`baskets.csv`) để chạy — kích hoạt khi có dữ liệu giao dịch.

---

## 7. Bảng tổng hợp kết quả

| Mô hình | Phương pháp | Chỉ số chính | **Kết quả tổng quát** |
|---|---|---|---|
| **Phân loại cảm xúc** | TF-IDF(1,2) + Ensemble (LogReg+LinearSVC) + underthesea | Accuracy / F1 / Recall | **Acc 0.920** · F1(pos) **0.937** · F1(macro) 0.913 · Recall(neg) 0.891 |
| **ABSA** | Rule + cue lexicon, weighted | Độ phủ · điểm TB | 138 SP × 5 khía cạnh; Thiết kế 0.77 ↔ Dịch vụ 0.39 |
| **Phân cụm** | K-Means + PCA | Silhouette | **0.498** (k=8) |
| **Độ co giãn giá** | Linear Regression/hãng | PED · R² | 13 hãng; Motorola PED +1.56 |
| **Dự báo doanh số** | XGBoost + Optuna | RMSE / MAE / R² | RMSE 38.353 · MAE 18.869 · **R² 0.461** |

### 7.1. Tổng hợp các kết quả so sánh phương pháp
Mỗi mô hình đều được **đối chứng với baseline & các phương án thay thế** (cùng split CV) để chứng minh lựa chọn phương pháp:

| Bài toán | Baseline | → Cải tiến trung gian | → **Phương pháp cuối** | Mức cải thiện |
|---|---|---|---|---|
| **Phân loại cảm xúc** | Đoán lớp đa số: Acc **0.638** | NB 0.904 · RF 0.898 · LogReg 0.921 | **Ensemble + underthesea: Acc 0.920, Recall(neg) 0.891** | +0.28 Acc vs baseline; cân bằng 2 lớp tốt nhất |
| **Dự báo doanh số** | Đoán TB: R² **−0.077** | Linear 0.06 · RF 0.16 · XGB 0.38 | **XGBoost+Optuna: R² 0.456** | +0.53 R² vs baseline |
| **Phân cụm** | k=2: Silhouette 0.361 | k=4..7: 0.40→0.48 | **k=8: Silhouette 0.498** | +0.137 vs k=2 |

### 7.2. Hai phát hiện then chốt về phương pháp
- 🔑 **Random Forest KHÔNG phù hợp text thưa:** recall lớp tiêu cực chỉ 0.798 (thấp nhất) → dự án **thay bằng Ensemble tuyến tính** (LogReg + Calibrated LinearSVC).
- 🔑 **NLP nuôi dự báo:** bỏ đặc trưng `Sentiment_Score` làm **R² rớt 0.456 → 0.171** → mô hình cảm xúc (6.1) đóng góp trực tiếp & lớn nhất cho mô hình dự báo (6.5), gắn kết toàn pipeline.

**Điểm kỹ thuật nổi bật:** chống rò rỉ dữ liệu (LOO brand mean) · chọn k bằng Silhouette · CV phân tầng + trọng số mẫu · ablation chứng minh giá trị từng đặc trưng · 100% CPU · seed cố định (tái lập).

---

## 8. Hệ thống triển khai & cài đặt

### 8.1. Dịch vụ
- **FastAPI** (`:8000`) — 5 endpoint: `/sentiment/predict`, `/forecast/sales`, `/products/clusters`, `/products/{id}/cross_sell`, `/customers/churn_alert`.
- **Streamlit** (`:8501`) — 6 tab: Phân khúc · Pros&Cons · Độ nhạy giá · Dự báo · Demo cảm xúc real-time · Từ điển teencode.
- **Next.js** (`:3000`) — dashboard quản trị, mô phỏng khuyến mãi, ML Sentiment Room.

### 8.2. Cài đặt
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows (macOS/Linux: source .venv/bin/activate)
pip install --upgrade pip && pip install -r requirements.txt
```

### 8.3. Chạy pipeline (từ thư mục gốc)
```bash
python scraper/tgdd_scraper.py --category smartphone --max-reviews 2000  # 1. Crawl
python preprocessing/preprocess.py                                       # 2. Tiền xử lý
python models/sentiment/train_tfidf_rf.py                                # 3. Phân loại cảm xúc
python preprocessing/aspect_analysis.py                                  # 4. ABSA
python models/clustering/product_cluster.py                              # 5. Phân cụm
python preprocessing/price_optimization.py                               # 6. Độ co giãn giá
python models/forecasting/train_xgboost.py --trials 100                  # 7. Dự báo
```

### 8.4. Khởi chạy dịch vụ
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload   # FastAPI → /docs
cd dashboard && npm install && npm run dev                              # Next.js :3000
streamlit run streamlit_app/app.py                                      # Streamlit :8501
```

### 8.5. Tái tạo số liệu báo cáo
```bash
python notebooks/compute_report_stats.py    # thống kê tổng hợp → _readme_stats.txt
python notebooks/eval_sentiment.py          # accuracy + per-class + confusion matrix → _sentiment_eval.txt
python notebooks/compare_sentiment.py       # bảng so sánh phương pháp cảm xúc → _compare_sentiment.txt
python notebooks/compare_forecast.py        # bảng so sánh phương pháp dự báo → _compare_forecast.txt
```

---

*Mọi số liệu được tính trực tiếp từ `data-project/` và các file metrics trong `models/`. Chạy 2 script trong `notebooks/` để tái tạo.*
