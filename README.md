# 📱 Phân Tích Cảm Xúc Khách Hàng & Dự Báo Doanh Số Thị Trường Điện Thoại (TGDD)

Dự án là hệ thống phân tích dữ liệu và học máy end-to-end cho thị trường điện thoại di động Thế Giới Di Động (TGDD), bao gồm:

1. **Thu thập dữ liệu**: Trình cào Selenium tự động thu thập thông tin sản phẩm và đánh giá từ website TGDD.
2. **Tiền xử lý & Trích xuất đặc trưng**: Chuẩn hóa đơn vị đo, làm sạch văn bản và dịch từ viết tắt (Teen code) tiếng Việt bằng từ điển tùy chỉnh. Tự động trích xuất RAM/ROM/Pin/Camera từ tên sản phẩm.
3. **Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment)**: Chia đánh giá thành 5 khía cạnh (Pin & Sạc, Hiệu năng, Camera, Thiết kế, Dịch vụ) để tìm ưu/nhược điểm từng sản phẩm.
4. **Phân loại cảm xúc 3 lớp (Sentiment Classification)**: Pipeline thiên về dữ liệu — làm sạch nhiễu, gán nhãn 3 lớp (tiêu cực / trung tính / tích cực), feature engineering xử lý phủ định, huấn luyện TF-IDF + Logistic Regression. Kèm benchmark so sánh với đặc trưng PhoBERT.
5. **Học máy không giám sát (Clustering)**: KMeans kết hợp PCA để gom cụm và trực quan hóa phân khúc smartphone 2D.
6. **Độ nhạy giá (Price Elasticity)**: Hồi quy tuyến tính định lượng độ nhạy sức mua theo tỉ lệ chiết khấu.
7. **Dự báo doanh số (XGBoost Regressor)**: XGBoost + Optuna + KFold cross-validation dự đoán doanh số ngày tiếp theo.
8. **FastAPI Backend**: RESTful API cho dự báo doanh số, phân tích cảm xúc thời gian thực, trích xuất phân khúc sản phẩm.
9. **Streamlit App**: Dashboard kiểm thử nhanh cho Khoa học dữ liệu.
10. **Next.js Frontend Dashboard**: Giao diện giám sát, tương tác và trực quan hóa cho doanh nghiệp.

---

## YÊU CẦU HỆ THỐNG
* **Python**: `>= 3.10`
* **Node.js**: `>= 18.0` (để chạy Next.js dashboard)
* **Google Chrome & Chromedriver**: cùng phiên bản, để chạy Selenium scraper.
* **RAM**: tối thiểu 8GB. Bước benchmark PhoBERT (tùy chọn) dùng ~3-4GB và tự động chạy trên GPU Apple (MPS) / CUDA nếu có, không bắt buộc GPU.

---

## 1. CÀI ĐẶT MÔI TRƯỜNG

### Bước 1.1: Thiết lập Python Virtual Environment
Mở terminal tại thư mục gốc của dự án:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
source .venv/bin/activate

# Cập nhật pip và cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```
Lệnh `pip install -r requirements.txt` đã bao gồm `torch`, `transformers`, `underthesea`, `pyvi` cần cho bước xử lý tiếng Việt và benchmark PhoBERT.

**Lưu ý về tải mô hình PhoBERT (chỉ cho bước benchmark tùy chọn):** Lần đầu chạy benchmark, thư viện `transformers` sẽ tự tải `vinai/phobert-base` (~500MB) về cache `~/.cache/huggingface`. Cần kết nối internet ở lần chạy đầu; các lần sau dùng lại cache, không tải lại.

### Bước 1.2: Cài đặt Next.js Frontend
```bash
cd dashboard
npm install
cp .env.example .env
npm run dev
```
(Nếu FastAPI chạy ở cổng khác `http://localhost:8000`, cập nhật biến `NEXT_PUBLIC_API_BASE_URL` trong `dashboard/.env`.)

---

## 2. QUY TRÌNH CHẠY PIPELINE (END-TO-END)

Chạy tuần tự các lệnh sau từ **thư mục gốc** (đảm bảo `.venv` đã kích hoạt).

### Bước 2.1: Thu thập dữ liệu từ TGDD
```bash
# Cào sản phẩm smartphone và tối đa 2000 reviews/sản phẩm
python scraper/tgdd_scraper.py --category smartphone --max-reviews 2000
```
Kết quả: `products.csv` và `reviews.csv`.

### Bước 2.2: Tiền xử lý dữ liệu & Trích xuất thông số kỹ thuật
Chuẩn hóa dữ liệu thô, trích xuất RAM/ROM/Pin/Camera, làm sạch văn bản và đối chiếu Teen code:
```bash
python preprocessing/preprocess.py
```

### Bước 2.3: Chuẩn bị dữ liệu cảm xúc (làm sạch + gán nhãn 3 lớp + xử lý phủ định)
Đây là bước trọng tâm khai phá dữ liệu của module cảm xúc. Nó:
* Cắt bỏ phần nhiễu hệ thống TGDD chèn sau dấu `||` ("Bộ phận hỗ trợ đã liên hệ...").
* Gán nhãn 3 lớp theo số sao: 1-2 sao = tiêu cực (0), 3 sao = trung tính (1), 4-5 sao = tích cực (2).
* Khử trùng lặp đánh giá.
* Feature engineering xử lý phủ định: gắn tiền tố `neg_` cho từ trong phạm vi phủ định và phát token chỉ báo cực tính dùng chung (ví dụ "không tốt" -> "không neg_tốt neg_pos_marker").
```bash
python preprocessing/prepare_sentiment.py
```
Kết quả: `data-project/processed/reviews_labeled.csv` (in ra phân bố nhãn).

### Bước 2.4: Huấn luyện mô hình Phân loại cảm xúc 3 lớp (TF-IDF + Logistic Regression)
Huấn luyện mô hình chính, đánh giá bằng 5-fold cross-validation (F1/precision/recall macro), lưu mô hình và smoke test trên vài câu phủ định:
```bash
python models/sentiment/train_sentiment.py
```
Kết quả: `models/sentiment/sentiment_clf.pkl`, `label_names.json`, `sentiment_metrics.csv`.

### Bước 2.5 (tùy chọn): Benchmark đặc trưng PhoBERT vs TF-IDF
So sánh khách quan TF-IDF, PhoBERT (đóng băng) và hybrid trên cùng các fold. Lần đầu sẽ tải PhoBERT (~500MB) và trích embedding (vài phút):
```bash
# Trích embedding PhoBERT + huấn luyện classifier trên embedding + so sánh
python models/sentiment/train_phobert_clf.py
# So sánh nhanh 3 nhóm đặc trưng (dùng lại embedding đã cache)
python models/sentiment/compare_features.py
```
Kết quả: `model_comparison.csv`, `feature_comparison.csv`, `confusion_matrix.png`. Kết luận thực nghiệm: trên dữ liệu này TF-IDF mạnh hơn PhoBERT đóng băng (cảm xúc review thiên từ vựng), nên mô hình production dùng TF-IDF.

### Bước 2.6: Phân tích khía cạnh đánh giá (Aspect Analysis)
```bash
python preprocessing/aspect_analysis.py
```

### Bước 2.7: Định lượng độ co giãn giá (Price Elasticity)
```bash
python preprocessing/price_optimization.py
```

### Bước 2.8: Phân cụm phân khúc smartphone (K-Means & PCA)
```bash
python models/clustering/product_cluster.py
```
**Kết quả gán nhãn:** 1.510 tích cực · 753 tiêu cực · 493 trung tính (loại) → sau khi bỏ text rỗng còn **2.075 mẫu** huấn luyện.

### Bước 2.9: Huấn luyện mô hình Dự báo doanh số (XGBoost + Optuna)
```bash
python models/forecasting/train_xgboost.py --trials 100
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

## 3. KHỞI CHẠY ỨNG DỤNG

Sau khi pipeline ở phần 2 chạy xong và các file `.pkl` cùng dữ liệu đã lưu trong `data-project/processed/`, khởi động các dịch vụ:

### Dịch vụ 1: FastAPI Backend (Port 8000)
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows (macOS/Linux: source .venv/bin/activate)
pip install --upgrade pip && pip install -r requirements.txt
```
* Tài liệu API tương tác (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Dịch vụ 2: Next.js Frontend Dashboard (Port 3000)
```bash
python scraper/tgdd_scraper.py --category smartphone --max-reviews 2000  # 1. Crawl
python preprocessing/preprocess.py                                       # 2. Tiền xử lý
python models/sentiment/train_tfidf_rf.py                                # 3. Phân loại cảm xúc
python preprocessing/aspect_analysis.py                                  # 4. ABSA
python models/clustering/product_cluster.py                              # 5. Phân cụm
python preprocessing/price_optimization.py                               # 6. Độ co giãn giá
python models/forecasting/train_xgboost.py --trials 100                  # 7. Dự báo
```
* Địa chỉ truy cập: [http://localhost:3000](http://localhost:3000)

### Dịch vụ 3: Streamlit Analytics Webapp (Port 8501)
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

## 4. CHI TIẾT CÁC ENDPOINT API CHÍNH (FASTAPI)

* `GET /api/v1/products/clusters`: Danh sách sản phẩm kèm tên phân cụm và tọa độ `PC1`/`PC2`.
* `POST /api/v1/sentiment/predict`: Nhận văn bản thô, áp cùng pipeline làm sạch như lúc huấn luyện (cắt nhiễu `||`, chuẩn hóa Teen code, tách từ, gắn phủ định) và trả về:
  * `label` (0 = tiêu cực, 1 = trung tính, 2 = tích cực) và `label_name`,
  * `confidence` của lớp dự đoán,
  * `probabilities` xác suất cả 3 lớp,
  * `cleaned_text` văn bản sau xử lý.
* `POST /api/v1/forecast/sales`: Nhận cấu hình sản phẩm (giá, % chiết khấu, cảm xúc) để dự báo doanh số 7 ngày tới.
* `GET /api/v1/products/{product_id}/cross_sell`: Gợi ý bán kèm phụ kiện theo luật kết hợp Apriori.

---

## 5. CẤU TRÚC MODULE CẢM XÚC (THAM CHIẾU)

| File | Vai trò |
| --- | --- |
| `preprocessing/prepare_sentiment.py` | Làm sạch, gán nhãn 3 lớp, khử trùng lặp, feature engineering phủ định |
| `models/sentiment/train_sentiment.py` | Huấn luyện mô hình production TF-IDF + Logistic Regression (3 lớp) |
| `models/sentiment/phobert_embedder.py` | Trích đặc trưng PhoBERT đóng băng (dùng cho benchmark) |
| `models/sentiment/train_phobert_clf.py` | Huấn luyện classifier trên embedding PhoBERT + đánh giá |
| `models/sentiment/compare_features.py` | So sánh TF-IDF / PhoBERT / hybrid trên cùng các fold |
