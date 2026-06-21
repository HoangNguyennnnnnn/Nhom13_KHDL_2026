# HƯỚNG DẪN CÀI ĐẶT & CHẠY PIPELINE
## Dự án Phân tích Cảm xúc Khách hàng và Dự báo Doanh số Điện thoại (TGDD)

Dự án là hệ thống phân tích dữ liệu và học máy end-to-end cho thị trường điện thoại di động Thế Giới Di Động (TGDD), bao gồm:

1. **Thu thập dữ liệu**: Trình cào Selenium tự động thu thập thông tin sản phẩm và đánh giá từ website TGDD.
2. **Tiền xử lý & Trích xuất đặc trưng**: Chuẩn hóa đơn vị đo, làm sạch văn bản và dịch từ viết tắt (Teen code) tiếng Việt bằng từ điển tùy chỉnh. Tự động trích xuất RAM/ROM/Pin/Camera từ tên sản phẩm.
3. **Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment)**: Chia đánh giá thành 5 khía cạnh (Pin & Sạc, Hiệu năng, Camera, Thiết kế, Dịch vụ) để tìm ưu/nhược điểm từng sản phẩm.
4. **Phân loại cảm xúc 3 lớp (Sentiment Classification)**: Làm sạch nhiễu, gán nhãn 3 lớp (tiêu cực / trung tính / tích cực) từ số sao, rồi fine-tune mô hình PhoBERT để hiểu ngữ cảnh và phủ định tiếng Việt ("máy không tốt" -> tiêu cực).
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
* **RAM**: tối thiểu 8GB. Fine-tune PhoBERT dùng ~3-4GB, tự động chạy trên GPU Apple (MPS) / CUDA nếu có, không bắt buộc GPU (CPU vẫn train được, chậm hơn).

---
1. CÀI ĐẶT MÔI TRƯỜNG
Bước 1.1: Thiết lập Python Virtual Environment
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
Lệnh `pip install -r requirements.txt` đã bao gồm `underthesea`, `pyvi` (tách từ tiếng Việt) và `torch`, `transformers`, `accelerate` (fine-tune PhoBERT). Lần đầu chạy bước cảm xúc, `transformers` tự tải `vinai/phobert-base` (~500MB) về cache `~/.cache/huggingface` — cần internet ở lần đầu, các lần sau dùng lại cache.

### Bước 1.2: Cài đặt Next.js Frontend
```bash
cd dashboard
npm install
cp .env.example .env
npm run dev
```
(Nếu FastAPI chạy ở cổng khác `http://localhost:8000`, cập nhật biến `NEXT_PUBLIC_API_BASE_URL` trong `dashboard/.env`.)
---
2. QUY TRÌNH CHẠY PIPELINE (END-TO-END)
Chạy tuần tự các lệnh sau từ thư mục gốc (đảm bảo `.venv` đã kích hoạt).
Bước 2.1: Thu thập dữ liệu từ TGDD
```bash
# Cào sản phẩm smartphone và tối đa 2000 reviews/sản phẩm
python scraper/tgdd_scraper.py --category smartphone --max-reviews 2000
```
Kết quả: `products.csv` và `reviews.csv`.
Bước 2.2: Tiền xử lý dữ liệu & Trích xuất thông số kỹ thuật
Chuẩn hóa dữ liệu thô, trích xuất RAM/ROM/Pin/Camera, làm sạch văn bản và đối chiếu Teen code:
```bash
python preprocessing/preprocess.py
```

### Bước 2.3: Chuẩn bị dữ liệu cảm xúc (làm sạch + gán nhãn 3 lớp)
Bước trọng tâm khai phá dữ liệu của module cảm xúc. Nó:
* Cắt bỏ phần nhiễu hệ thống TGDD chèn sau dấu `||` ("Bộ phận hỗ trợ đã liên hệ...").
* Gán nhãn 3 lớp theo số sao: 1-2 sao = tiêu cực (0), 3 sao = trung tính (1), 4-5 sao = tích cực (2).
* Khử trùng lặp đánh giá và tách từ tiếng Việt (PhoBERT cần văn bản đã tách từ).
```bash
python preprocessing/prepare_sentiment.py
```
Kết quả: `data-project/processed/reviews_labeled.csv` (in ra phân bố nhãn).

### Bước 2.4: Fine-tune mô hình Phân loại cảm xúc 3 lớp (PhoBERT)
Fine-tune `vinai/phobert-base` trên dữ liệu đã gán nhãn để hiểu ngữ cảnh và phủ định tiếng Việt. Tự dùng GPU Apple (MPS) / CUDA nếu có, mất ~10-20 phút trên MPS:
```bash
python models/sentiment/train_phobert.py
```
Kết quả: `models/sentiment/phobert_finetuned/` (mô hình + tokenizer), `label_names.json`, `phobert_metrics.json` (metrics trên tập test).

### Bước 2.5: Phân tích khía cạnh đánh giá (Aspect Analysis)
```bash
python preprocessing/aspect_analysis.py
```

### Bước 2.6: Định lượng độ co giãn giá (Price Elasticity)
```bash
python preprocessing/price_optimization.py
```

### Bước 2.7: Phân cụm phân khúc smartphone (K-Means & PCA)
```bash
python models/clustering/product_cluster.py
```

### Bước 2.8: Huấn luyện mô hình Dự báo doanh số (XGBoost + Optuna)
```bash
python models/forecasting/train_xgboost.py --trials 100
```

---

## 3. KHỞI CHẠY ỨNG DỤNG

Sau khi pipeline ở phần 2 chạy xong và các file `.pkl` cùng dữ liệu đã lưu trong `data-project/processed/`, khởi động các dịch vụ:
Dịch vụ 1: FastAPI Backend (Port 8000)
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Tài liệu API tương tác (Swagger UI): http://127.0.0.1:8000/docs
Dịch vụ 2: Next.js Frontend Dashboard (Port 3000)
```bash
cd dashboard
npm run dev
```
* Địa chỉ truy cập: [http://localhost:3000](http://localhost:3000)

### Dịch vụ 3: Streamlit Analytics Webapp (Port 8501)
```bash
streamlit run streamlit_app/app.py
```
* Địa chỉ truy cập: [http://localhost:8501](http://localhost:8501)

---

## 3.1. DEMO NHANH PHÂN TÍCH CẢM XÚC

Điều kiện: đã chạy Bước 2.3 và 2.4 để có `models/sentiment/phobert_finetuned/`.

**Cách 1 — Giao diện Streamlit (khuyến nghị khi demo):**
```bash
streamlit run streamlit_app/app.py
```
Mở tab **"Thử Nghiệm Mô Hình Sentiment"** → gõ một bình luận → bấm **"Phân tích ý kiến"**.
Kết quả hiển thị: nhãn (Tích cực / Trung tính / Tiêu cực), độ tin cậy, xác suất cả 3 lớp, và văn bản sau khi làm sạch (cắt nhiễu `||`, chuẩn hóa, tách từ).

**Cách 2 — Gọi trực tiếp API:**
```bash
# Terminal 1: chạy backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: gửi một bình luận
curl -X POST http://127.0.0.1:8000/api/v1/sentiment/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "máy không tốt, pin tụt nhanh, thất vọng"}'
```
Hoặc mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), chọn `POST /api/v1/sentiment/predict` → "Try it out" → nhập bình luận → "Execute".

---

## 4. CHI TIẾT CÁC ENDPOINT API CHÍNH (FASTAPI)

* `GET /api/v1/products/clusters`: Danh sách sản phẩm kèm tên phân cụm và tọa độ `PC1`/`PC2`.
* `POST /api/v1/sentiment/predict`: Nhận văn bản thô, áp cùng pipeline làm sạch như lúc huấn luyện (cắt nhiễu `||`, chuẩn hóa Teen code, tách từ) rồi đưa qua PhoBERT fine-tuned, trả về:
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
| `preprocessing/prepare_sentiment.py` | Làm sạch nhiễu, gán nhãn 3 lớp, khử trùng lặp, tách từ |
| `models/sentiment/train_phobert.py` | Fine-tune PhoBERT 3 lớp trên dữ liệu đã gán nhãn |
| `models/sentiment/predict.py` | Inference dùng chung cho FastAPI + Streamlit (load PhoBERT, làm sạch, dự đoán) |
