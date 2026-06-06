# HƯỚNG DẪN CÀI ĐẶT & CHẠY PIPELINE
## Dự án Phân tích Cảm xúc Khách hàng và Dự báo Doanh số Điện thoại (TGDD)

Dự án này là hệ thống phân tích dữ liệu và học máy end-to-end cho thị trường điện thoại di động Thế Giới Di Động (TGDD), bao gồm:
1. **Thu thập dữ liệu**: Trình cào dữ liệu Selenium tự động thu thập thông tin sản phẩm và reviews từ website TGDD.
2. **Tiền xử lý & Trích xuất đặc trưng**: Chuẩn hóa đơn vị đo lường, làm sạch dữ liệu văn bản và dịch từ viết tắt (Teen code) tiếng Việt bằng bộ từ điển tùy chỉnh. Tự động trích xuất thông số RAM/ROM/Pin/Camera từ tên sản phẩm.
3. **Phân tích Aspect-Based Sentiment**: Phân chia đánh giá thành 5 khía cạnh trọng tâm (Pin & Sạc, Hiệu năng/Hệ điều hành, Camera, Thiết kế ngoại hình, Dịch vụ/Nhân viên) để tìm ra ưu điểm và nhược điểm (Pros & Cons) của từng sản phẩm.
4. **Học máy không giám sát (Clustering)**: Sử dụng KMeans kết hợp phân tích thành phần chính (PCA) để gom cụm và trực quan hóa các phân khúc smartphone 2D sinh động.
5. **Độ nhạy giá (Price Elasticity)**: Hồi quy tuyến tính định lượng mức độ nhạy cảm của sức mua dựa trên tỉ lệ chiết khấu (Discount Rate), mô phỏng tăng trưởng doanh số khuyến mãi.
6. **Dự báo doanh số (XGBoost Regressor)**: Sử dụng XGBoost kết hợp tối ưu hóa siêu tham số Optuna và KFold cross-validation để dự đoán doanh số bán hàng ngày tiếp theo dựa trên các chỉ số cấu hình, cảm xúc và giá cả.
7. **FastAPI Backend**: Cung cấp các RESTful APIs cho mô hình dự báo doanh số, phân tích cảm xúc thời gian thực và trích xuất phân khúc sản phẩm.
8. **Streamlit App**: Giao diện Dashboard kiểm thử nhanh cho Khoa học dữ liệu.
9. **Next.js Frontend Dashboard**: Giao diện giám sát cao cấp, tương tác và hiển thị biểu đồ phân tích trực quan cho doanh nghiệp.

---

## 🛠️ YÊU CẦU HỆ THỐNG
* **Python**: Phiên bản `>= 3.10`
* **Node.js**: Phiên bản `>= 18.0` (để chạy Next.js dashboard)
* **Google Chrome & Chromedriver**: Trùng phiên bản với nhau để chạy Selenium scraper.

---

## 🚀 1. CÀI ĐẶT MÔI TRƯỜNG

### Bước 1.1: Thiết lập Python Virtual Environment
Mở terminal tại thư mục gốc của dự án và chạy các lệnh sau:
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

### Bước 1.2: Cài đặt Next.js Frontend
```bash
cd dashboard
npm install
cp .env.example .env
npm run dev
```
*(Nếu FastAPI chạy ở cổng khác `http://localhost:8000`, hãy cập nhật biến `NEXT_PUBLIC_API_BASE_URL` trong file `dashboard/.env`)*.

---

## 📈 2. QUY TRÌNH CHẠY PIPELINE (END-TO-END)

Hãy chạy tuần tự các lệnh sau từ **thư mục gốc** (đảm bảo `.venv` đã được kích hoạt):

### Bước 2.1: Thu thập dữ liệu từ TGDD
Thu thập sản phẩm điện thoại di động và đánh giá của khách hàng:
```bash
# Cào không giới hạn số lượng sản phẩm điện thoại và lấy tối đa 2000 reviews/sản phẩm
python scraper/tgdd_scraper.py --category smartphone --max-reviews 2000
```

### Bước 2.2: Tiền xử lý dữ liệu & Trích xuất thông số kỹ thuật
Chuẩn hóa dữ liệu thô, tự động phân tích trích xuất RAM/ROM/Pin/Camera từ tên sản phẩm, làm sạch văn bản đánh giá tiếng Việt và đối chiếu từ điển Teen code:
```bash
python preprocessing/preprocess.py
```

### Bước 2.3: Phân tích khía cạnh đánh giá (Aspect Analysis)
Trích xuất đánh giá của khách hàng thành các khía cạnh và tính điểm hài lòng cùng các Pros & Cons:
```bash
python preprocessing/aspect_analysis.py
```

### Bước 2.4: Định lượng độ co giãn giá (Price Elasticity)
Tính toán hệ số co giãn nhu cầu theo giá bán của từng thương hiệu nhằm hỗ trợ xây dựng chiến lược khuyến mãi thông minh:
```bash
python preprocessing/price_optimization.py
```

### Bước 2.5: Huấn luyện mô hình Phân loại cảm xúc (TF-IDF + Random Forest)
Huấn luyện mô hình NLP Random Forest để nhận diện bình luận Tích cực/Tiêu cực:
```bash
python models/sentiment/train_tfidf_rf.py
```

### Bước 2.6: Phân cụm phân khúc smartphone (K-Means & PCA)
Gom cụm sản phẩm dựa trên cấu hình, giá bán, lượt review. Tự động tính toán tọa độ PCA (`PC1`, `PC2`) để vẽ bản đồ 2D:
```bash
python models/clustering/product_cluster.py
```

### Bước 2.7: Huấn luyện mô hình Dự báo doanh số (XGBoost + Optuna)
Huấn luyện mô hình XGBoost dự đoán doanh số ngày tiếp theo bằng tối ưu hóa Optuna 100 trials:
```bash
python models/forecasting/train_xgboost.py --trials 100
```

---

## 🖥️ 3. KHỞI CHẠY ỨNG DỤNG

Sau khi quy trình chạy pipeline ở phần 2 hoàn thành thành công và các file mô hình `.pkl` cùng dữ liệu đã lưu trữ đầy đủ trong `data-project/processed/`, bạn có thể khởi động các dịch vụ:

### Dịch vụ 1: FastAPI Backend (Port 8000)
Expose các APIs để phục vụ các dự đoán học máy của Next.js:
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Địa chỉ tài liệu API tương tác (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Dịch vụ 2: Next.js Frontend Dashboard (Port 3000)
Giao diện quản trị, tương tác, mô phỏng khuyến mãi, ML Sentiment Room thời gian thực:
```bash
cd dashboard
npm run dev
```
* **Địa chỉ truy cập**: [http://localhost:3000](http://localhost:3000)

### Dịch vụ 3: Streamlit Analytics Webapp (Port 8501)
Ứng dụng kiểm thử nhanh cho Data Science:
```bash
streamlit run streamlit_app/app.py
```

---

## 📋 4. CHI TIẾT CÁC ENDPOINT API CHÍNH (FASTAPI)
Hệ thống Next.js giao tiếp trực tiếp với FastAPI qua các API:

* `GET /api/v1/products/clusters`: Trả về danh sách sản phẩm kèm tên phân cụm tiếng Việt chuẩn hóa và tọa độ `PC1`/`PC2`.
* `POST /api/v1/sentiment/predict`: Nhận vào đoạn văn bản thô, làm sạch teen code và trả về nhãn Cảm xúc (`label`: Tích cực/Tiêu cực) cùng độ tin cậy (`confidence` %) của mô hình Random Forest.
* `POST /api/v1/forecast/sales`: Nhận vào cấu hình sản phẩm hiện tại (Giá, % chiết khấu, Cảm xúc) để dự báo doanh số bán ra trong vòng 7 ngày tới.
* `GET /api/v1/products/{product_id}/cross_sell`: Gợi ý bán kèm phụ kiện phù hợp dựa trên thuật toán luật kết hợp Apriori.
