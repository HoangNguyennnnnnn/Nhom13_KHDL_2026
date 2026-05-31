# Nhom13_KHDL_2026

## Dự án phân tích cảm xúc khách hàng và dự báo doanh số điện thoại

Dự án khoa học dữ liệu end-to-end cho bài toán phân tích dữ liệu điện thoại tại TGDD: thu thập dữ liệu, tiền xử lý, phân loại cảm xúc tiếng Việt, phân cụm sản phẩm và khách hàng, khai phá luật bán chéo, dự báo doanh số, xây dựng API FastAPI, demo Streamlit và dashboard Next.js.

## Cấu trúc dự án

```text
data-project/
  raw/
  processed/
  teencode_dict.json
scraper/tgdd_scraper.py
preprocessing/preprocess.py
models/
  sentiment/
  clustering/
  apriori/
  forecasting/
api/main.py
streamlit_app/app.py
dashboard/
notebooks/
requirements.txt
```

## Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Với Selenium, cần cài Chrome và Chromedriver tương thích. Với PhoBERT, nên fine-tune trên môi trường có GPU.

## Quy trình chạy pipeline

```bash
python scraper/tgdd_scraper.py --limit 30
python preprocessing/preprocess.py
python models/sentiment/train_tfidf_rf.py
python models/clustering/product_cluster.py
python models/clustering/customer_rfm.py --input data-project/raw/purchases.csv
python models/apriori/cross_sell.py --input data-project/raw/baskets.csv
python models/forecasting/train_xgboost.py --trials 100
```

Huấn luyện PhoBERT tùy chọn:

```bash
python models/sentiment/train_phobert.py --epochs 2 --batch-size 8
```

## Chạy API

```bash
uvicorn api.main:app --reload
```

Các endpoint chính:

- `POST /api/v1/forecast/sales`: dự báo doanh số 7 ngày tiếp theo.
- `GET /api/v1/customers/churn_alert`: lấy danh sách khách hàng có nguy cơ rời bỏ cao.
- `GET /api/v1/products/clusters`: lấy danh sách sản phẩm kèm cụm phân khúc.
- `GET /api/v1/products/{product_id}/cross_sell`: gợi ý phụ kiện bán chéo theo sản phẩm.

## Chạy demo Streamlit

```bash
streamlit run streamlit_app/app.py
```

Demo gồm 2 phần:

- Khám phá cụm sản phẩm bằng biểu đồ PCA 2D.
- Nhập đánh giá tiếng Việt để dự đoán cảm xúc bằng TF-IDF + Random Forest.

## Chạy dashboard Next.js

```bash
cd dashboard
npm install
cp .env.example .env
npm run prisma:generate
npm run dev
```

Nếu FastAPI không chạy tại `http://localhost:8000`, cập nhật biến `NEXT_PUBLIC_API_BASE_URL` trong file `.env`.

## Các file dữ liệu đầu vào

`data-project/raw/products.csv`

- `Product_ID`, `Brand`, `Original_Price`, `Discounted_Price`, `Delivery_Options`, `Inward_Date`, `Sales_Volume`, `Avg_Star_Rating`, `Total_Reviews`
- Cột tùy chọn cho mô hình: `RAM`, `ROM`, `Battery`, `Camera_MP`, `Date`

`data-project/raw/reviews.csv`

- `Review_ID`, `Product_ID`, `Review_Date`, `Star_Rating`, `Review_Text`, `Language_Code`

`data-project/raw/purchases.csv`

- `User_ID`, `Purchase_Date`, `Transaction_ID`, `Amount`
- Cột tùy chọn: `email`, `Frequency_Previous`, `Monetary_Previous`

`data-project/raw/baskets.csv`

- `Transaction_ID`, `Product_ID`
- Cột tùy chọn: `Product_Type` với giá trị `smartphone` hoặc `accessory`
