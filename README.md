# Customer Sentiment & Smartphone Sales Forecasting

End-to-end data science project for TGDD smartphone analytics: scraping, preprocessing, Vietnamese sentiment classification, product and customer clustering, cross-sell mining, sales forecasting, FastAPI serving, Streamlit demo, and a Next.js dashboard scaffold.

## Project Layout

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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Selenium, install a compatible Chrome/Chromedriver pair. For PhoBERT fine-tuning, a GPU environment is recommended.

## Pipeline

```bash
python scraper/tgdd_scraper.py --limit 30
python preprocessing/preprocess.py
python models/sentiment/train_tfidf_rf.py
python models/clustering/product_cluster.py
python models/clustering/customer_rfm.py --input data-project/raw/purchases.csv
python models/apriori/cross_sell.py --input data-project/raw/baskets.csv
python models/forecasting/train_xgboost.py --trials 100
```

Optional PhoBERT training:

```bash
python models/sentiment/train_phobert.py --epochs 2 --batch-size 8
```

## APIs

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `POST /api/v1/forecast/sales`
- `GET /api/v1/customers/churn_alert`
- `GET /api/v1/products/clusters`
- `GET /api/v1/products/{product_id}/cross_sell`

## Streamlit Demo

```bash
streamlit run streamlit_app/app.py
```

## Next.js Dashboard

```bash
cd dashboard
npm install
cp .env.example .env
npm run prisma:generate
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` if FastAPI is not running at `http://localhost:8000`.

## Expected Input Files

`data-project/raw/products.csv`

- `Product_ID`, `Brand`, `Original_Price`, `Discounted_Price`, `Delivery_Options`, `Inward_Date`, `Sales_Volume`, `Avg_Star_Rating`, `Total_Reviews`
- Optional modeling columns: `RAM`, `ROM`, `Battery`, `Camera_MP`, `Date`

`data-project/raw/reviews.csv`

- `Review_ID`, `Product_ID`, `Review_Date`, `Star_Rating`, `Review_Text`, `Language_Code`

`data-project/raw/purchases.csv`

- `User_ID`, `Purchase_Date`, `Transaction_ID`, `Amount`
- Optional: `email`, `Frequency_Previous`, `Monetary_Previous`

`data-project/raw/baskets.csv`

- `Transaction_ID`, `Product_ID`
- Optional: `Product_Type` with `smartphone` and `accessory`
