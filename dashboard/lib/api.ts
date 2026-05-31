const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type ProductCluster = {
  Product_ID: string;
  Brand?: string;
  cluster_id?: number;
  cluster_name?: string;
  Discounted_Price?: number;
};

export type ChurnAlert = {
  User_ID: string;
  email?: string;
  churn_probability: number;
  rfm_segment?: number;
};

export type ForecastRow = {
  date: string;
  predicted_volume: number;
  trend: string;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json();
}

export async function getProductClusters() {
  return getJson<ProductCluster[]>("/api/v1/products/clusters");
}

export async function getChurnAlerts() {
  return getJson<ChurnAlert[]>("/api/v1/customers/churn_alert");
}

export async function forecastSales(productId: string): Promise<ForecastRow[]> {
  const response = await fetch(`${API_BASE}/api/v1/forecast/sales`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      product_id: productId,
      current_price: 10000000,
      discount_rate: 0.1,
      sentiment_score_yesterday: 0.7
    })
  });
  if (!response.ok) {
    throw new Error(`Forecast failed: ${response.status}`);
  }
  return response.json();
}
