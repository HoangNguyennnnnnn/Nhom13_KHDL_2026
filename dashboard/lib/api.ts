const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type ProductCluster = {
  Product_ID: string;
  Name?: string;
  Brand?: string;
  cluster_id?: number;
  cluster_name?: string;
  PC1?: number;
  PC2?: number;
  RAM?: number;
  ROM?: number;
  Discounted_Price?: number;
};

export type ForecastRow = {
  date: string;
  predicted_volume: number;
  trend: string;
};

export type ForecastInput = {
  product_id: string;
  current_price: number;
  discount_rate: number;
  sentiment_score_yesterday: number;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json();
}

async function fetchWithTimeout(url: string, options: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Yeu cau qua thoi gian ${Math.round(timeoutMs / 1000)}s`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function getProductClusters() {
  return getJson<ProductCluster[]>("/api/v1/products/clusters");
}

export async function forecastSales(input: ForecastInput): Promise<ForecastRow[]> {
  const response = await fetchWithTimeout(
    `${API_BASE}/api/v1/forecast/sales`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
    12000
  );
  if (!response.ok) {
    throw new Error(`Forecast failed: ${response.status}`);
  }
  return response.json();
}
