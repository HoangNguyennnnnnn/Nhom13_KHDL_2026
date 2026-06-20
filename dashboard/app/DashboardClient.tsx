"use client";

import React, { useState } from "react";
import { forecastSales } from "../lib/api";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  LineChart,
  Line,
} from "recharts";

type ProductCluster = {
  Product_ID: string;
  Name?: string;
  Brand?: string;
  cluster_id?: number;
  cluster_name?: string;
  Discounted_Price?: number;
  RAM?: number;
  ROM?: number;
  Battery?: number;
  Camera_MP?: number;
  Avg_Star_Rating?: number;
  Total_Reviews?: number;
  PC1?: number;
  PC2?: number;
};

type AspectDetails = {
  Score: number;
  Pros: string[];
  Cons: string[];
};

type ProductAspects = {
  Product_Name: string;
  Aspects: {
    [key: string]: AspectDetails;
  };
};

type BrandElasticity = {
  Price_Elasticity: number;
  Product_Count: number;
  Average_Discount_Rate: number;
  Average_Sales_Volume: number;
  Model_R2: number;
};

type DashboardClientProps = {
  products: ProductCluster[];
  aspects: { [key: string]: ProductAspects };
  elasticity: { [key: string]: BrandElasticity };
  teencode: { [key: string]: string };
};

export default function DashboardClient({
  products,
  aspects,
  elasticity,
  teencode,
}: DashboardClientProps) {
  const [activeTab, setActiveTab] = useState("phankhuc");
  const [selectedBrand, setSelectedBrand] = useState("Tất cả");
  const [selectedSegment, setSelectedSegment] = useState("Tất cả");

  // Tab 2: Aspect Sentiment
  const aspectProductKeys = Object.keys(aspects);
  const [selectedAspectProductKey, setSelectedAspectProductKey] = useState(
    aspectProductKeys[0] || "",
  );

  // Tab 3: Elasticity Simulator
  const elasticityBrands = Object.keys(elasticity);
  const [simulatorBrand, setSimulatorBrand] = useState(
    elasticityBrands[0] || "",
  );
  const [discountIncrement, setDiscountIncrement] = useState(5);

  // Tab 4: XGBoost Simulator
  const [xgbProdId, setXgbProdId] = useState(
    products[0]?.Product_ID || "iphone-17-pro-max",
  );
  const [xgbPrice, setXgbPrice] = useState(12490000);
  const [xgbDiscountRate, setXgbDiscountRate] = useState(8);
  const [xgbSentimentScore, setXgbSentimentScore] = useState(0.75);
  const [forecastResults, setForecastResults] = useState<any[]>([]);
  const [forecasting, setForecasting] = useState(false);
  const [forecastError, setForecastError] = useState("");
  const [compareSortBy, setCompareSortBy] = useState<
    "name" | "brand" | "price" | "ramrom" | "rating" | "reviews"
  >("price");
  const [compareSortDir, setCompareSortDir] = useState<"asc" | "desc">("asc");

  // Tab 5: ML Sentiment Room (1:1 identical to Streamlit App Tab 5)
  const [sentimentInput, setSentimentInput] = useState("");
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentResult, setSentimentResult] = useState<{
    label: number; // 0 = negative, 1 = neutral, 2 = positive
    label_name: string;
    confidence: number;
    probabilities?: Record<string, number>;
    cleaned_text: string;
  } | null>(null);
  const [sentimentError, setSentimentError] = useState("");

  // Tab 6: Teencode translator (client-side using the provided dictionary)
  const [teencodeInput, setTeencodeInput] = useState("");
  const [translatedText, setTranslatedText] = useState("");

  const handleTranslate = () => {
    const tokens = teencodeInput.toLowerCase().split(/\s+/).filter(Boolean);
    const translated = tokens
      .map((token) => {
        const stripped = token.replace(
          /^[.,!?"'()\[\]{}]+|[.,!?"'()\[\]{}]+$/g,
          "",
        );
        const replacement = teencode[stripped];
        return replacement ? token.replace(stripped, replacement) : token;
      })
      .join(" ");
    setTranslatedText(translated);
  };

  const handleForecast = async () => {
    setForecasting(true);
    setForecastError("");
    try {
      const data = await forecastSales({
        product_id: xgbProdId,
        current_price: xgbPrice,
        discount_rate: xgbDiscountRate / 100,
        sentiment_score_yesterday: xgbSentimentScore,
      });
      setForecastResults(data);
    } catch (err) {
      setForecastResults([]);
      const message = err instanceof Error ? err.message : String(err);
      setForecastError(
        `Lỗi backend dự báo: ${message}. Hãy kiểm tra API FastAPI ở cổng 8000.`,
      );
    } finally {
      setForecasting(false);
    }
  };

  const handlePredictSentiment = async () => {
    if (!sentimentInput.trim()) {
      alert("Vui lòng nhập nội dung đánh giá cần phân tích.");
      return;
    }
    setSentimentLoading(true);
    setSentimentError("");
    setSentimentResult(null);
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/v1/sentiment/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentimentInput }),
        },
      );
      if (response.ok) {
        const data = await response.json();
        setSentimentResult(data);
      } else {
        setSentimentError(
          "Không thể kết nối đến backend API. Hãy đảm bảo FastAPI đã khởi động thành công!",
        );
      }
    } catch (err) {
      setSentimentError("Lỗi kết nối FastAPI backend: " + err);
    } finally {
      setSentimentLoading(false);
    }
  };

  const handleCompareSort = (
    key: "name" | "brand" | "price" | "ramrom" | "rating" | "reviews",
  ) => {
    if (compareSortBy === key) {
      setCompareSortDir(compareSortDir === "asc" ? "desc" : "asc");
      return;
    }
    setCompareSortBy(key);
    setCompareSortDir("asc");
  };

  const getSortIndicator = (
    key: "name" | "brand" | "price" | "ramrom" | "rating" | "reviews",
  ) => {
    if (compareSortBy !== key) return "";
    return compareSortDir === "asc" ? "" : "";
  };

  // Filter products by brand and segment
  const brandFilteredProducts =
    selectedBrand === "Tất cả"
      ? products
      : products.filter((p) => p.Brand === selectedBrand);

  const segments = Array.from(
    new Set(
      brandFilteredProducts
        .map((p) => p.cluster_name)
        .filter((value): value is string => Boolean(value)),
    ),
  );

  const filteredProducts =
    selectedSegment === "Tất cả"
      ? brandFilteredProducts
      : brandFilteredProducts.filter(
          (p) => (p.cluster_name || "Không xác định") === selectedSegment,
        );

  const brands = Array.from(
    new Set(products.map((p) => p.Brand).filter(Boolean)),
  );

  // Generate color palette for K-Means clusters
  const clusterColors: { [key: string]: string } = {
    "Điện thoại giá rẻ - Phổ thông cơ bản": "#e53e3e",
    "Tầm trung - Cấu hình tốt & Pin khoẻ": "#3182ce",
    "Tầm trung - Tiết kiệm & Đủ dùng": "#dd6b20",
    "Cận cao cấp - Hiệu năng & Đa năng": "#319795",
    "Cận cao cấp - Thiết kế & Đủ dùng": "#805ad5",
    "Cao cấp - Siêu phẩm công nghệ": "#2b6cb0",
  };

  // Prepare PCA Scatter plot data
  const scatterData = filteredProducts.map((p) => ({
    x: Number.isFinite(Number(p.PC1)) ? Number(p.PC1) : 0,
    y: Number.isFinite(Number(p.PC2)) ? Number(p.PC2) : 0,
    name: p.Name || p.Product_ID,
    cluster: p.cluster_name || "Không xác định",
    price: p.Discounted_Price,
    ram: p.RAM,
    rom: p.ROM,
  }));

  // Aspect sentiment chart data
  const selectedProdData = aspects[selectedAspectProductKey];
  const aspectChartData = selectedProdData
    ? Object.keys(selectedProdData.Aspects).map((aspectName) => ({
        name: aspectName,
        Score: selectedProdData.Aspects[aspectName].Score * 100,
      }))
    : [];

  // Elasticity chart data
  const elasticityChartData = Object.keys(elasticity).map((brand) => ({
    name: brand,
    "Độ co giãn": elasticity[brand].Price_Elasticity,
  }));

  const comparedRows = [...filteredProducts].sort((a, b) => {
    let left: string | number = 0;
    let right: string | number = 0;

    if (compareSortBy === "name") {
      left = a.Name || a.Product_ID;
      right = b.Name || b.Product_ID;
    } else if (compareSortBy === "brand") {
      left = a.Brand || "";
      right = b.Brand || "";
    } else if (compareSortBy === "price") {
      left = Number(a.Discounted_Price || 0);
      right = Number(b.Discounted_Price || 0);
    } else if (compareSortBy === "ramrom") {
      left = Number(a.RAM || 0) * 10000 + Number(a.ROM || 0);
      right = Number(b.RAM || 0) * 10000 + Number(b.ROM || 0);
    } else if (compareSortBy === "rating") {
      left = Number(a.Avg_Star_Rating || 0);
      right = Number(b.Avg_Star_Rating || 0);
    } else if (compareSortBy === "reviews") {
      left = Number(a.Total_Reviews || 0);
      right = Number(b.Total_Reviews || 0);
    }

    let result = 0;
    if (typeof left === "string" && typeof right === "string") {
      result = left.localeCompare(right, "vi", { sensitivity: "base" });
    } else {
      result = Number(left) - Number(right);
    }

    return compareSortDir === "asc" ? result : -result;
  });

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand-logo">
          <img
            src="https://img.icons8.com/clouds/200/smart-phone.png"
            alt="logo"
            width="64"
          />
          <h2>TGDD Smart Analytics</h2>
        </div>
        <nav className="nav-menu">
          <button
            className={activeTab === "phankhuc" ? "active" : ""}
            onClick={() => setActiveTab("phankhuc")}
          >
            Phân Khúc K-Means & PCA
          </button>
          <button
            className={activeTab === "sentiment" ? "active" : ""}
            onClick={() => setActiveTab("sentiment")}
          >
            Ý Kiến Khách Hàng (Aspect)
          </button>
          <button
            className={activeTab === "danhygia" ? "active" : ""}
            onClick={() => setActiveTab("danhygia")}
          >
            Độ Nhạy Giá & Khuyến Mãi
          </button>
          <button
            className={activeTab === "forecasting" ? "active" : ""}
            onClick={() => setActiveTab("forecasting")}
          >
            Dự Báo Doanh Số (XGBoost)
          </button>
          <button
            className={activeTab === "sentiment_room" ? "active" : ""}
            onClick={() => setActiveTab("sentiment_room")}
          >
            ML Sentiment Room
          </button>
          <button
            className={activeTab === "teencode" ? "active" : ""}
            onClick={() => setActiveTab("teencode")}
          >
            Từ Điển Teen Code
          </button>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="main-viewport">
        {/* Global Header Metrics */}
        <section className="kpi-ribbon">
          <div className="kpi-card">
            <span className="label">Tổng số sản phẩm</span>
            <strong className="value">{products.length}</strong>
            <span className="desc">Thế Giới Di Động</span>
          </div>
          <div className="kpi-card">
            <span className="label">Tổng thương hiệu</span>
            <strong className="value warning">{brands.length}</strong>
            <span className="desc">Đang được theo dõi</span>
          </div>
          <div className="kpi-card">
            <span className="label">Khung dự báo tối ưu</span>
            <strong className="value">7 Ngày</strong>
            <span className="desc">API Machine Learning</span>
          </div>
        </section>

        {/* Tab Content 1: PCA Product Segmentation */}
        {activeTab === "phankhuc" && (
          <div className="tab-pane animate-fade">
            <div className="pane-header">
              <h2>Bản Đồ Phân Phân Khúc Điện Thoại 2D (K-Means & PCA)</h2>
              <div className="filters">
                <label>Lọc Thương hiệu: </label>
                <select
                  value={selectedBrand}
                  onChange={(e) => setSelectedBrand(e.target.value)}
                >
                  <option value="Tất cả">Tất cả thương hiệu</option>
                  {brands.map((b) => (
                    <option key={b} value={b}>
                      {b}
                    </option>
                  ))}
                </select>
                <label style={{ marginLeft: "12px" }}>Lọc Phân khúc: </label>
                <select
                  value={selectedSegment}
                  onChange={(e) => setSelectedSegment(e.target.value)}
                >
                  <option value="Tất cả">Tất cả phân khúc</option>
                  {segments.map((seg) => (
                    <option key={seg} value={seg}>
                      {seg}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="explanation-box">
              <h4>Giải thích trực quan về mặt Khoa học Dữ liệu:</h4>
              <ul>
                <li>
                  <b>Trục Ngang (PC1):</b> Đại diện cho{" "}
                  <b>Cấu hình & Giá tiền</b>. Càng đi sang phải, máy càng đắt,
                  RAM/ROM khủng và camera xịn (Cao cấp).
                </li>
                <li>
                  <b>Trục Đứng (PC2):</b> Đại diện cho{" "}
                  <b>Sức mua & Lượt tương tác</b>. Càng lên cao, máy càng có
                  doanh số bán ra lớn từ Thế Giới Di Động.
                </li>
                <li>
                  <b>Mỗi Màu sắc:</b> Đại diện cho một phân khúc giá và cấu hình
                  rõ ràng, giúp so sánh đối thủ cạnh tranh trực tiếp một cách
                  công bằng nhất.
                </li>
              </ul>
            </div>

            {/* PCA Scatter Chart */}
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="PC1"
                    label={{
                      value: "Độ Cao Cấp & Giá Bán (PC1)",
                      position: "bottom",
                      offset: 0,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="PC2"
                    label={{
                      value: "Sức Mua & Lượt Đánh Giá (PC2)",
                      angle: -90,
                      position: "left",
                    }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="custom-tooltip">
                            <strong className="tooltip-title">
                              {data.name}
                            </strong>
                            <p>Phân khúc: {data.cluster}</p>
                            <p>Giá: {data.price?.toLocaleString("vi-VN")} đ</p>
                            <p>
                              RAM: {data.ram} GB | ROM: {data.rom} GB
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter name="Điện thoại" data={scatterData}>
                    {scatterData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={clusterColors[entry.cluster] || "#4a5568"}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="table-wrapper" style={{ marginTop: "16px" }}>
              <h3>So sánh các sản phẩm trong cùng phân khúc</h3>
              {selectedSegment === "Tất cả" ? (
                <p>
                  Chọn một phân khúc ở bộ lọc phía trên để mở bảng so sánh trực
                  tiếp trong cùng nhóm.
                </p>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th
                        style={{ cursor: "pointer" }}
                        onClick={() => handleCompareSort("name")}
                      >
                        Sản phẩm {getSortIndicator("name")}
                      </th>
                      <th
                        style={{ cursor: "pointer" }}
                        onClick={() => handleCompareSort("brand")}
                      >
                        Thương hiệu {getSortIndicator("brand")}
                      </th>
                      <th
                        style={{ cursor: "pointer" }}
                        onClick={() => handleCompareSort("price")}
                      >
                        Giá {getSortIndicator("price")}
                      </th>
                      <th
                        style={{ cursor: "pointer" }}
                        onClick={() => handleCompareSort("ramrom")}
                      >
                        RAM/ROM {getSortIndicator("ramrom")}
                      </th>
                      <th
                        style={{ cursor: "pointer" }}
                        onClick={() => handleCompareSort("rating")}
                      >
                        Điểm sao {getSortIndicator("rating")}
                      </th>
                      <th
                        style={{ cursor: "pointer" }}
                        onClick={() => handleCompareSort("reviews")}
                      >
                        Tổng đánh giá {getSortIndicator("reviews")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparedRows.map((p) => (
                      <tr key={`cmp-${p.Product_ID}`}>
                        <td>
                          <strong>{p.Name || p.Product_ID}</strong>
                        </td>
                        <td>{p.Brand || "N/A"}</td>
                        <td>
                          {Number(p.Discounted_Price || 0).toLocaleString(
                            "vi-VN",
                          )}{" "}
                          đ
                        </td>
                        <td>
                          {p.RAM ?? "?"}GB / {p.ROM ?? "?"}GB
                        </td>
                        <td>{Number(p.Avg_Star_Rating || 0).toFixed(1)}</td>
                        <td>
                          {Number(p.Total_Reviews || 0).toLocaleString("vi-VN")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            {/* Products Table */}
            <div className="table-wrapper">
              <h3>Danh sách sản phẩm chi tiết theo Phân khúc</h3>
              <table>
                <thead>
                  <tr>
                    <th>Tên Sản Phẩm</th>
                    <th>Thương Hiệu</th>
                    <th>Giá Bán Khuyến Mãi</th>
                    <th>Cấu Hình (RAM/ROM)</th>
                    <th>Phân Cụm / Segment</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredProducts.map((p) => (
                    <tr key={p.Product_ID}>
                      <td>
                        <strong>{p.Product_ID}</strong>
                      </td>
                      <td>{p.Brand}</td>
                      <td>
                        <span className="price-tag">
                          {p.Discounted_Price?.toLocaleString("vi-VN")} đ
                        </span>
                      </td>
                      <td>
                        {p.RAM}GB / {p.ROM}GB
                      </td>
                      <td>
                        <span
                          className="segment-badge"
                          style={{
                            backgroundColor:
                              clusterColors[p.cluster_name || ""] + "1A",
                            color: clusterColors[p.cluster_name || ""],
                          }}
                        >
                          ● {p.cluster_name}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab Content 2: Aspect Sentiment (Pros & Cons) */}
        {activeTab === "sentiment" && (
          <div className="tab-pane animate-fade">
            <div className="pane-header">
              <h2>
                Phân Tích Ý Kiến Người Dùng & Ưu/Nhược Điểm Từng Sản Phẩm
                (Pros & Cons)
              </h2>
              <div className="filters">
                <label>Chọn sản phẩm: </label>
                <select
                  value={selectedAspectProductKey}
                  onChange={(e) => setSelectedAspectProductKey(e.target.value)}
                >
                  {aspectProductKeys.map((k) => (
                    <option key={k} value={k}>
                      {aspects[k].Product_Name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {selectedProdData ? (
              <>
                {/* Satisfaction scores bar chart */}
                <div className="chart-wrapper">
                  <h3>Chỉ Số Hài Lòng Theo Khía Cạnh (%)</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={aspectChartData}
                      layout="vertical"
                      margin={{ left: 30, right: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                      <XAxis type="number" domain={[0, 100]} />
                      <YAxis type="category" dataKey="name" />
                      <Tooltip
                        formatter={(value) => [
                          `${Number(value).toFixed(1)}%`,
                          "Độ hài lòng",
                        ]}
                      />
                      <Bar dataKey="Score" radius={[0, 4, 4, 0]}>
                        {aspectChartData.map((entry, index) => {
                          const color =
                            entry.Score >= 60
                              ? "#48bb78"
                              : entry.Score >= 45
                                ? "#ecc94b"
                                : "#f56565";
                          return <Cell key={`cell-${index}`} fill={color} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Pros and Cons panels */}
                <div className="pros-cons-grid">
                  <div className="panel strength">
                    <h4>Ưu Điểm Nổi Bật (Strengths)</h4>
                    <ul>
                      {Object.keys(selectedProdData.Aspects).map((aspect) => {
                        const details = selectedProdData.Aspects[aspect];
                        if (details.Score >= 0.6) {
                          return (
                            <li key={aspect}>
                              <strong>
                                {aspect} ({Math.round(details.Score * 100)}%):
                              </strong>
                              <p className="quote">
                                “
                                {details.Pros[0] ||
                                  "Khách hàng rất hài lòng về mặt này."}
                                ”
                              </p>
                            </li>
                          );
                        }
                        return null;
                      })}
                    </ul>
                  </div>

                  <div className="panel weakness">
                    <h4>Điểm Cần Cải Thiện (Weaknesses)</h4>
                    <ul>
                      {Object.keys(selectedProdData.Aspects).map((aspect) => {
                        const details = selectedProdData.Aspects[aspect];
                        if (details.Score < 0.45) {
                          return (
                            <li key={aspect}>
                              <strong>
                                {aspect} ({Math.round(details.Score * 100)}%):
                              </strong>
                              <p className="quote negative">
                                “
                                {details.Cons[0] ||
                                  "Ghi nhận một số phàn nàn nhẹ."}
                                ”
                              </p>
                            </li>
                          );
                        }
                        return null;
                      })}
                    </ul>
                  </div>
                </div>
              </>
            ) : (
              <p>Chưa có dữ liệu phân tích khía cạnh.</p>
            )}
          </div>
        )}

        {/* Tab Content 3: Price Elasticity Simulator */}
        {activeTab === "danhygia" && (
          <div className="tab-pane animate-fade">
            <div className="pane-header">
              <h2>Độ Nhạy Giá & Lập Kế Hoạch Chiến Lược Khuyến Mãi</h2>
            </div>

            <div className="chart-wrapper">
              <h3>So Sánh Hệ Số Co Giãn Giá Theo Từng Thương Hiệu</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={elasticityChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar
                    dataKey="Độ co giãn"
                    fill="#319795"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Elasticity Promo Simulator */}
            <div className="panel promo-simulator">
              <h3>Trình mô phỏng Kế hoạch Khuyến mãi thông minh</h3>
              <div className="simulator-grid">
                <div className="controls">
                  <label>1. Chọn Thương Hiệu: </label>
                  <select
                    value={simulatorBrand}
                    onChange={(e) => setSimulatorBrand(e.target.value)}
                  >
                    {elasticityBrands.map((b) => (
                      <option key={b} value={b}>
                        {b}
                      </option>
                    ))}
                  </select>

                  <label style={{ marginTop: "16px", display: "block" }}>
                    2. Tỷ lệ tăng thêm chiết khấu dự kiến (%):{" "}
                    <strong>+{discountIncrement}%</strong>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="30"
                    value={discountIncrement}
                    onChange={(e) =>
                      setDiscountIncrement(Number(e.target.value))
                    }
                    className="slider"
                  />
                </div>

                <div className="results">
                  {elasticity[simulatorBrand] ? (
                    (() => {
                      const info = elasticity[simulatorBrand];
                      const elas = info.Price_Elasticity;
                      const avgSales = info.Average_Sales_Volume;
                      const growthPercent = elas * discountIncrement;
                      const addedSales = (growthPercent / 100) * avgSales;
                      const directionWord =
                        growthPercent >= 0 ? "tăng" : "giảm";
                      const growthSign = growthPercent >= 0 ? "+" : "-";
                      return (
                        <div className="alert-success">
                          <h4>Ước tính Kết quả cho {simulatorBrand}:</h4>
                          <ul>
                            <li>
                              Độ nhạy giá (Elasticity Coefficient):{" "}
                              <code>{elas.toFixed(4)}</code>
                            </li>
                            <li>
                              Doanh số TB hiện tại:{" "}
                              <code>
                                {avgSales.toLocaleString("vi-VN")} máy
                              </code>
                            </li>
                            <li>
                              Lượng hàng bán ra dự kiến {directionWord}:{" "}
                              <strong className="highlight">
                                {growthSign}
                                {Math.abs(addedSales).toFixed(1)} máy
                              </strong>
                            </li>
                            <li>
                              Tỷ lệ tăng trưởng doanh số:{" "}
                              <strong className="highlight">
                                {growthSign}
                                {Math.abs(growthPercent).toFixed(2)}%
                              </strong>
                            </li>
                          </ul>
                        </div>
                      );
                    })()
                  ) : (
                    <p>Chưa chọn thương hiệu hoặc không có dữ liệu.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab Content 4: Sales Forecasting */}
        {activeTab === "forecasting" && (
          <div className="tab-pane animate-fade">
            <div className="pane-header">
              <h2>
                Mô Hình Học Máy Dự Báo Doanh Số Smartphone Tương Lai
                (XGBoost)
              </h2>
            </div>

            <div className="forecast-simulator-panel">
              <div className="form-column">
                <h3>Cấu Hình Máy Cần Dự Báo</h3>

                <label>Sản phẩm (ID):</label>
                <select
                  value={xgbProdId}
                  onChange={(e) => setXgbProdId(e.target.value)}
                >
                  {products.map((p) => (
                    <option key={p.Product_ID} value={p.Product_ID}>
                      {p.Product_ID}
                    </option>
                  ))}
                </select>

                <label>Giá bán dự kiến (đ):</label>
                <input
                  type="number"
                  value={xgbPrice}
                  onChange={(e) => setXgbPrice(Number(e.target.value))}
                  min="500000"
                  max="50000000"
                />

                <label>Tỷ lệ giảm giá (%):</label>
                <input
                  type="number"
                  value={xgbDiscountRate}
                  onChange={(e) => setXgbDiscountRate(Number(e.target.value))}
                  min="0"
                  max="80"
                />

                <label>Chỉ số cảm xúc bình luận (Sentiment Score 0-1):</label>
                <input
                  type="number"
                  value={xgbSentimentScore}
                  onChange={(e) => setXgbSentimentScore(Number(e.target.value))}
                  step="0.05"
                  min="0"
                  max="1"
                />

                <button
                  className="btn-predict"
                  onClick={handleForecast}
                  disabled={forecasting}
                >
                  {forecasting
                    ? "Đang tính toán..."
                    : "Chạy Dự Báo 7 Ngày"}
                </button>
                {forecastError && (
                  <p style={{ color: "#e53e3e", marginTop: 12 }}>
                    {forecastError}
                  </p>
                )}
              </div>

              <div className="chart-column">
                <h3>Kết Quả Dự Báo Doanh Số (7 Ngày Tới)</h3>
                {forecastResults.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={forecastResults}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="predicted_volume"
                          name="Lượng máy bán ra"
                          stroke="#e43f5a"
                          strokeWidth={3}
                        />
                      </LineChart>
                    </ResponsiveContainer>

                    <div className="forecast-table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>Ngày</th>
                            <th>Doanh số dự đoán</th>
                            <th>Xu hướng</th>
                          </tr>
                        </thead>
                        <tbody>
                          {forecastResults.map((row) => (
                            <tr key={row.date}>
                              <td>{row.date}</td>
                              <td>
                                <strong>{row.predicted_volume} máy</strong>
                              </td>
                              <td>
                                <span className={`trend-badge ${row.trend}`}>
                                  {row.trend === "up"
                                    ? "▲ Tăng"
                                    : row.trend === "down"
                                      ? "▼ Giảm"
                                      : "● Ổn định"}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <div className="placeholder-forecast">
                    <p>
                      Nhấp nút <b>"Chạy Dự Báo 7 Ngày"</b> để xem đồ thị trực
                      quan và bảng số liệu phân tích của mô hình XGBoost.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab Content 5: ML Sentiment Room (1:1 with Streamlit Tab 5) */}
        {activeTab === "sentiment_room" && (
          <div className="tab-pane animate-fade">
            <div className="pane-header">
              <h2>
                Trải Nghiệm Mô Hình Nhận Diện Ý Kiến Đánh Giá (TF-IDF +
                Random Forest)
              </h2>
            </div>

            <div className="explanation-box">
              <h4>Giải thích trực quan về mặt Khoa học Dữ liệu (NLP):</h4>
              <ul>
                <li>
                  Bình luận bạn nhập sẽ tự động chạy qua bộ dịch **Teen code &
                  viết tắt** để chuẩn hoá từ ngữ.
                </li>
                <li>
                  Hệ thống chạy qua bộ trích xuất đặc trưng **TF-IDF** và đưa
                  vào mô hình học máy **Random Forest Classifier** đã huấn luyện
                  để phân tích ý kiến chính xác.
                </li>
              </ul>
            </div>

            <div
              className="teencode-tool-grid"
              style={{ gridTemplateColumns: "1fr" }}
            >
              <div className="translator-panel">
                <h3>Nhập đánh giá của khách hàng cần kiểm thử</h3>
                <textarea
                  placeholder="Ví dụ: máy xài siu ngon, pin cực trâu k giật lag j cả, camera chụp rất nét, sạc pin hơi nóng máy tí..."
                  value={sentimentInput}
                  onChange={(e) => setSentimentInput(e.target.value)}
                  rows={5}
                />
                <button
                  className="btn-predict"
                  onClick={handlePredictSentiment}
                  disabled={sentimentLoading}
                  style={{ marginTop: "16px" }}
                >
                  {sentimentLoading
                    ? "Đang phân tích cảm xúc..."
                    : "Phân Tích Cảm Xúc ML"}
                </button>
                {sentimentError && (
                  <p style={{ color: "#e53e3e", marginTop: 12 }}>
                    {sentimentError}
                  </p>
                )}

                {sentimentResult && (
                  <div
                    className="translation-result animate-fade"
                    style={{ marginTop: "24px" }}
                  >
                    <div
                      style={{
                        display: "flex",
                        gap: "20px",
                        marginBottom: "16px",
                      }}
                    >
                      <div
                        style={{
                          flex: 1,
                          backgroundColor: "#fff",
                          padding: "12px",
                          borderRadius: "6px",
                          borderLeft: "4px solid #3182ce",
                        }}
                      >
                        <strong>Văn bản gốc:</strong>
                        <p
                          style={{
                            fontStyle: "italic",
                            marginTop: "4px",
                            color: "#4a5568",
                          }}
                        >
                          “{sentimentInput}”
                        </p>
                      </div>
                      <div
                        style={{
                          flex: 1,
                          backgroundColor: "#fff",
                          padding: "12px",
                          borderRadius: "6px",
                          borderLeft: "4px solid #38a169",
                        }}
                      >
                        <strong>Đã dịch Teen code & chuẩn hoá:</strong>
                        <p
                          style={{
                            fontStyle: "italic",
                            marginTop: "4px",
                            color: "#2f855a",
                          }}
                        >
                          “{sentimentResult.cleaned_text}”
                        </p>
                      </div>
                    </div>

                    {(() => {
                      const SENTIMENT_UI: Record<
                        number,
                        { bg: string; border: string; color: string; bar: string; title: string }
                      > = {
                        2: { bg: "#f0fff4", border: "#c6f6d5", color: "#22543d", bar: "#48bb78", title: "Kết quả: Tích cực (Positive)" },
                        1: { bg: "#fffaf0", border: "#feebc8", color: "#7b341e", bar: "#ed8936", title: "Kết quả: Trung tính (Neutral)" },
                        0: { bg: "#fff5f5", border: "#fed7d7", color: "#742a2a", bar: "#e53e3e", title: "Kết quả: Tiêu cực / Góp ý (Negative)" },
                      };
                      const ui = SENTIMENT_UI[sentimentResult.label] ?? SENTIMENT_UI[1];
                      const probs = sentimentResult.probabilities ?? {};
                      return (
                        <div
                          style={{
                            padding: "20px",
                            borderRadius: "8px",
                            backgroundColor: ui.bg,
                            border: `1px solid ${ui.border}`,
                            textAlign: "center",
                          }}
                        >
                          <h3 style={{ color: ui.color }}>{ui.title}</h3>

                          <div style={{ marginTop: "16px" }}>
                            <span
                              style={{
                                fontWeight: "700",
                                fontSize: "14px",
                                color: "#4a5568",
                              }}
                            >
                              Độ tin cậy của mô hình (TF-IDF + Logistic Regression):{" "}
                              {(sentimentResult.confidence * 100).toFixed(2)}%
                            </span>
                            <div
                              style={{
                                width: "100%",
                                height: "12px",
                                backgroundColor: "#edf2f7",
                                borderRadius: "6px",
                                overflow: "hidden",
                                marginTop: "8px",
                              }}
                            >
                              <div
                                style={{
                                  width: `${sentimentResult.confidence * 100}%`,
                                  height: "100%",
                                  backgroundColor: ui.bar,
                                }}
                              />
                            </div>
                          </div>

                          {Object.keys(probs).length > 0 && (
                            <div
                              style={{
                                marginTop: "16px",
                                display: "flex",
                                justifyContent: "center",
                                gap: "16px",
                                flexWrap: "wrap",
                                fontSize: "13px",
                                color: "#4a5568",
                              }}
                            >
                              {["negative", "neutral", "positive"].map((k) =>
                                probs[k] !== undefined ? (
                                  <span key={k}>
                                    {k === "positive" ? "Tích cực" : k === "neutral" ? "Trung tính" : "Tiêu cực"}:{" "}
                                    <strong>{(probs[k] * 100).toFixed(1)}%</strong>
                                  </span>
                                ) : null,
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab Content 6: Teencode Dictionary */}
        {activeTab === "teencode" && (
          <div className="tab-pane animate-fade">
            <div className="pane-header">
              <h2>Bộ Từ Điển Tiền Xử Lý Teen Code Tiếng Việt</h2>
            </div>

            <div className="teencode-tool-grid">
              <div className="translator-panel">
                <h3>Trình thử nghiệm dịch từ viết tắt nhanh</h3>
                <textarea
                  placeholder="Nhập đoạn bình luận ngắn chứa teencode để thử nghiệm dịch..."
                  value={teencodeInput}
                  onChange={(e) => setTeencodeInput(e.target.value)}
                  rows={4}
                />
                <button
                  className="btn-translate"
                  onClick={handleTranslate}
                  style={{ marginTop: "12px" }}
                >
                  Dịch Dữ Liệu
                </button>

                {translatedText && (
                  <div
                    className="translation-result animate-fade"
                    style={{ marginTop: "16px" }}
                  >
                    <h5>Bản dịch chuẩn hoá:</h5>
                    <p style={{ fontStyle: "italic" }}>“{translatedText}”</p>
                  </div>
                )}
              </div>

              <div className="dictionary-panel">
                <h3>Danh mục bộ từ điển đối sánh</h3>
                <div className="dict-list-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Từ gốc / Teen code</th>
                        <th>Từ thay thế chuẩn mực</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(teencode).map((key) => (
                        <tr key={key}>
                          <td>
                            <code>{key}</code>
                          </td>
                          <td>{teencode[key]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
