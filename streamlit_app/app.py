"""Streamlit Dashboard for Vietnamese Smartphone Analytics, Aspect Sentiment & Price Elasticity."""

from __future__ import annotations

import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set page configuration for a premium, wide dashboard layout
st.set_page_config(
    page_title="Hệ Thống Phân Tích KHDL Smartphone TGDD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1f4068;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f4068;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .explanation-box {
        background-color: #edf2f7;
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 5px solid #e43f5a;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📈 HỆ THỐNG PHÂN TÍCH THỊ TRƯỜNG & TỐI ƯU HÓA SMARTPHONE (TGDD)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Khoa học Dữ liệu, Học máy Không giám sát, Aspect-Based Sentiment và Dự báo Doanh số</div>', unsafe_allow_html=True)

# ----------------- LOAD DATA & PRESETS -----------------
@st.cache_data
def load_data():
    prod_path = Path("data-project/processed/products_clustered.csv")
    reviews_path = Path("data-project/processed/reviews_scored.csv")
    elasticity_path = Path("data-project/processed/price_elasticity.json")
    aspects_path = Path("data-project/processed/aspect_sentiment.json")
    teencode_path = Path("data-project/teencode_dict.json")
    
    df_prod = pd.read_csv(prod_path) if prod_path.exists() else pd.DataFrame()
    df_reviews = pd.read_csv(reviews_path) if reviews_path.exists() else pd.DataFrame()
    
    elasticity_dict = {}
    if elasticity_path.exists():
        with open(elasticity_path, "r", encoding="utf-8") as f:
            elasticity_dict = json.load(f)
            
    aspects_dict = {}
    if aspects_path.exists():
        with open(aspects_path, "r", encoding="utf-8") as f:
            aspects_dict = json.load(f)
            
    teencode_dict = {}
    if teencode_path.exists():
        with open(teencode_path, "r", encoding="utf-8") as f:
            teencode_dict = json.load(f)
            
    return df_prod, df_reviews, elasticity_dict, aspects_dict, teencode_dict

df_prod, df_reviews, elasticity_dict, aspects_dict, teencode_dict = load_data()

def clean_teencode(text: str, dict_map: dict) -> str:
    if not isinstance(text, str):
        return ""
    words = text.lower().split()
    cleaned_words = []
    for w in words:
        w_clean = w.strip(".,!?\"'()[]{}")
        if w_clean in dict_map:
            w = w.replace(w_clean, dict_map[w_clean])
        cleaned_words.append(w)
    return " ".join(cleaned_words)


# ----------------- SIDEBAR & GLOBAL METRICS -----------------
st.sidebar.image("https://img.icons8.com/clouds/200/smart-phone.png", width=120)
st.sidebar.title("🛠️ Điều khiển & Cấu hình")
st.sidebar.markdown("---")

if not df_prod.empty:
    brands = sorted(df_prod["Brand"].dropna().unique())
    selected_brand = st.sidebar.selectbox("Lọc theo Thương hiệu", ["Tất cả"] + list(brands))
else:
    selected_brand = "Tất cả"

st.sidebar.markdown(
    """
    ### 📊 Phương pháp & Model:
    1. **Phân cụm KMeans & PCA**: Nhóm sản phẩm dựa trên 7 chỉ số phần cứng & thương mại.
    2. **Phân tích Aspect-Based Sentiment**: Tách các câu đánh giá của khách hàng thành 5 khía cạnh trọng tâm với trọng số tương tác (Likes).
    3. **Độ co giãn Giá bán (Price Elasticity)**: Hồi quy tuyến tính định lượng sức mua theo Discount Rate.
    4. **Dự báo Sales XGBoost**: Tinh chỉnh siêu tham số với Optuna để dự đoán lượng bán ra của sản phẩm ngày kế tiếp.
    """
)

# Render Global KPI metrics
if not df_prod.empty and not df_reviews.empty:
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tổng số sản phẩm</div><div class="metric-value">{len(df_prod)}</div><div class="metric-label">Đã phân loại</div></div>', 
            unsafe_allow_html=True
        )
    with col_kpi2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tổng số đánh giá</div><div class="metric-value">{len(df_reviews)}</div><div class="metric-label">Thế Giới Di Động</div></div>', 
            unsafe_allow_html=True
        )
    with col_kpi3:
        avg_rating = df_prod["Avg_Star_Rating"].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Điểm sao trung bình</div><div class="metric-value">{avg_rating:.2f} ⭐</div><div class="metric-label">Mức hài lòng chung</div></div>', 
            unsafe_allow_html=True
        )
    with col_kpi4:
        best_elastic_brand = "N/A"
        max_elasticity = -999.0
        for b, v in elasticity_dict.items():
            coef = v.get("Price_Elasticity", 0)
            if coef > max_elasticity and coef > 0:
                max_elasticity = coef
                best_elastic_brand = b
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Thương hiệu nhạy giá nhất</div><div class="metric-value">{best_elastic_brand}</div><div class="metric-label">Hệ số co giãn: {max_elasticity:.2f}</div></div>', 
            unsafe_allow_html=True
        )

# ----------------- TABS SETUP -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Phân Khúc Sản Phẩm (PCA)", 
    "💬 Ý Kiến Theo Khía Cạnh (Aspect)", 
    "📊 Độ Nhạy Giá & Khuyến Mãi",
    "🔮 Dự Báo Doanh Số (XGBoost)",
    "🤖 Thử Nghiệm Mô Hỏi Sentiment",
    "📚 Từ Điển Teen Code"
])

# ================= TAB 1: PCA PRODUCT CLUSTER =================
with tab1:
    st.header("🎯 Phân Tích Phân Khúc & Thuật Toán Phân Nhóm (K-Means & PCA)")
    
    st.markdown(
        """
        <div class="explanation-box">
        <h4>💡 Giải thích đơn giản về Cách Phân Nhóm Điện Thoại:</h4>
        <ul>
            <li><b>PCA là gì? Tại sao vẽ được bản đồ 2D này?</b> 
                Mỗi chiếc điện thoại có rất nhiều thông số phức tạp (dung lượng pin, RAM, ROM, camera, giá tiền, số lượng bán ra...). Rất khó để nhìn bằng mắt thường và so sánh tất cả các thông số này cùng lúc. 
                Công nghệ <b>PCA</b> giúp chúng ta gom toàn bộ các thông số đó lại và biến đổi thành 2 tọa độ đơn giản là <b>Trục ngang (PC1)</b> và <b>Trục đứng (PC2)</b> để vẽ lên bản đồ phẳng.
            </li>
            <li><b>Hai trục của bản đồ có ý nghĩa gì?</b>
                <ul>
                    <li><b>Trục Ngang (PC1):</b> Đại diện cho <b>"Độ xịn và Giá tiền"</b> của máy. Càng đi về phía bên phải, máy càng đắt tiền, RAM/ROM khủng và camera xịn hơn (Dòng cao cấp). Càng đi về bên trái là các dòng máy giá rẻ, cấu hình cơ bản.</li>
                    <li><b>Trục Đứng (PC2):</b> Đại diện cho <b>"Mức độ nổi tiếng và sức mua"</b>. Càng đi lên phía trên, máy càng có nhiều lượt đánh giá và số lượng bán ra lớn hơn từ Thế Giới Di Động.</li>
                </ul>
            </li>
            <li><b>Thuật toán K-Means phân cụm như thế nào?</b>
                Thuật toán tự động quét qua toàn bộ danh sách, so sánh thông số và tự động gom các máy "cùng hệ" (giống nhau về cấu hình, giá tiền, sức mua) vào chung một nhóm màu để chúng ta dễ dàng so sánh các đối thủ cạnh tranh trực tiếp.
            </li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if df_prod.empty:
        st.warning("Vui lòng chạy luồng xử lý hoặc kiểm tra tệp data-project/processed/products_clustered.csv")
    else:
        plot_df = df_prod.copy()
        if selected_brand != "Tất cả":
            plot_df = plot_df[plot_df["Brand"] == selected_brand]
            
        numeric_cols = plot_df.select_dtypes(include=["number"]).columns
        features = [col for col in ["RAM", "ROM", "Battery", "Camera_MP", "Discounted_Price", "Original_Price", "Sales_Volume", "Avg_Star_Rating", "Total_Reviews"] if col in numeric_cols]
        
        if len(plot_df) < 2 or len(features) < 2:
            st.error("Không đủ mẫu dữ liệu (cần tối thiểu 2 điện thoại) hoặc đặc trưng số để thực hiện giải thuật PCA.")
        else:
            X = plot_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            
            plot_df["PC1"] = coords[:, 0]
            plot_df["PC2"] = coords[:, 1]
            
            hover_cols = [col for col in ["Product_ID", "Brand", "Discounted_Price", "RAM", "ROM", "Battery", "Avg_Star_Rating"] if col in plot_df.columns]
            
            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="cluster_name" if "cluster_name" in plot_df.columns else "cluster_id",
                title=f"Bản Đồ Phân Phân Khúc Điện Thoại 2D (PCA) - Thương hiệu: {selected_brand}",
                hover_data=hover_cols,
                labels={"PC1": "Thành phần Chính 1 (PC1)", "PC2": "Thành phần Chính 2 (PC2)"},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
            fig.update_layout(legend_title_text="Phân cụm sản phẩm (K-Means)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Danh sách sản phẩm chi tiết theo Phân khúc")
            display_cols = [col for col in ["Product_ID", "Product_Name", "Brand", "Discounted_Price", "RAM", "ROM", "Battery", "cluster_name"] if col in plot_df.columns]
            st.dataframe(
                plot_df[display_cols].rename(
                    columns={
                        "Product_Name": "Tên sản phẩm",
                        "Discounted_Price": "Giá bán khuyến mãi (đ)",
                        "cluster_name": "Phân cụm / Segment"
                    }
                ),
                use_container_width=True
            )


# ================= TAB 2: ASPECT SENTIMENT ANALYSIS =================
with tab2:
    st.header("💬 Phân Tích Ý Kiến Người Dùng & Điểm Mạnh/Yếu Từng Sản Phẩm (Pros & Cons)")
    
    st.markdown(
        """
        <div class="explanation-box">
        <h4>💡 Giải thích chi tiết về mặt Khoa học Dữ liệu (KHDL):</h4>
        <ul>
            <li><b>Aspect-Based Sentiment Analysis là gì?</b> Thay vì chỉ đánh giá bình luận là tích cực hay tiêu cực chung chung, mô hình chia nhỏ văn bản và chấm điểm độ hài lòng theo từng khía cạnh cụ thể: <b>Pin & Sạc, Hiệu năng/Mượt mà, Camera chụp ảnh, Thiết kế ngoại hình, và Dịch vụ khách hàng/Thái độ nhân viên</b>.</li>
            <li><b>Phân tích Điểm mạnh / Điểm yếu (Pros & Cons) của từng sản phẩm:</b> Phần này giúp bạn chọn ra sản phẩm cụ thể và xem ngay lập tức các khía cạnh khách hàng hài lòng nhất (Ưu điểm) cũng như các điểm bị phàn nàn nhiều nhất (Nhược điểm), đi kèm trích dẫn nguyên văn đánh giá.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if not aspects_dict:
        st.warning("Không tìm thấy dữ liệu phân tích khía cạnh (data-project/processed/aspect_sentiment.json). Hãy chạy luồng xử lý trước.")
    else:
        product_keys = list(aspects_dict.keys())
        product_options = {aspects_dict[k]["Product_Name"]: k for k in product_keys}
        
        selected_prod_name = st.selectbox("Chọn Sản phẩm để phân tích chuyên sâu", list(product_options.keys()))
        selected_key = product_options[selected_prod_name]
        
        prod_data = aspects_dict[selected_key]
        
        # Grid layout for selected product overview
        st.subheader(f"📱 Báo cáo phản hồi: {prod_data['Product_Name']}")
        
        aspects_scores = []
        for aspect_name, details in prod_data["Aspects"].items():
            aspects_scores.append({
                "Khía cạnh": aspect_name,
                "Chỉ số hài lòng (Score)": details.get("Score", 0.5) * 100
            })
            
        df_scores = pd.DataFrame(aspects_scores)
        
        # Render a beautiful Bar Chart for aspect satisfaction scores
        fig_bar = px.bar(
            df_scores,
            x="Chỉ số hài lòng (Score)",
            y="Khía cạnh",
            orientation="h",
            color="Chỉ số hài lòng (Score)",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            title=f"Độ Hài Lòng Khách Hàng (%) theo Từng Khía Cạnh - {prod_data['Product_Name']}",
            labels={"Chỉ số hài lòng (Score)": "Chỉ số hài lòng (%)"},
            template="plotly_white"
        )
        fig_bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Summarize Strengths and Weaknesses of the product based on high/low scores
        st.markdown("### 🏆 Đánh giá Điểm mạnh & Điểm yếu tổng quát của sản phẩm")
        
        col_strength, col_weakness = st.columns(2)
        
        with col_strength:
            st.markdown("<div style='background-color:#d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745;'><h4 style='color:#155724; margin:0 0 10px 0;'>🌟 ĐIỂM MẠNH NỔI BẬT (Strengths)</h4>", unsafe_allow_html=True)
            strengths_found = False
            for aspect_name, details in prod_data["Aspects"].items():
                if details.get("Score", 0.5) >= 0.6:
                    st.markdown(f"**🔹 {aspect_name} ({details.get('Score')*100:.1f}%)**: Khách hàng rất thích khía cạnh này.")
                    strengths_found = True
            if not strengths_found:
                st.markdown("Chưa ghi nhận khía cạnh nào vượt trội nổi bật.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_weakness:
            st.markdown("<div style='background-color:#f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545;'><h4 style='color:#721c24; margin:0 0 10px 0;'>⚠️ ĐIỂM YẾU / HẠN CHẾ (Weaknesses)</h4>", unsafe_allow_html=True)
            weaknesses_found = False
            for aspect_name, details in prod_data["Aspects"].items():
                if details.get("Score", 0.5) < 0.45:
                    st.markdown(f"**🔸 {aspect_name} ({details.get('Score')*100:.1f}%)**: Nhận nhiều phản hồi tiêu cực/cần cải thiện.")
                    weaknesses_found = True
            if not weaknesses_found:
                st.markdown("Sản phẩm hoạt động rất ổn định, không có khuyết điểm nghiêm trọng.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Detailed Pros & Cons list with expansion panels
        st.subheader("🔍 Chi tiết Đóng góp của Khách hàng theo Khía cạnh")
        
        for aspect_name, details in prod_data["Aspects"].items():
            score_val = details.get("Score", 0.5)
            sentiment_indicator = "🟢 Tốt/Hài lòng" if score_val >= 0.6 else ("🟡 Tạm ổn/Trung lập" if score_val >= 0.4 else "🔴 Cần cải thiện/Tiêu cực")
            
            with st.expander(f"⚙️ Khía cạnh {aspect_name} ({sentiment_indicator} - {score_val*100:.1f}%)"):
                col_pros, col_cons = st.columns(2)
                
                with col_pros:
                    st.markdown("##### 👍 Phản hồi Tích cực (Pros)")
                    pros_list = details.get("Pros", [])
                    if not pros_list:
                        st.info("Chưa ghi nhận ưu điểm nổi bật cho khía cạnh này.")
                    else:
                        for p in pros_list[:5]:
                            st.markdown(f"- *\"{p}\"*")
                            
                with col_cons:
                    st.markdown("##### 👎 Phản hồi Tiêu cực & Góp ý (Cons)")
                    cons_list = details.get("Cons", [])
                    if not cons_list:
                        st.success("Không có phàn nàn hay nhược điểm đáng kể nào!")
                    else:
                        for c in cons_list[:5]:
                            st.markdown(f"- *\"{c}\"*")

# ================= TAB 3: PRICE ELASTICITY & DISCOUNT OPTIMIZATION =================
with tab3:
    st.header("📊 Độ Nhạy Giá & Lập Kế Hoạch Chiến Lược Khuyến Mãi (Price Elasticity)")
    
    st.markdown(
        """
        <div class="explanation-box">
        <h4>💡 Giải thích chi tiết về mặt Khoa học Dữ liệu (KHDL):</h4>
        <ul>
            <li><b>Độ co giãn của cầu theo giá (Price Elasticity of Demand) là gì?</b> Là thước đo lượng bán ra (Sales Volume) của một thương hiệu thay đổi như thế nào khi tỷ lệ giảm giá (Discount Rate) biến động. Hệ số co giãn dương biểu diễn xu hướng <b>giảm giá tăng lượng bán</b> rất nhạy bén.</li>
            <li><b>Ý nghĩa thực tiễn:</b> 
                <ul>
                    <li>Nếu một thương hiệu có hệ số co giãn cao (ví dụ Samsung), việc tăng khuyến mãi 5%-10% sẽ giúp <b>bùng nổ lượng bán ra rất lớn</b>.</li>
                </ul>
            </li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if not elasticity_dict:
        st.warning("Không tìm thấy dữ liệu hồi quy độ nhạy giá. Vui lòng chạy luồng Price Optimization.")
    else:
        elasticity_data = []
        for brand_name, info in elasticity_dict.items():
            elasticity_data.append({
                "Thương hiệu": brand_name,
                "Hệ số co giãn giá (Elasticity)": info.get("Price_Elasticity", 0),
                "Số lượng sản phẩm": info.get("Product_Count", 0),
                "Mức giảm giá TB (%)": info.get("Average_Discount_Rate", 0) * 100,
                "Doanh số TB (máy)": info.get("Average_Sales_Volume", 0),
                "Độ tin cậy R-squared": info.get("Model_R2", 0)
            })
            
        df_elasticity = pd.DataFrame(elasticity_data)
        
        fig_elas = px.bar(
            df_elasticity,
            x="Thương hiệu",
            y="Hệ số co giãn giá (Elasticity)",
            color="Hệ số co giãn giá (Elasticity)",
            color_continuous_scale="Tealgrn",
            title="So Sánh Hệ Số Co Giãn Giá Theo Từng Thương Hiệu",
            labels={"Hệ số co giãn giá (Elasticity)": "Hệ số co giãn (Elasticity)"},
            template="plotly_white"
        )
        st.plotly_chart(fig_elas, use_container_width=True)
        
        st.markdown("### 🎛️ Trình mô phỏng Kế hoạch Khuyến mãi thông minh")
        st.markdown("Chọn một thương hiệu và di chuyển thanh trượt để ước tính lượng hàng bán ra tăng thêm.")
        
        sim_brand = st.selectbox("Chọn Thương hiệu để mô phỏng", list(elasticity_dict.keys()))
        brand_info = elasticity_dict[sim_brand]
        
        col_slider, col_result = st.columns([1, 1])
        with col_slider:
            discount_increment = st.slider(
                "Tỷ lệ tăng thêm chiết khấu / giảm giá dự kiến (%)", 
                min_value=1, 
                max_value=30, 
                value=5, 
                step=1
            )
            
            elas = brand_info.get("Price_Elasticity", 0)
            avg_sales = brand_info.get("Average_Sales_Volume", 0)
            
            percentage_increase = elas * discount_increment
            sales_volume_increase = (percentage_increase / 100) * avg_sales
            
        with col_result:
            st.markdown(f"#### 🔮 Kết quả Ước tính cho **{sim_brand}**:")
            if percentage_increase > 0:
                st.success(
                    f"""
                    - **Hệ số co giãn của thương hiệu:** `{elas:.4f}`
                    - **Doanh số bán ra trung bình hiện tại:** `{avg_sales:,.0f} máy`
                    - **Dự kiến Lượng hàng bán ra Tăng thêm:** `+{sales_volume_increase:,.1f} máy`
                    - **Tỷ lệ tăng trưởng doanh số:** `+{percentage_increase:.2f}%`
                    """
                )
            elif percentage_increase < 0:
                st.warning(
                    f"""
                    - **Hệ số co giãn của thương hiệu:** `{elas:.4f}`
                    - **Doanh số bán ra trung bình hiện tại:** `{avg_sales:,.0f} máy`
                    - **Dự kiến Lượng hàng bán ra giảm:** `{sales_volume_increase:,.1f} máy`
                    - **Tỷ lệ tăng trưởng doanh số:** `{percentage_increase:.2f}%`
                    
                    *💡 Giải thích kinh tế học: Hệ số co giãn âm biểu thị việc tăng chiết khấu/khuyến mãi không kích thích lượng mua tăng thêm, ngược lại lượng bán ra có xu hướng giảm nhẹ. Điều này có thể xảy ra ở các thương hiệu phân khúc giá rẻ hoặc do tâm lý khách hàng nghi ngờ chất lượng khi sản phẩm giảm giá quá sâu.*
                    """
                )
            else:
                st.info(
                    f"""
                    - **Hệ số co giãn của thương hiệu:** `{elas:.4f}`
                    - **Doanh số bán ra trung bình hiện tại:** `{avg_sales:,.0f} máy`
                    - **Dự kiến Lượng hàng bán ra Tăng thêm:** `0.0 máy`
                    - **Tỷ lệ tăng trưởng doanh số:** `0.00%`
                    
                    *💡 Giải thích kinh tế học: Hệ số co giãn bằng 0 biểu thị sản phẩm này hoàn toàn không nhạy cảm về giá trong tập dữ liệu. Việc tăng hay giảm khuyến mãi không làm thay đổi doanh số đáng kể.*
                    """
                )

# ================= TAB 4: XGBOOST SALES FORECASTING =================
with tab4:
    st.header("🔮 Mô Hình Học Máy Dự Báo Doanh Số Smartphone Tương Lai (XGBoost)")
    
    st.markdown(
        """
        <div class="explanation-box">
        <h4>💡 Giải thích chi tiết về mặt Khoa học Dữ liệu (KHDL):</h4>
        <ul>
            <li><b>XGBoost (Extreme Gradient Boosting)</b> là thuật toán học máy mạnh mẽ dựa trên cây quyết định tuần tự. Model được huấn luyện để dự đoán doanh số bán hàng ngày tiếp theo dựa trên các yếu tố lịch sử đa biến.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    metrics_path = Path("models/forecasting/xgboost_metrics.csv")
    model_path = Path("models/forecasting/xgboost_model.pkl")
    
    if metrics_path.exists():
        df_met = pd.read_csv(metrics_path)
        st.subheader("📊 Kết quả Đánh giá Mô hình XGBoost khi Huấn luyện:")
        st.dataframe(df_met, use_container_width=True)
        
    if not model_path.exists():
        st.warning("Mô hình XGBoost (models/forecasting/xgboost_model.pkl) chưa được huấn luyện. Vui lòng chạy train_xgboost.py")
    else:
        st.subheader("🕹️ Thử nghiệm Dự báo Doanh số sản phẩm")
        st.markdown("Nhập các thông số hiện tại của sản phẩm để mô hình XGBoost tính toán doanh số ngày tiếp theo:")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            disc_price = st.number_input("Giá bán khuyến mãi hiện tại (VNĐ)", min_value=500000, max_value=50000000, value=12490000, step=100000)
            disc_rate = st.slider("Tỷ lệ giảm giá hiện tại (%)", min_value=0, max_value=70, value=8)
            star_rating = st.slider("Điểm sao đánh giá trung bình", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
            sent_score = st.slider("Chỉ số tích cực từ bình luận khách hàng (Sentiment Score)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
            
        with col_f2:
            sales_7d = st.number_input("Trung bình doanh số 7 ngày trước đó (máy)", min_value=0, max_value=1000000, value=45000)
            day_of_week = st.selectbox("Ngày trong tuần", ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"])
            is_holiday = st.checkbox("Có phải ngày Lễ tết không?", value=False)
            
            dow_map = {"Thứ 2": 0, "Thứ 3": 1, "Thứ 4": 2, "Thứ 5": 3, "Thứ 6": 4, "Thứ 7": 5, "Chủ Nhật": 6}
            val_dow = dow_map[day_of_week]
            val_weekend = 1 if val_dow >= 5 else 0
            val_holiday = 1 if is_holiday else 0
            
        if st.button("🔮 Thực hiện Dự Báo Doanh Số", type="primary"):
            model_data = joblib.load(model_path)
            model = model_data if not isinstance(model_data, dict) else model_data.get("model")
            trained_features = model_data.get("features", []) if isinstance(model_data, dict) else []
            
            input_map = {
                "Discounted_Price": disc_price,
                "Discount_Rate": disc_rate / 100.0,
                "Avg_Star_Rating": star_rating,
                "Sentiment_Score": sent_score,
                "Day_of_Week": val_dow,
                "Is_Weekend": val_weekend,
                "Is_Holiday": val_holiday,
                "Sales_Volume_7d_mean": sales_7d
            }
            
            if len(trained_features) > 0:
                vec = [input_map.get(feat, 0.0) for feat in trained_features]
            else:
                # Fallback to standard feature list order
                vec = [
                    disc_price,
                    disc_rate / 100.0,
                    star_rating,
                    sent_score,
                    val_dow,
                    val_weekend,
                    val_holiday,
                    sales_7d
                ]
            
            features_input = np.array([vec])
            
            predicted_sales = float(model.predict(features_input)[0])
            if predicted_sales < 0:
                predicted_sales = 0.0
                
            st.success(f"### 🎯 Kết quả Dự đoán Doanh số Ngày tiếp theo: **{predicted_sales:,.1f} máy**")

# ================= TAB 5: ML SENTIMENT ROOM =================
with tab5:
    st.header("🤖 Trải Nghiệm Mô Hình Nhận Diện Ý Kiến Đánh Giá (TF-IDF + Random Forest)")
    
    st.markdown(
        """
        <div class="explanation-box">
        <h4>💡 Giải thích chi tiết về mặt Khoa học Dữ liệu (KHDL):</h4>
        <ul>
            <li><b>Pipeline xử lý ngôn ngữ tự nhiên (NLP):</b> Khi người dùng nhập một bình luận, hệ thống sẽ thực hiện các bước dịch từ viết tắt và teen code để hiểu đúng nghĩa của khách hàng.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    vectorizer_path = Path("models/sentiment/tfidf_vectorizer.pkl")
    model_path = Path("models/sentiment/rf_classifier.pkl")
    
    user_input_review = st.text_area("Nhập đánh giá của khách hàng cần kiểm thử (ví dụ: máy dùng siu ngon, pin trâu k giật lag gì cả, sạc pin hơi nóng máy...)", height=150)
    
    if st.button("🔥 Phân Tích Ý Kiến", type="primary"):
        if not vectorizer_path.exists() or not model_path.exists():
            st.error("Chưa tìm thấy tệp trọng số mô hình Sentiment (models/sentiment/rf_classifier.pkl). Vui lòng huấn luyện model.")
        elif not user_input_review.strip():
            st.warning("Vui lòng nhập nội dung đánh giá để kiểm thử.")
        else:
            cleaned_text = clean_teencode(user_input_review, teencode_dict)
            
            st.markdown("##### 🔍 Quá trình tiền xử lý ngôn ngữ tự nhiên (NLP):")
            col_pre1, col_pre2 = st.columns(2)
            with col_pre1:
                st.info(f"**Văn bản gốc:**\n*\"{user_input_review}\"*")
            with col_pre2:
                st.success(f"**Đã xử lý & dịch Teen code:**\n*\"{cleaned_text}\"*")
                
            vectorizer = joblib.load(vectorizer_path)
            classifier = joblib.load(model_path)
            
            X_text = vectorizer.transform([cleaned_text])
            prob_score = float(classifier.predict_proba(X_text)[0][list(classifier.classes_).index(1)])
            
            st.markdown("##### 📊 Kết quả từ Mô hình Random Forest:")
            if prob_score >= 0.5:
                st.success(f"### 🎉 Nhãn: TÍCH CỰC (Positive) - Độ tin cậy: {prob_score:.2%}")
                st.progress(prob_score)
            else:
                st.error(f"### ⚠️ Nhãn: TIÊU CỰC / GÓP Ý (Negative) - Độ tin cậy: {(1 - prob_score):.2%}")
                st.progress(prob_score)

# ================= TAB 6: TEENCODE DICTIONARY =================
with tab6:
    st.header("📚 Bộ Từ Điển Tiền Xử Lý Teen Code & Viết Tắt Tiếng Việt")
    st.markdown("Bảng ánh xạ từ teen code, từ lóng hoặc từ viết tắt phổ biến trong các bình luận sang từ ngữ chuẩn mực tiếng Việt để tăng cường chất lượng phân tích của mô hình NLP:")
    
    if not teencode_dict:
        st.warning("Không tìm thấy dữ liệu từ điển Teen code tại data-project/teencode_dict.json")
    else:
        df_teen = pd.DataFrame(list(teencode_dict.items()), columns=["Từ gốc / Teen code", "Từ thay thế chuẩn mực"])
        st.dataframe(df_teen, use_container_width=True, height=500)
