# ================= IMPORT =================
import streamlit as st
import pandas as pd
import joblib
import sklearn
import sys


# ================= LOAD MODEL =================
model = joblib.load("model_simple_v2.pkl")
cols = joblib.load("columns_v2.pkl")

# ================= UI =================
st.image("group7Banner.png", width='stretch')

st.markdown(
    "<h2 style='text-align: center; color: orange;'>"
    "Dự đoán giá nhà – Gợi ý nhà phù hợp"
    "</h2>",
    unsafe_allow_html=True
)

menu = [
    "🏠 Home",
    "💰 Dự đoán giá nhà",
    "🏡 Gợi ý nhà",
    "📊 Kiểm tra nhà bất thường",
    "👥 Nhóm"
]

choice = st.sidebar.selectbox("Menu", menu)

# ================= HOME =================
if choice == "🏠 Home":

    st.markdown("""
    ## 🎯 Ứng dụng Data Science trong bất động sản

    Ứng dụng hỗ trợ:
    - 💰 Dự đoán giá nhà
    - 🏡 Gợi ý nhà tương tự
    - 📊 Kiểm tra nhà bất thường

    👉 Giúp người dùng ra quyết định chính xác hơn
    """)

# ================= PROJECT 1 =================
elif choice == "💰 Dự đoán giá nhà":

    st.subheader("💰 Dự đoán giá nhà")

    # ===== INPUT =====
    dien_tich = st.number_input("📐 Diện tích (m²)", min_value=1.0)
    so_tang = st.slider("🏢 Số tầng", 0, 10, 1)

    khu_vuc = st.selectbox("📍 Khu vực", ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"])

    # ===== BUTTON =====
    if st.button("🚀 Dự đoán", key="predict_price"):

        # Encode quận
        e_govap = 1 if khu_vuc == "Gò Vấp" else 0
        e_phunhuan = 1 if khu_vuc == "Phú Nhuận" else 0

        # Input đúng model
        input_data = pd.DataFrame([{
            "dien_tich": dien_tich,
            "tong_so_tang": so_tang,
            "e_Quận Gò Vấp": e_govap,
            "e_Quận Phú Nhuận": e_phunhuan
        }])

        input_data = input_data[cols]

        st.write("📊 Input:", input_data)

        # ===== PREDICT =====
        gia = model.predict(input_data)[0]

        # ===== HIỂN THỊ =====
        st.success(f"💰 {gia:.2f} tỷ VND (~ {gia*1e9:,.0f} VND)")


# ================= PROJECT 2 =================
elif choice == "🏡 Gợi ý nhà":

    st.subheader("🏡 Gợi ý nhà tương tự")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # ===== LOAD MODEL =====
    df_rec = joblib.load("df_recommend.pkl")
    scaler = joblib.load("scaler_recommend.pkl")
    features = joblib.load("features_recommend.pkl")

    # ===== INPUT =====
    dien_tich = st.number_input("📐 Diện tích (m²)", min_value=1.0)
    gia_input = st.number_input("💰 Giá mong muốn (tỷ VND)", min_value=0.1)

    khu_vuc = st.selectbox(
        "📍 Khu vực",
        ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"]
    )

    # ===== BUTTON =====
    if st.button("🔍 Gợi ý"):

        # ===== ENCODE QUẬN =====
        e_govap = 1 if khu_vuc == "Gò Vấp" else 0
        e_phunhuan = 1 if khu_vuc == "Phú Nhuận" else 0

        # ===== INPUT DATA (QUAN TRỌNG) =====
        input_df = pd.DataFrame([{
            "dien_tich_num": dien_tich,
            "gia_ban_num": gia_input,
            "e_Quận Gò Vấp": e_govap,
            "e_Quận Phú Nhuận": e_phunhuan
        }])

        input_df = input_df[features]

        # ===== SCALE =====
        input_scaled = scaler.transform(input_df)
        X_scaled = scaler.transform(df_rec[features])

        # ===== SIMILARITY =====
        sims = cosine_similarity(input_scaled, X_scaled)[0]

        # ===== TOP 5 =====
        top_idx = np.argsort(sims)[-5:][::-1]

        result = df_rec.iloc[top_idx][[
            "tieu_de",
            "dien_tich_num",
            "quan",
            "gia_ban_num"
        ]].copy()

        result["similarity"] = sims[top_idx]

        # ===== HIỂN THỊ =====
        st.dataframe(result, use_container_width=True)


# ================= Kiểm tra bất thường =================
elif choice == "📊 Kiểm tra nhà bất thường":

    st.subheader("📊 Kiểm tra nhà bất thường")

    # ===== INPUT =====
    dien_tich = st.number_input("📐 Diện tích (m²)", min_value=1.0, key="area_ab")
    so_tang = st.slider("🏢 Số tầng", 0, 10, 1, key="floor_ab")

    khu_vuc = st.selectbox(
        "📍 Khu vực",
        ["Bình Thạnh", "Gò Vấp", "Phú Nhuận"],
        key="area_select_ab"
    )

    gia_nhap = st.number_input(
        "💰 Giá đăng bán (tỷ VND)",
        min_value=0.0,
        step=0.1
    )

    # ===== BUTTON =====
    if st.button("🚨 Kiểm tra", key="check_anomaly"):

        # Encode quận
        e_govap = 1 if khu_vuc == "Gò Vấp" else 0
        e_phunhuan = 1 if khu_vuc == "Phú Nhuận" else 0

        # Input model
        input_data = pd.DataFrame([{
            "dien_tich": dien_tich,
            "tong_so_tang": so_tang,
            "e_Quận Gò Vấp": e_govap,
            "e_Quận Phú Nhuận": e_phunhuan
        }])

        input_data = input_data[cols]

        # ===== PREDICT =====
        gia_du_doan = model.predict(input_data)[0]

        # ===== TÍNH ĐỘ LỆCH =====
        sai_lech = abs(gia_nhap - gia_du_doan) / gia_du_doan

        # ===== HIỂN THỊ =====
        st.write(f"💰 Giá dự đoán: {gia_du_doan:.2f} tỷ")
        st.write(f"📊 Giá nhập: {gia_nhap:.2f} tỷ")
        st.write(f"📉 Sai lệch: {sai_lech*100:.1f}%")

        # ===== KẾT LUẬN =====
        if sai_lech > 0.5:
            st.error("⚠️ Nhà có dấu hiệu bất thường!")
        else:
            st.success("✅ Nhà bình thường")

# ================= TEAM =================
elif choice == "👥 Nhóm":   
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")

    st.write("""### Có 2 chủ đề trong khóa học:    
    - Project 1: Dự đoán giá nhà, phát hiện tin đăng bán nhà bất thường
    - Project 2: Hệ thống gợi ý nhà dựa trên nội dung, phân cụm nhà
    """)

    st.markdown("## 👥 Thông tin & Phân công nhóm")

    import pandas as pd

    data = {
        "Thành viên": [
            "Huỳnh Lê Xuân Ánh",
            "Nguyễn Thị Tuyết Vân",
            "Đặng Đức Duy"
        ],
        "Project 1": [
            "Xử lý dữ liệu, xây dựng models ML thường (Python)",
            "Xây dựng models trong môi trường PySpark",
            "Phát hiện giá bất thường, báo cáo"
        ],
        "Project 2": [
            "Recommendation System (Scikit-learn)",
            "Phân cụm dữ liệu BĐS (Scikit-learn & PySpark)",
            "Data & Preprocessing"
        ],
        "GUI": [
            "Tích hợp mô hình dự đoán & phát hiện bất thường, triển khai hệ thống",
            "Tích hợp hệ thống gợi ý & phân cụm vào ứng dụng",
            "Thiết kế giao diện & trải nghiệm người dùng"
        ],
        "Email": [
            "huynhlexuananh2002@gmail.com",
            "tuyetvan1418393@gmail.com",
            "duydd1411@gmail.com"
        ]
    }

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)