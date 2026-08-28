# ==========================================================
# Breast Cancer Histopathology Classification Dashboard
# MSc Final Year Project
# Author: Waqar Ul Hassan
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from app.styles import load_css
from app.prediction import predict
from app.charts import (
    probability_chart,
    comparison_chart,
    performance_table,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Breast Cancer Classification System",
    
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Load Custom CSS
# ==========================================================

load_css()

# ==========================================================
# Session State
# ==========================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("🩺 AI Diagnosis")

    st.markdown("---")

    st.subheader("Project Information")

    st.write("**Dataset**")
    st.write("BreaKHis Histopathology")

    st.write("**Models**")
    st.write("• VGG16")
    st.write("• ResNet50")
    st.write("• DenseNet201")
    st.write("• MobileNetV2")
    st.write("• Linear SVM")

    st.markdown("---")

    st.subheader("Performance")

    st.metric(
    "🏆 Best Model",
    "DenseNet201",
    "93.09% Accuracy",
)

    st.markdown("---")

    st.success(
    """
    ✔ Upload Histopathology Image

    ✔ Select AI Model

    ✔ Run Prediction

    ✔ Compare All Models

    ✔ View Results
    """
)

# ==========================================================
# Main Header
# ==========================================================

st.markdown(
    """
    <div class="main-title">
    Breast Cancer Histopathology Classification System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Comparative Analysis using
    VGG16, ResNet50,
    DenseNet201,
    MobileNetV2
    and Linear SVM
    </div>
    """,
    unsafe_allow_html=True,
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "BreaKHis")

with col2:
    st.metric("Images", "7,909")

with col3:
    st.metric("Classes", "2")

with col4:
    st.metric("Models", "5")
st.markdown("---")

# ==========================================================
# TOP DASHBOARD
# ==========================================================

st.markdown(
    """
    <div class="dashboard-section">
    <div class="section-title">
    1️⃣ Upload & Prediction Setup
    </div>
    """,
    unsafe_allow_html=True,
)

left_column, middle_column, right_column = st.columns(
    [1.6, 1.2, 2.2],
    gap="medium"
)

# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------

with left_column:

    st.markdown("#### 📤 Upload Image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Upload a breast histopathology image",
    )

    if uploaded_file is None:

        st.info("Upload an image to begin diagnosis.")

# ---------------------------------------------------------
# Select Model
# ---------------------------------------------------------

with middle_column:

    st.markdown("#### 🧠 Select Model")

    selected_model = st.selectbox(
        "",
        [
            "DenseNet201",
            "VGG16",
            "MobileNetV2",
            "ResNet50",
            "Linear SVM",
        ],
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "🔍 Run Prediction",
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("Selected Model")

    st.success(selected_model)

# ---------------------------------------------------------
# Preview
# ---------------------------------------------------------

with right_column:

    st.markdown("#### 🖼 Image Preview")

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            use_container_width=True,
        )

        st.caption(uploaded_file.name)

    else:

        st.info("Image preview will appear here.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# PREDICTION SECTION
# ==========================================================

prediction_column, details_column = st.columns(
    [1.6, 1],
    gap="medium"
)

# ---------------------------------------------------------
# Prediction Result
# ---------------------------------------------------------

with prediction_column:

    st.markdown(
        """
        <div class="dashboard-section">
        <div class="section-title">
        🩺 Prediction Result
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is None:

        st.info("Upload an image to begin diagnosis.")

    else:

        if predict_button:

            with st.spinner("Running AI prediction..."):

                result = predict(
                    uploaded_file,
                    selected_model,
                )

                st.session_state.prediction_result = result

        if st.session_state.prediction_result is not None:

            result = st.session_state.prediction_result

            prediction = result["label"]
            confidence = result["confidence"]

            banner_color = "#16A34A" if prediction == "Benign" else "#DC2626"
            banner_bg = "#ECFDF5" if prediction == "Benign" else "#FEF2F2"

            st.markdown(
                f"""
                <div style="
                    background:{banner_bg};
                    border-left:8px solid {banner_color};
                    border-radius:14px;
                    padding:20px;
                    margin-bottom:18px;
                ">

                <h2 style="
                    color:{banner_color};
                    margin:0;
                    font-size:32px;
                ">
                {'🟢 BENIGN' if prediction=='Benign' else '🔴 MALIGNANT'}
                </h2>

                <p style="
                    color:#555;
                    margin-top:10px;
                    font-size:16px;
                ">
                AI Classification Completed Successfully
                </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2 = st.columns(2)

            with metric1:

                st.metric(
                    "Prediction",
                    prediction,
                )

            with metric2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                )

            st.progress(confidence / 100)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction Details
# ---------------------------------------------------------

with details_column:

    st.markdown(
        """
        <div class="dashboard-section">
        <div class="section-title">
        📋 Prediction Details
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.prediction_result is None:

        st.info("Prediction details will appear here.")

    else:

        result = st.session_state.prediction_result

        st.metric(
            "Selected Model",
            selected_model,
        )

        st.metric(
            "Prediction",
            result["label"],
        )

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )

        st.metric(
            "Processing Time",
            f"{result['processing_time']:.2f} sec"
        )

        st.metric(
            "Input Size",
            "224 × 224"
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# ==========================================================
# COMPARISON SECTION
# ==========================================================

left_compare, right_compare = st.columns(
    [1.7, 1],
    gap="medium"
)

# ---------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------

with left_compare:

    st.markdown(
        """
        <div class="dashboard-section">
        <div class="section-title">
        📊 Model Performance Comparison
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:

        models = [
            "DenseNet201",
            "VGG16",
            "ResNet50",
            "MobileNetV2",
            "Linear SVM",
        ]

        comparison_results = []

        for model_name in models:

            uploaded_file.seek(0)

            result = predict(
                uploaded_file,
                model_name,
            )

            comparison_results.append(
                {
                    "Model": model_name,
                    "Prediction": result["label"],
                    "Confidence (%)": round(
                        result["confidence"], 2
                    ),
                }
            )

        comparison_df = pd.DataFrame(comparison_results)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        best_model = comparison_df.loc[
            comparison_df["Confidence (%)"].idxmax()
        ]

        st.success(
            f"🏆 Best Performing Model: "
            f"{best_model['Model']} "
            f"({best_model['Confidence (%)']:.2f}%)"
        )

    else:

        st.info(
            "Upload an image to compare all models."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Probability Distribution
# ---------------------------------------------------------

with right_compare:

    st.markdown(
        """
        <div class="dashboard-section">
        <div class="section-title">
        🍩 Probability Distribution
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.prediction_result is None:

        st.info(
            "Prediction required."
        )

    else:

        probability = st.session_state.prediction_result[
            "probability"
        ]

        benign = (1 - probability) * 100
        malignant = probability * 100

        probability_df = pd.DataFrame({

            "Class":[
                "Benign",
                "Malignant",
            ],

            "Probability":[
                benign,
                malignant,
            ],

        })

        fig = px.pie(

            probability_df,

            values="Probability",

            names="Class",

            hole=.70,

            color="Class",

            color_discrete_map={

                "Benign":"#22C55E",

                "Malignant":"#EF4444",

            },

        )

        fig.update_traces(

            textinfo="percent",

            textfont_size=16,

        )

        fig.update_layout(

            height=360,

            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),

            showlegend=True,

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# ==========================================================
# ACCURACY CHART
# ==========================================================

st.markdown(
    """
    <div class="dashboard-section">
    <div class="section-title">
    📈 Overall Model Accuracy Comparison
    </div>
    """,
    unsafe_allow_html=True,
)

accuracy_df = pd.DataFrame({

    "Model":[
        "DenseNet201",
        "VGG16",
        "MobileNetV2",
        "ResNet50",
        "Linear SVM",
    ],

    "Accuracy":[
        93.09,
        87.11,
        86.52,
        83.49,
        72.70,
    ],

})

fig = px.bar(

    accuracy_df,

    x="Model",

    y="Accuracy",

    text="Accuracy",

    color_discrete_sequence=["#0B5394"],

)

fig.update_layout(

    height=320,

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),

    coloraxis_showscale=False,

    plot_bgcolor="white",

)

fig.update_traces(

    texttemplate="%{text:.2f}%",

    textposition="outside",

)

st.plotly_chart(

    fig,

    use_container_width=True,

)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    """
    <div style="
        background:white;
        border-radius:18px;
        padding:20px;
        text-align:center;
        border:1px solid #E7EDF5;
        box-shadow:0 2px 10px rgba(0,0,0,.06);
    ">

    <h3 style="color:#0B5394;margin-bottom:5px;">
    🩺 Breast Cancer Histopathology Classification System
    </h3>

    <p style="margin:0;color:#666;">
    MSc Computer Science and Technology
    </p>

    <p style="margin:0;color:#666;">
    Ulster University
    </p>

    <p style="margin-top:10px;color:#0B5394;font-weight:bold;">
    Developed by Waqar Ul Hassan
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)

