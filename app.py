import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Cancer Analytics Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("🩺 Cancer Analytics Dashboard")
st.markdown("Analyze Cancer Dataset with Interactive Visualizations and Insights")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Cancer Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Sidebar
    st.sidebar.header("Filters")

    # Dataset Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

    # Diagnosis Distribution
    if "diagnosis" in df.columns:

        st.subheader("🧬 Cancer Diagnosis Distribution")

        diagnosis_count = df["diagnosis"].value_counts()

        fig = px.pie(
            values=diagnosis_count.values,
            names=diagnosis_count.index,
            title="Cancer Diagnosis Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Numeric Columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:

        st.subheader("📉 Feature Analysis")

        selected_feature = st.selectbox(
            "Select Feature",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_feature,
            title=f"Distribution of {selected_feature}"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(
            df,
            y=selected_feature,
            title=f"Box Plot of {selected_feature}"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    if len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot
    if len(numeric_cols) >= 2:

        st.subheader("📌 Feature Relationship")

        x_axis = st.selectbox(
            "Select X-axis",
            numeric_cols,
            key="x"
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_cols,
            key="y"
        )

        color_col = None

        if "diagnosis" in df.columns:
            color_col = "diagnosis"

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_col,
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.subheader("💡 Insights")

    st.success(
        f"""
        ✔ Dataset contains {df.shape[0]} records and {df.shape[1]} features.

        ✔ Missing Values: {df.isnull().sum().sum()}

        ✔ Numeric Features: {len(numeric_cols)}

        ✔ Dashboard successfully analyzed the uploaded cancer dataset.
        """
    )

else:
    st.info("Upload a CSV file to begin analysis.")
