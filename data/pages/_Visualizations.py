import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Config
st.set_page_config(page_title="Visualizations", page_icon="📊")

st.title("📊 Cancer Data Visualizations")

# Load Dataset
df = pd.read_csv("data/cancer_dataset.csv")

# Numeric Columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Diagnosis Distribution
if "diagnosis" in df.columns:
    st.subheader("🧬 Diagnosis Distribution")

    fig = px.pie(
        df,
        names="diagnosis",
        title="Cancer Diagnosis Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# Histogram
st.subheader("📈 Feature Distribution")

selected_feature = st.selectbox(
    "Select Feature",
    numeric_cols
)

fig = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    title=f"Distribution of {selected_feature}"
)

st.plotly_chart(fig, use_container_width=True)

# Box Plot
st.subheader("📦 Box Plot")

fig = px.box(
    df,
    y=selected_feature,
    title=f"Box Plot of {selected_feature}"
)

st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")

corr = df[numeric_cols].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Matrix"
)

st.plotly_chart(fig, use_container_width=True)

# Scatter Plot
st.subheader("📌 Feature Relationship")

x_feature = st.selectbox(
    "Select X Feature",
    numeric_cols,
    key="x"
)

y_feature = st.selectbox(
    "Select Y Feature",
    numeric_cols,
    key="y"
)

if "diagnosis" in df.columns:
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="diagnosis",
        title=f"{x_feature} vs {y_feature}"
    )
else:
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        title=f"{x_feature} vs {y_feature}"
    )

st.plotly_chart(fig, use_container_width=True)

# Pairwise Analysis
st.subheader("📊 Feature Comparison")

feature1 = st.selectbox(
    "Feature 1",
    numeric_cols,
    key="f1"
)

feature2 = st.selectbox(
    "Feature 2",
    numeric_cols,
    key="f2"
)

fig = px.scatter(
    df,
    x=feature1,
    y=feature2,
    color="diagnosis" if "diagnosis" in df.columns else None,
    title=f"{feature1} vs {feature2}"
)

st.plotly_chart(fig, use_container_width=True)
