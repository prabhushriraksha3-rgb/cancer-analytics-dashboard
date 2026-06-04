import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Insights", page_icon="💡")

st.title("💡 Cancer Dataset Insights")

# Load Dataset
df = pd.read_csv("data/cancer_dataset.csv")

# Basic Information
st.subheader("📊 Dataset Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", df.shape[0])

with col2:
    st.metric("Total Features", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

# Diagnosis Insights
if "diagnosis" in df.columns:

    st.subheader("🧬 Diagnosis Analysis")

    diagnosis_counts = df["diagnosis"].value_counts()

    for diagnosis, count in diagnosis_counts.items():
        percentage = (count / len(df)) * 100
        st.write(
            f"**{diagnosis}** : {count} cases ({percentage:.2f}%)"
        )

# Numeric Feature Insights
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.subheader("📈 Feature Insights")

for col in numeric_cols[:10]:  # Top 10 features
    st.write(f"### {col}")

    st.write(f"Mean: {df[col].mean():.2f}")
    st.write(f"Median: {df[col].median():.2f}")
    st.write(f"Minimum: {df[col].min():.2f}")
    st.write(f"Maximum: {df[col].max():.2f}")

# Correlation Insights
if len(numeric_cols) > 1:

    st.subheader("🔥 Correlation Insights")

    corr_matrix = df[numeric_cols].corr()

    corr_pairs = (
        corr_matrix.unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
    )

    high_corr = corr_pairs[corr_pairs < 1].head(10)

    st.write("Top Correlated Features:")

    for idx, value in high_corr.items():
        st.write(
            f"{idx[0]} ↔ {idx[1]} : {value:.2f}"
        )

# Automated Insights
st.subheader("🤖 Automated Insights")

insights = []

if df.isnull().sum().sum() == 0:
    insights.append("Dataset contains no missing values.")

if len(numeric_cols) > 0:
    insights.append(
        f"Dataset contains {len(numeric_cols)} numerical features."
    )

if "diagnosis" in df.columns:
    insights.append(
        "Diagnosis column is available for cancer classification analysis."
    )

for insight in insights:
    st.success(insight)

# Conclusion
st.subheader("📌 Conclusion")

st.info(
    """
    This dataset can be used for:
    
    • Cancer Classification
    
    • Machine Learning Prediction
    
    • Statistical Analysis
    
    • Data Visualization
    
    • Healthcare Analytics
    """
)
