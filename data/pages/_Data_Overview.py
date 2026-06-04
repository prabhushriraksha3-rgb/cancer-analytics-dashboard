import streamlit as st
import pandas as pd

# Page Config
st.set_page_config(page_title="Data Overview", page_icon="📋")

st.title("📋 Data Overview")

# Load Dataset
df = pd.read_csv("data/cancer_dataset.csv")

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset Shape
st.subheader("Dataset Shape")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

# Column Information
st.subheader("Column Information")

info_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(info_df)

# Missing Values
st.subheader("Missing Values")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum()
})

st.dataframe(missing_df)

# Statistical Summary
st.subheader("Statistical Summary")
st.dataframe(df.describe())

# Dataset Insights
st.subheader("Quick Insights")

st.success(f"""
✅ Total Records: {df.shape[0]}

✅ Total Features: {df.shape[1]}

✅ Missing Values: {df.isnull().sum().sum()}

✅ Dataset Loaded Successfully
""")
