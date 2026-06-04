import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Cancer Prediction", page_icon="🧠")

st.title("🧠 Cancer Prediction")

# Load Dataset
df = pd.read_csv("data/cancer_dataset.csv")

# Check if diagnosis column exists
if "diagnosis" not in df.columns:
    st.error("Dataset must contain a 'diagnosis' column.")
    st.stop()

# Encode diagnosis column
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Keep only numeric columns
X = X.select_dtypes(include=np.number)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{accuracy:.2%}")

# Prediction Form
st.subheader("🔍 Predict Cancer Diagnosis")

user_input = {}

for col in X.columns:
    user_input[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    if prediction == 1:
        st.error(
            f"Prediction: Malignant\n\nConfidence: {probability:.2%}"
        )
    else:
        st.success(
            f"Prediction: Benign\n\nConfidence: {probability:.2%}"
        )

# Feature Importance
st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance_df.head(10))
