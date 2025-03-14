import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load trained model
model = joblib.load("customer_segmentation_model.pkl")

st.title("🛍️ Customer Segmentation App")
st.write("Predict which segment a customer belongs to based on their income & spending.")

# User input
income = st.slider("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending = st.slider("Spending Score (1-100)", min_value=0, max_value=100, value=50)

if st.button("Predict Segment"):
    cluster = model.predict([[income, spending]])[0]
    st.success(f"✅ Customer belongs to Segment {cluster}!")

# Load customer dataset (Ensure this file exists)
df = pd.read_csv("customer_data.csv")

# Plot customer clusters
st.subheader("📊 Customer Segmentation Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="viridis", ax=ax)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
st.pyplot(fig)
