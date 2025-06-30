import streamlit as st
st.set_page_config(page_title="Fraud Detection", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

xgb_model=joblib.load('xgb_grid.pkl')
knn_model=joblib.load('knn_grid.pkl')
scaler=joblib.load('scaler.pkl')



st.title("💳 Credit Card Fraud Detection")
st.markdown("Compare predictions from **XGBoost** and **KNN** on transaction data.")

st.sidebar.header("📝 Input Transaction Features")

# Input features from user
amount = st.sidebar.slider("Amount", 0.0, 5000.0, 100.0)
time = st.sidebar.slider("Time", 0.0, 200000.0, 100000.0)
v_inputs = [st.sidebar.slider(f"V{i}", -5.0, 5.0, 0.0) for i in range(1, 29)]

# Prepare input for prediction
input_df = pd.DataFrame([v_inputs + [amount, time]], columns=[f"V{i}" for i in range(1, 29)] + ['Amount', 'Time'])
input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])
# Get correct feature order from training
correct_order = xgb_model.feature_names_in_

# Reorder features to match model training
correct_order = xgb_model.feature_names_in_
input_df = input_df[correct_order]

if st.button("🔍 Predict Transaction"):
    col1, col2 = st.columns(2)

    for model, name, col in zip([xgb_model, knn_model], ["XGBoost", "KNN"], [col1, col2]):
        # No error: input_df is now in correct order
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        with col:
            st.subheader(f"{name} Prediction")
            if pred == 1:
                st.error(f"🚨 Fraudulent Transaction\nProbability: {prob:.2f}" if prob else "🚨 Fraud!")
            else:
                st.success(f"✅ Legitimate Transaction\nProbability: {1 - prob:.2f}" if prob else "✅ Legit")


# Load sample data for visualization
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df.sample(5000, random_state=42)

df = load_data()

# Visualization Section
st.markdown("---")
st.header("📊 Data Visualizations")

st.subheader("Class Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax1)
ax1.set_title("Fraud (1) vs Legit (0) Transactions")
st.pyplot(fig1)

st.subheader("Amount Distribution by Class")
fig2, ax2 = plt.subplots()
sns.boxplot(x='Class', y='Amount', data=df, ax=ax2)
ax2.set_yscale("log")
st.pyplot(fig2)

st.subheader("Correlation with Fraud")
fig3, ax3 = plt.subplots(figsize=(8, 12))
sns.heatmap(df.corr()[['Class']].sort_values(by='Class', ascending=False), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

st.markdown("---")
st.subheader("📁 Sample Data")
st.dataframe(df.head(10))