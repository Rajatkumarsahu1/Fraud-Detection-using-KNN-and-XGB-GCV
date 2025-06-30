import streamlit as st
st.set_page_config(page_title="Fraud Detection", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import tempfile

# Function to download a model file from a URL
def load_model_from_url(url):
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        response = requests.get(url)
        response.raise_for_status()
        tmp_file.write(response.content)
        tmp_file.flush()
        model = joblib.load(tmp_file.name)
    return model

# Load models and scaler
xgb_model = joblib.load('xgb_grid.pkl')
scaler = joblib.load('scaler.pkl')

# Load KNN model from GitHub releases URL
knn_url = 'https://github.com/Rajatkumarsahu1/Fraud-Detection-using-KNN-and-XGB-GCV/releases/download/v1.0/knn_grid.pkl'
knn_model = load_model_from_url(knn_url)

st.title("💳 Credit Card Fraud Detection")
st.markdown("Compare predictions from **XGBoost** and **KNN** on transaction data.")

st.sidebar.header("📝 Input Transaction Features")

# Input features
amount = st.sidebar.slider("Amount", 0.0, 5000.0, 100.0)
time = st.sidebar.slider("Time", 0.0, 200000.0, 100000.0)
v_inputs = [st.sidebar.slider(f"V{i}", -5.0, 5.0, 0.0) for i in range(1, 29)]

# Prepare input DataFrame
input_df = pd.DataFrame([v_inputs + [amount, time]], columns=[f"V{i}" for i in range(1, 29)] + ['Amount', 'Time'])
input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])
correct_order = xgb_model.feature_names_in_
input_df = input_df[correct_order]

# Prediction
if st.button("🔍 Predict Transaction"):
    col1, col2 = st.columns(2)

    for model, name, col in zip([xgb_model, knn_model], ["XGBoost", "KNN"], [col1, col2]):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        with col:
            st.subheader(f"{name} Prediction")
            if pred == 1:
                st.error(f"🚨 Fraudulent Transaction\nProbability: {prob:.2f}" if prob else "🚨 Fraud!")
            else:
                st.success(f"✅ Legitimate Transaction\nProbability: {1 - prob:.2f}" if prob else "✅ Legit")

# Load CSV data from GitHub releases
@st.cache_data
def load_data():
    url = "https://github.com/Rajatkumarsahu1/Fraud-Detection-using-KNN-and-XGB-GCV/releases/download/v1.0/creditcard.csv"
    df = pd.read_csv(url)
    return df.sample(5000, random_state=42)

df = load_data()

# Visualizations
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

# Footer with your portfolio link
# Footer with your portfolio, LinkedIn, Medium, and email
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        <strong>Made by <a href='https://www.linkedin.com/in/rajat-kumar-sahu1/' target='_blank'>Rajat Kumar Sahu</a></strong><br>
        📧 <a href='mailto:rajatks1997@gmail.com'>rajatks1997@gmail.com</a> &nbsp; | &nbsp;
        <a href='https://www.datascienceportfol.io/rajatks' target='_blank'>Portfolio</a> &nbsp; | &nbsp;
        <a href='https://medium.com/@pythonshortcodes' target='_blank'>Medium</a>
    </div>
    """,
    unsafe_allow_html=True
)


