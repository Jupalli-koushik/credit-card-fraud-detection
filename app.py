import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Config ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# --- Load Models (Cached for Speed) ---
@st.cache_resource
def load_resources():
    model = joblib.load('artifacts/fraud_model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_resources()
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'train_model.py' first.")
    st.stop()

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Performance", "Detect Fraud"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("""
    Welcome to the Fraud Detection Dashboard. This tool uses a **Random Forest Classifier** trained on historical transaction data to identify suspicious activities.
    
    ### How to use:
    1. Go to **Detect Fraud** in the sidebar.
    2. Upload a CSV file containing transaction data.
    3. The system will flag high-risk transactions instantly.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*9QUL9_wS8kM4aL5n4gwTqg.png", caption="Fraud Detection Workflow")

# --- PAGE 2: PERFORMANCE ---
elif page == "Model Performance":
    st.title("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "99.9%")
    col2.metric("Fraud Recall", "83%", "High Sensitivity")
    col3.metric("Precision", "85%", "Low False Alarms")
    
    st.subheader("Confusion Matrix Analysis")
    st.write("""
    - **True Positives (81):** Frauds correctly caught.
    - **False Negatives (17):** Frauds missed (Critical area for improvement).
    - **False Positives (14):** Legitimate transactions flagged as fraud.
    """)
    
    # You can plot the feature importance here if you want to be fancy
    st.subheader("Top Risk Factors")
    feat_importances = pd.Series(model.feature_importances_, index=[f"V{i}" for i in range(1, 29)] + ['scaled_amount'])
    fig, ax = plt.subplots()
    feat_importances.nlargest(10).plot(kind='barh', ax=ax, color='#ff4b4b')
    st.pyplot(fig)

# --- PAGE 3: DETECT FRAUD ---
elif page == "Detect Fraud":
    st.title("🚨 Fraud Detection Engine")
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 1. Read Data
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", df.head())
            
            if st.button("Analyze Transactions"):
                # 2. Preprocess (Mirror the steps from Day 1)
                # We need to scale 'Amount' and drop 'Time'
                if 'Amount' in df.columns:
                    df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1,1))
                    data_for_pred = df.drop(['Time', 'Amount', 'Class'], axis=1, errors='ignore') # Drop Class if it exists
                else:
                    st.warning("Column 'Amount' not found. Using raw data (might be inaccurate).")
                    data_for_pred = df
                
                # 3. Predict
                predictions = model.predict(data_for_pred)
                probabilities = model.predict_proba(data_for_pred)[:, 1]
                
                # 4. Results
                df['Fraud_Probability'] = probabilities
                df['Prediction'] = predictions
                df['Prediction'] = df['Prediction'].apply(lambda x: "⚠️ FRAUD" if x == 1 else "✅ Legit")
                
                fraud_count = len(df[df['Prediction'] == "⚠️ FRAUD"])
                
                if fraud_count > 0:
                    st.error(f"ALERT: {fraud_count} suspicious transactions detected!")
                    st.dataframe(df[df['Prediction'] == "⚠️ FRAUD"].style.background_gradient(subset=['Fraud_Probability'], cmap='Reds'))
                else:
                    st.success("No suspicious transactions found.")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")