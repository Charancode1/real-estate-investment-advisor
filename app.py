import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# ----------------------------
# Load Models & Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    clf = joblib.load("best_classifier.pkl")
    reg = joblib.load("best_regressor.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_list.json", "r") as f:
        feature_list = json.load(f)
    return clf, reg, scaler, feature_list


clf_model, reg_model, scaler, feature_list = load_artifacts()

st.title("🏡 Real Estate Investment Advisor")
st.subheader("AI-powered prediction for Good Investment & 5-year Future Price")

st.write("---")

# ---------------------------------------------------------
# USER INPUT FORM
# ---------------------------------------------------------
st.sidebar.header("Enter Property Details")

def user_input():
    BHK = st.sidebar.number_input("BHK", 1, 10, 3)
    Size = st.sidebar.number_input("Size (Sq Ft)", 300, 10000, 1200)
    Price = st.sidebar.number_input("Current Price (Lakhs)", 1, 10000, 120)
    YearBuilt = st.sidebar.number_input("Year Built", 1980, 2024, 2010)
    FloorNo = st.sidebar.number_input("Floor Number", 0, 50, 2)
    TotalFloors = st.sidebar.number_input("Total Floors", 1, 50, 10)
    NearbySchools = st.sidebar.slider("Nearby Schools (0–10)", 0, 10, 5)
    NearbyHospitals = st.sidebar.slider("Nearby Hospitals (0–10)", 0, 10, 5)
    AmenitiesCount = st.sidebar.slider("Amenities Count (0–10)", 0, 10, 3)
    SecurityScore = st.sidebar.slider("Security Score (0–1)", 0.0, 1.0, 0.5)
    TransportScore = st.sidebar.slider("Transport Score (0–1)", 0.0, 1.0, 0.5)

    data = {
        "BHK": BHK,
        "Size_in_SqFt": Size,
        "Price_in_Lakhs": Price,
        "Price_per_SqFt": Price / Size,
        "Year_Built": YearBuilt,
        "Floor_No": FloorNo,
        "Total_Floors": TotalFloors,
        "Age_of_Property": 2024 - YearBuilt,
        "Nearby_Schools": NearbySchools,
        "Nearby_Hospitals": NearbyHospitals,
        "Amenities_count": AmenitiesCount,
        "Security_score": SecurityScore,
        "Transport_score": TransportScore,
    }

    return pd.DataFrame([data])

input_df = user_input()

# ---------------------------------------------------------
# FEATURE MATCHING
# ---------------------------------------------------------
def prepare_input(df):
    # Fill missing columns
    for col in feature_list:
        if col not in df:
            df[col] = 0

    df = df[feature_list]
    df_scaled = scaler.transform(df)
    return df_scaled

final_input = prepare_input(input_df)

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
st.write("### 🔍 AI Predictions")

investment_pred = clf_model.predict(final_input)[0]
investment_prob = clf_model.predict_proba(final_input)[0][1]

future_price = reg_model.predict(final_input)[0]

col1, col2 = st.columns(2)

with col1:
    st.write("### 📌 Investment Decision")
    if investment_pred == 1:
        st.success(f"✔ Good Investment (Confidence: {investment_prob:.2f})")
    else:
        st.error(f"❌ Not a Good Investment (Confidence: {1 - investment_prob:.2f})")

with col2:
    st.write("### 💰 Estimated Future Price (5 Years)")
    st.info(f"Predicted Price: **₹ {future_price:.2f} Lakhs**")

st.write("---")

# ---------------------------------------------------------
# ADD VISUAL INSIGHTS
# ---------------------------------------------------------
st.write("## 📊 Visual Insights")

tab1, tab2 = st.tabs(["Feature Importance", "Correlation Heatmap"])

with tab1:
    st.write("### Feature Importance (Classifier)")
    try:
        importances = clf_model.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 12))
        idx = np.argsort(importances)[::-1][:20]
        ax.barh(np.array(feature_list)[idx][::-1], importances[idx][::-1])
        ax.set_title("Top 20 Important Features")
        st.pyplot(fig)
    except:
        st.warning("Feature importance not available for this classifier.")

with tab2:
    st.write("### Feature Correlation Heatmap")
    try:
        importances = reg_model.coef_
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(np.reshape(importances, (1, -1)), 
                    cmap="coolwarm", 
                    xticklabels=feature_list,
                    yticklabels=["Impact"])
        st.pyplot(fig)
    except:
        st.warning("Heatmap unavailable for this model type.")
