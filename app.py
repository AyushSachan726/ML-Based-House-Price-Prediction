"""
Streamlit Web Application for House Price Prediction.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    model_data = joblib.load("models/best_model.pkl")
    preprocessor_data = joblib.load("models/preprocessor.pkl")
    return model_data, preprocessor_data


def main():
    # Page config
    st.set_page_config(
        page_title="🏠 House Price Predictor",
        page_icon="🏠",
        layout="wide"
    )

    # Header
    st.title("🏠 ML-Based House Price Prediction")
    st.markdown("---")

    # Check if model exists
    if not os.path.exists("models/best_model.pkl"):
        st.error("⚠️ Model not found! Please run `python main.py` first to train the model.")
        st.code("python main.py", language="bash")
        return

    # Load model
    model_data, preprocessor_data = load_model_and_preprocessor()
    model = model_data["model"]
    model_name = model_data["model_name"]
    scaler = preprocessor_data["scaler"]
    label_encoders = preprocessor_data["label_encoders"]

    # Sidebar - Model info
    st.sidebar.header("📊 Model Information")
    st.sidebar.success(f"**Model:** {model_name}")
    st.sidebar.info(f"**R² Score:** {model_data['metrics']['R2_Score']:.4f}")
    st.sidebar.info(f"**RMSE:** ${model_data['metrics']['RMSE']:,.2f}")

    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.write(
        "This app uses Machine Learning to predict house prices based on "
        "various features like area, bedrooms, location, and more."
    )

    # Input form
    st.header("📝 Enter House Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Area (sqft)", min_value=200, max_value=10000,
                               value=1500, step=50)
        bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
        stories = st.selectbox("Stories", options=[1, 2, 3], index=0)
        parking = st.selectbox("Parking Spots", options=[0, 1, 2, 3], index=1)

    with col2:
        age = st.slider("House Age (years)", 0, 50, 10)
        location = st.selectbox("Location", options=[
            "Downtown", "Suburban", "Rural", "Midtown", "Uptown",
            "Westside", "Eastside", "Northend", "Southend", "Lakeview"
        ])
        furnishing = st.selectbox("Furnishing", options=[
            "Furnished", "Semi-Furnished", "Unfurnished"
        ])

    with col3:
        road_access = st.selectbox("Road Access", options=["Yes", "No"])
        guestroom = st.selectbox("Guestroom", options=["Yes", "No"], index=1)
        basement = st.selectbox("Basement", options=["Yes", "No"], index=1)
        hot_water = st.selectbox("Hot Water Heating", options=["Yes", "No"])
        ac = st.selectbox("Air Conditioning", options=["Yes", "No"])
        preferred_area = st.selectbox("Preferred Area", options=["Yes", "No"], index=1)

    st.markdown("---")

    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            "Area_sqft": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Stories": [stories],
            "Parking": [parking],
            "Age_years": [age],
            "Location": [location],
            "Furnishing": [furnishing],
            "Road_access": [road_access],
            "Guestroom": [guestroom],
            "Basement": [basement],
            "Hot_water": [hot_water],
            "AC": [ac],
            "Preferred_area": [preferred_area],
        })

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Display result
        st.markdown("---")
        st.header("💰 Predicted House Price")

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.metric(
                label="Estimated Price",
                value=f"${prediction:,.2f}",
            )

        # Price range estimate
        rmse = model_data["metrics"]["RMSE"]
        st.info(
            f"📊 **Price Range Estimate:** "
            f"${max(0, prediction - rmse):,.2f} — ${prediction + rmse:,.2f}"
        )

    # Show sample data
    st.markdown("---")
    if st.checkbox("📋 Show Sample Dataset"):
        if os.path.exists("data/housing_data.csv"):
            df = pd.read_csv("data/housing_data.csv")
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.warning("Dataset not found. Run `python main.py` first.")


if __name__ == "__main__":
    main()
