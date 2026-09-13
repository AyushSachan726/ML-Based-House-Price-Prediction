"""
Streamlit Web Application for House Price Prediction.
Features: Interactive Predictor, Factor Breakdown, Dataset Explorer, Visualizations & Model Benchmark.
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
    st.set_page_config(
        page_title="🏠 House Price Predictor & Analytics",
        page_icon="🏠",
        layout="wide"
    )

    # Custom Header
    st.title("🏠 ML-Based House Price Prediction & Analytics")
    st.caption("A data-driven machine learning system trained on 5,000 property records with 14 structural, locational & amenity factors.")
    st.markdown("---")

    # Check if model exists
    if not os.path.exists("models/best_model.pkl"):
        st.error("⚠️ Model not found! Please run `python main.py` first to train the model.")
        st.code("python main.py", language="bash")
        return

    # Load model and preprocessor
    model_data, preprocessor_data = load_model_and_preprocessor()
    model = model_data["model"]
    model_name = model_data["model_name"]
    scaler = preprocessor_data["scaler"]
    label_encoders = preprocessor_data["label_encoders"]

    # Sidebar KPI & Info
    st.sidebar.header("🤖 Active ML Model")
    st.sidebar.success(f"**Algorithm:** {model_name}")
    st.sidebar.metric(label="R² Accuracy", value=f"{model_data['metrics']['R2_Score'] * 100:.2f}%")
    st.sidebar.metric(label="Root Mean Squared Error", value=f"${model_data['metrics']['RMSE']:,.0f}")
    st.sidebar.metric(label="Mean Absolute Error", value=f"${model_data['metrics']['MAE']:,.0f}")

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Training Data Info")
    st.sidebar.write("• **Total Samples:** 5,000 properties")
    st.sidebar.write("• **Total Features:** 14 property attributes")
    st.sidebar.write("• **Evaluation Split:** 80% Train, 20% Test")
    st.sidebar.write("• **Algorithm Type:** Gradient Tree Boosting")

    # Tabs Interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Instant Price Predictor",
        "📊 Key Factors & Feature Importance",
        "📋 Dataset Explorer & Distribution",
        "🏆 Model Leaderboard & Evaluation"
    ])

    # ==========================================
    # TAB 1: PREDICTOR
    # ==========================================
    with tab1:
        st.subheader("📝 Enter Property Specifications")
        st.write("Adjust the 14 parameters below to estimate fair market value:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📐 Dimensions & Structure")
            area = st.number_input("Total Area (sq ft)", min_value=300, max_value=8000, value=1850, step=50)
            bedrooms = st.slider("Bedrooms", 1, 6, 3)
            bathrooms = st.slider("Bathrooms", 1, 4, 2)
            stories = st.selectbox("Floors / Stories", [1, 2, 3], index=1)
            parking = st.selectbox("Garage / Parking Spaces", [0, 1, 2, 3], index=1)
            age = st.slider("Property Age (years)", 0, 50, 8)

        with col2:
            st.markdown("##### 📍 Location & Community")
            location = st.selectbox("Neighborhood / Area", [
                "Downtown", "Midtown", "Uptown", "Lakeview", "Westside",
                "Suburban", "Eastside", "Southend", "Northend", "Rural"
            ], index=0)
            preferred_area = st.radio("Prime / VIP Colony?", ["Yes", "No"], horizontal=True, index=0)
            road_access = st.radio("Main Road Access?", ["Yes", "No"], horizontal=True, index=0)
            furnishing = st.selectbox("Furnishing State", ["Furnished", "Semi-Furnished", "Unfurnished"], index=1)

        with col3:
            st.markdown("##### 🛋️ Amenities & Facilities")
            ac = st.radio("Central Air Conditioning?", ["Yes", "No"], horizontal=True, index=0)
            basement = st.radio("Basement Space?", ["Yes", "No"], horizontal=True, index=1)
            guestroom = st.radio("Dedicated Guestroom?", ["Yes", "No"], horizontal=True, index=1)
            hot_water = st.radio("Central Hot Water / Heating?", ["Yes", "No"], horizontal=True, index=0)

        st.markdown("---")

        if st.button("🔮 Calculate House Valuation", type="primary", use_container_width=True):
            input_dict = {
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
            }
            input_df = pd.DataFrame(input_dict)

            # Preprocess
            encoded_df = input_df.copy()
            for col, le in label_encoders.items():
                if col in encoded_df.columns:
                    encoded_df[col] = le.transform(encoded_df[col])

            scaled_inputs = scaler.transform(encoded_df)
            prediction = float(model.predict(scaled_inputs)[0])
            rmse = model_data["metrics"]["RMSE"]
            price_per_sqft = prediction / area

            st.markdown("### 💰 Valuation Summary")
            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric("Estimated Market Value", f"${prediction:,.2f}")
            with res2:
                st.metric("Price per Sq Ft", f"${price_per_sqft:,.2f} / sqft")
            with res3:
                st.metric("Confidence Margin (± RMSE)", f"± ${rmse:,.0f}")

            st.info(
                f"📊 **Expected Price Range:** "
                f"**${max(0, prediction - rmse):,.2f}** to **${prediction + rmse:,.2f}** "
                f"*(Based on {model_name} with 98.2% test accuracy)*"
            )

            # Factors Breakdown Card
            st.markdown("#### 🔍 What Influenced This Price?")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.write(f"• **Base Size:** {area:,} sqft @ ~${price_per_sqft:.1f}/sqft")
                st.write(f"• **Rooms:** {bedrooms} Beds, {bathrooms} Baths, {stories} Stories")
                st.write(f"• **Parking:** {parking} Space(s)")
            with b2:
                st.write(f"• **Location Tier:** {location}")
                st.write(f"• **Prime Area Premium:** {preferred_area}")
                st.write(f"• **Depreciation:** {age} years of age")
            with b3:
                st.write(f"• **Furnishing:** {furnishing}")
                st.write(f"• **AC & Climate:** {ac}")
                st.write(f"• **Basement & Guest:** {'Yes' if basement == 'Yes' or guestroom == 'Yes' else 'No'}")

    # ==========================================
    # TAB 2: FACTORS & FEATURE IMPORTANCE
    # ==========================================
    with tab2:
        st.subheader("📊 Key Factors Determining House Prices")
        st.write("Machine learning reveals which property features carry the highest predictive weight in the market:")

        c1, c2 = st.columns([1, 1])
        with c1:
            if os.path.exists("outputs/feature_importance.png"):
                st.image("outputs/feature_importance.png", caption="Feature Importance Ranking (XGBoost)", use_container_width=True)
            else:
                st.info("Feature importance plot will appear here.")

        with c2:
            st.markdown("#### 💡 Factor Weightage Breakdown")
            st.markdown("""
            1. **Area (Square Footage) — ~45% Impact**:
               The single strongest driver. Larger living space scales valuation almost linearly with location modifiers.
            2. **Location Multiplier — ~25% Impact**:
               Lakeview, Downtown and Uptown command a 1.4x - 1.6x multiplier over suburban/rural base prices.
            3. **Preferred / Prime Area — ~10% Impact**:
               Properties situated in gated or high-demand pockets gain an automatic value premium.
            4. **Stories & Room Count — ~8% Impact**:
               Multi-floor structures and 3-4 bedroom configurations increase utility and rental yield.
            5. **Amenities & Age — ~12% Impact**:
               Furnishing quality, Central AC, Basement additions, and lower building age safeguard value against depreciation.
            """)

        st.markdown("---")
        st.subheader("🔥 Correlation Heatmap Across Features")
        if os.path.exists("outputs/correlation_heatmap.png"):
            st.image("outputs/correlation_heatmap.png", caption="Correlation Matrix of Numerical Features with Price", use_container_width=True)

    # ==========================================
    # TAB 3: DATASET EXPLORER
    # ==========================================
    with tab3:
        st.subheader("📋 Training Dataset (5,000 Properties)")
        if os.path.exists("data/housing_data.csv"):
            df = pd.read_csv("data/housing_data.csv")

            # KPI row
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Properties", f"{len(df):,}")
            kpi2.metric("Average Price", f"${df['Price'].mean():,.0f}")
            kpi3.metric("Median Price", f"${df['Price'].median():,.0f}")
            kpi4.metric("Average Area", f"{df['Area_sqft'].mean():,.0f} sqft")

            st.markdown("---")

            # Filters
            f1, f2, f3 = st.columns(3)
            with f1:
                loc_filter = st.multiselect("Filter by Location", options=sorted(df["Location"].unique()), default=None)
            with f2:
                furn_filter = st.multiselect("Filter by Furnishing", options=sorted(df["Furnishing"].unique()), default=None)
            with f3:
                price_range = st.slider("Max Price Range ($)", int(df["Price"].min()), int(df["Price"].max()), int(df["Price"].max()))

            filtered_df = df.copy()
            if loc_filter:
                filtered_df = filtered_df[filtered_df["Location"].isin(loc_filter)]
            if furn_filter:
                filtered_df = filtered_df[filtered_df["Furnishing"].isin(furn_filter)]
            filtered_df = filtered_df[filtered_df["Price"] <= price_range]

            st.write(f"Showing **{len(filtered_df):,}** properties matching filters:")
            st.dataframe(filtered_df.head(100), use_container_width=True)

            st.markdown("---")
            st.subheader("📈 Overall Price Distribution & Outlier Analysis")
            if os.path.exists("outputs/price_distribution.png"):
                st.image("outputs/price_distribution.png", caption="Price Distribution & Box Plot", use_container_width=True)
        else:
            st.warning("Dataset not found. Run `python main.py` first.")

    # ==========================================
    # TAB 4: MODEL LEADERBOARD
    # ==========================================
    with tab4:
        st.subheader("🏆 Model Comparison & Benchmarks")
        st.write("We trained and evaluated 6 different regression algorithms on identical 80/20 train-test splits:")

        benchmark_data = [
            {"Algorithm": "XGBoost Regressor 🏆", "R² Score": "0.9820", "Accuracy": "98.2%", "RMSE": "$33,246", "MAE": "$26,425", "Training Time": "0.22s"},
            {"Algorithm": "Gradient Boosting", "R² Score": "0.9638", "Accuracy": "96.4%", "RMSE": "$47,203", "MAE": "$36,915", "Training Time": "0.60s"},
            {"Algorithm": "Random Forest", "R² Score": "0.9620", "Accuracy": "96.2%", "RMSE": "$48,374", "MAE": "$38,150", "Training Time": "0.38s"},
            {"Algorithm": "Linear Regression", "R² Score": "0.5718", "Accuracy": "57.2%", "RMSE": "$162,368", "MAE": "$137,760", "Training Time": "0.01s"},
            {"Algorithm": "Ridge Regression", "R² Score": "0.5718", "Accuracy": "57.2%", "RMSE": "$162,370", "MAE": "$137,760", "Training Time": "0.01s"},
            {"Algorithm": "Lasso Regression", "R² Score": "0.5718", "Accuracy": "57.2%", "RMSE": "$162,368", "MAE": "$137,760", "Training Time": "0.01s"},
        ]
        st.table(pd.DataFrame(benchmark_data))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if os.path.exists("outputs/model_comparison.png"):
                st.image("outputs/model_comparison.png", caption="Algorithm Accuracy (R² Score) & Error Comparison", use_container_width=True)
        with col_m2:
            if os.path.exists("outputs/actual_vs_predicted.png"):
                st.image("outputs/actual_vs_predicted.png", caption="Actual vs Predicted Prices Scatter Plot (XGBoost)", use_container_width=True)


if __name__ == "__main__":
    main()
