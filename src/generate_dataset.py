"""
Generate synthetic housing dataset for ML-Based House Price Prediction.
"""

import numpy as np
import pandas as pd
import os


def generate_dataset(n_samples=50000, random_state=42):
    """Generate a realistic, large-scale synthetic housing dataset (Kaggle Benchmark Standard)."""
    np.random.seed(random_state)

    # Location-based pricing tiers
    locations = [
        "Downtown", "Suburban", "Rural", "Midtown", "Uptown",
        "Westside", "Eastside", "Northend", "Southend", "Lakeview"
    ]
    location_multiplier = {
        "Downtown": 1.55, "Suburban": 1.0, "Rural": 0.72, "Midtown": 1.35,
        "Uptown": 1.42, "Westside": 1.12, "Eastside": 0.92, "Northend": 0.88,
        "Southend": 0.96, "Lakeview": 1.65
    }

    # Generate features
    data = {
        "Area_sqft": np.random.randint(500, 5200, n_samples),
        "Bedrooms": np.random.randint(1, 7, n_samples),
        "Bathrooms": np.random.randint(1, 5, n_samples),
        "Stories": np.random.randint(1, 4, n_samples),
        "Parking": np.random.randint(0, 4, n_samples),
        "Age_years": np.random.randint(0, 50, n_samples),
        "Location": np.random.choice(locations, n_samples),
        "Furnishing": np.random.choice(
            ["Furnished", "Semi-Furnished", "Unfurnished"], n_samples
        ),
        "Road_access": np.random.choice(["Yes", "No"], n_samples, p=[0.82, 0.18]),
        "Guestroom": np.random.choice(["Yes", "No"], n_samples, p=[0.32, 0.68]),
        "Basement": np.random.choice(["Yes", "No"], n_samples, p=[0.28, 0.72]),
        "Hot_water": np.random.choice(["Yes", "No"], n_samples, p=[0.62, 0.38]),
        "AC": np.random.choice(["Yes", "No"], n_samples, p=[0.55, 0.45]),
        "Preferred_area": np.random.choice(["Yes", "No"], n_samples, p=[0.38, 0.62]),
    }

    df = pd.DataFrame(data)

    # Calculate price based on features (realistic real estate formula)
    base_price = 50000
    price = (
        base_price
        + df["Area_sqft"] * 120
        + df["Bedrooms"] * 14000
        + df["Bathrooms"] * 12000
        + df["Stories"] * 18000
        + df["Parking"] * 10000
        - df["Age_years"] * 1800
        + df["Guestroom"].map({"Yes": 22000, "No": 0})
        + df["Basement"].map({"Yes": 28000, "No": 0})
        + df["AC"].map({"Yes": 20000, "No": 0})
        + df["Preferred_area"].map({"Yes": 45000, "No": 0})
        + df["Furnishing"].map({
            "Furnished": 38000, "Semi-Furnished": 18000, "Unfurnished": 0
        })
        + df["Road_access"].map({"Yes": 14000, "No": 0})
        + df["Hot_water"].map({"Yes": 9500, "No": 0})
    )

    # Apply location multiplier
    location_mult = df["Location"].map(location_multiplier)
    price = price * location_mult

    # Realistic market variance (unobserved condition, negotiations, seasonal spread: ~13.5%)
    market_variance = np.random.normal(1.0, 0.135, n_samples)
    price = price * market_variance
    price = np.maximum(price, 35000)  # Minimum price bound

    df["Price"] = np.round(price, 2)

    return df


if __name__ == "__main__":
    # Generate and save dataset
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=50000)
    df.to_csv("data/housing_data.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"\nDataset Summary:")
    print(df.describe())
    print(f"\nSaved to: data/housing_data.csv")
