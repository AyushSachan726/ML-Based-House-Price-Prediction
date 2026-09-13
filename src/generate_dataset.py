"""
Generate synthetic housing dataset for ML-Based House Price Prediction.
"""

import numpy as np
import pandas as pd
import os


def generate_dataset(n_samples=5000, random_state=42):
    """Generate a realistic synthetic housing dataset."""
    np.random.seed(random_state)

    # Location-based pricing tiers
    locations = [
        "Downtown", "Suburban", "Rural", "Midtown", "Uptown",
        "Westside", "Eastside", "Northend", "Southend", "Lakeview"
    ]
    location_multiplier = {
        "Downtown": 1.5, "Suburban": 1.0, "Rural": 0.7, "Midtown": 1.3,
        "Uptown": 1.4, "Westside": 1.1, "Eastside": 0.9, "Northend": 0.85,
        "Southend": 0.95, "Lakeview": 1.6
    }

    # Generate features
    data = {
        "Area_sqft": np.random.randint(500, 5000, n_samples),
        "Bedrooms": np.random.randint(1, 7, n_samples),
        "Bathrooms": np.random.randint(1, 5, n_samples),
        "Stories": np.random.randint(1, 4, n_samples),
        "Parking": np.random.randint(0, 4, n_samples),
        "Age_years": np.random.randint(0, 50, n_samples),
        "Location": np.random.choice(locations, n_samples),
        "Furnishing": np.random.choice(
            ["Furnished", "Semi-Furnished", "Unfurnished"], n_samples
        ),
        "Road_access": np.random.choice(["Yes", "No"], n_samples, p=[0.8, 0.2]),
        "Guestroom": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
        "Basement": np.random.choice(["Yes", "No"], n_samples, p=[0.25, 0.75]),
        "Hot_water": np.random.choice(["Yes", "No"], n_samples, p=[0.6, 0.4]),
        "AC": np.random.choice(["Yes", "No"], n_samples, p=[0.5, 0.5]),
        "Preferred_area": np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65]),
    }

    df = pd.DataFrame(data)

    # Calculate price based on features (realistic formula)
    base_price = 50000
    price = (
        base_price
        + df["Area_sqft"] * 120
        + df["Bedrooms"] * 15000
        + df["Bathrooms"] * 12000
        + df["Stories"] * 20000
        + df["Parking"] * 10000
        - df["Age_years"] * 2000
        + df["Guestroom"].map({"Yes": 25000, "No": 0})
        + df["Basement"].map({"Yes": 30000, "No": 0})
        + df["AC"].map({"Yes": 20000, "No": 0})
        + df["Preferred_area"].map({"Yes": 50000, "No": 0})
        + df["Furnishing"].map({
            "Furnished": 40000, "Semi-Furnished": 20000, "Unfurnished": 0
        })
        + df["Road_access"].map({"Yes": 15000, "No": 0})
        + df["Hot_water"].map({"Yes": 10000, "No": 0})
    )

    # Apply location multiplier
    location_mult = df["Location"].map(location_multiplier)
    price = price * location_mult

    # Add noise
    noise = np.random.normal(0, 20000, n_samples)
    price = price + noise
    price = np.maximum(price, 30000)  # Minimum price

    df["Price"] = np.round(price, 2)

    return df


if __name__ == "__main__":
    # Generate and save dataset
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/housing_data.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"\nDataset Summary:")
    print(df.describe())
    print(f"\nSaved to: data/housing_data.csv")
