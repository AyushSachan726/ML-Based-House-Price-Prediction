"""
Data Preprocessing Module for House Price Prediction.
Handles data loading, cleaning, encoding, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import shutil


class DataPreprocessor:
    """Handles all data preprocessing steps."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = []
        self.numerical_columns = []

    def load_data(self, filepath):
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_csv(filepath, on_bad_lines="skip", engine="python")
        print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def explore_data(self, df):
        """Print basic dataset information."""
        print("\n" + "=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        print(f"\nShape: {df.shape}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{df.describe()}")
        print("=" * 60)

    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Fill numerical columns with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  Filled {col} with median: {df[col].median()}")

        # Fill categorical columns with mode
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  Filled {col} with mode: {df[col].mode()[0]}")

        print("[OK] Missing values handled")
        return df

    def encode_features(self, df, target_column="Price"):
        """Encode categorical features using Label Encoding."""
        self.categorical_columns = df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        self.numerical_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if target_column in self.numerical_columns:
            self.numerical_columns.remove(target_column)

        for col in self.categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        print(f"[OK] Encoded {len(self.categorical_columns)} categorical columns: "
              f"{self.categorical_columns}")
        return df

    def scale_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("[OK] Features scaled using StandardScaler")
        return X_train_scaled, X_test_scaled

    def split_data(self, df, target_column="Price", test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"[OK] Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def save_preprocessor(self, filepath="models/preprocessor.pkl"):
        """Save preprocessing objects to both models/ and model/."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.makedirs("model", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        preprocessor_data = {
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
        }
        joblib.dump(preprocessor_data, filepath)
        # Also copy to model/
        model_copy_path = os.path.join("model", os.path.basename(filepath))
        joblib.dump(preprocessor_data, model_copy_path)
        print(f"[OK] Preprocessor saved to: {filepath} and {model_copy_path}")

    def load_preprocessor(self, filepath="models/preprocessor.pkl"):
        """Load preprocessing objects with fallback to model/."""
        if not os.path.exists(filepath) and os.path.exists("model/" + os.path.basename(filepath)):
            filepath = "model/" + os.path.basename(filepath)
        preprocessor_data = joblib.load(filepath)
        self.label_encoders = preprocessor_data["label_encoders"]
        self.scaler = preprocessor_data["scaler"]
        self.categorical_columns = preprocessor_data["categorical_columns"]
        self.numerical_columns = preprocessor_data["numerical_columns"]
        print(f"[OK] Preprocessor loaded from: {filepath}")

    def preprocess_pipeline(self, filepath, target_column="Price"):
        """Complete preprocessing pipeline."""
        print("\n[Start] Data Preprocessing Pipeline...")
        print("-" * 50)

        # Load data
        df = self.load_data(filepath)

        # Explore data
        self.explore_data(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode features
        df = self.encode_features(df, target_column)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, target_column)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # Save preprocessor
        self.save_preprocessor()

        print("-" * 50)
        print("[Complete] Preprocessing Finished!\n")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()
