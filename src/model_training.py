"""
Model Training Module for House Price Prediction.
Trains and compares multiple regression models.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import time


class ModelTrainer:
    """Handles model training, evaluation, and comparison."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def get_models(self):
        """Return dictionary of models to train."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1,
                verbosity=0
            ),
        }
        return models

    def evaluate_model(self, y_true, y_pred):
        """Calculate regression evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4),
            "MAPE (%)": round(mape, 2),
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate their performance."""
        print("\n🤖 Starting Model Training...")
        print("=" * 70)

        models = self.get_models()

        for name, model in models.items():
            print(f"\n📌 Training: {name}")
            start_time = time.time()

            # Train
            model.fit(X_train, y_train)
            training_time = round(time.time() - start_time, 2)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred)
            metrics["Training_Time (s)"] = training_time

            # Store results
            self.models[name] = model
            self.results[name] = metrics

            print(f"   R² Score: {metrics['R2_Score']:.4f} | "
                  f"RMSE: ${metrics['RMSE']:,.2f} | "
                  f"MAE: ${metrics['MAE']:,.2f} | "
                  f"Time: {training_time}s")

        print("\n" + "=" * 70)
        self._find_best_model()
        self._print_comparison()

        return self.results

    def _find_best_model(self):
        """Find the best model based on R² Score."""
        best_score = -float("inf")
        for name, metrics in self.results.items():
            if metrics["R2_Score"] > best_score:
                best_score = metrics["R2_Score"]
                self.best_model_name = name
                self.best_model = self.models[name]

        print(f"\n🏆 Best Model: {self.best_model_name} "
              f"(R² = {best_score:.4f})")

    def _print_comparison(self):
        """Print model comparison table."""
        print("\n📊 MODEL COMPARISON TABLE")
        print("-" * 90)
        header = f"{'Model':<25} {'R² Score':<12} {'RMSE ($)':<15} {'MAE ($)':<15} {'MAPE (%)':<12} {'Time (s)':<10}"
        print(header)
        print("-" * 90)

        # Sort by R² Score
        sorted_results = sorted(
            self.results.items(), key=lambda x: x[1]["R2_Score"], reverse=True
        )

        for name, metrics in sorted_results:
            marker = " 🏆" if name == self.best_model_name else ""
            row = (
                f"{name:<25} {metrics['R2_Score']:<12.4f} "
                f"{metrics['RMSE']:<15,.2f} {metrics['MAE']:<15,.2f} "
                f"{metrics['MAPE (%)']:<12.2f} "
                f"{metrics['Training_Time (s)']:<10}{marker}"
            )
            print(row)

        print("-" * 90)

    def save_best_model(self, filepath="models/best_model.pkl"):
        """Save the best performing model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "metrics": self.results[self.best_model_name],
        }
        joblib.dump(model_data, filepath)
        print(f"\n💾 Best model ({self.best_model_name}) saved to: {filepath}")

    def save_all_models(self, directory="models/"):
        """Save all trained models."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            filename = name.lower().replace(" ", "_") + ".pkl"
            filepath = os.path.join(directory, filename)
            joblib.dump(model, filepath)
        print(f"💾 All {len(self.models)} models saved to: {directory}")

    @staticmethod
    def load_model(filepath="models/best_model.pkl"):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        print(f"✅ Model loaded: {model_data['model_name']}")
        print(f"   R² Score: {model_data['metrics']['R2_Score']}")
        return model_data
