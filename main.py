"""
Main Pipeline Script for ML-Based House Price Prediction.
Runs the complete pipeline: data generation → preprocessing → training → evaluation.
"""

import os
import sys
import numpy as np

# Configure UTF-8 for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate_dataset import generate_dataset
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.visualize import (
    plot_price_distribution,
    plot_correlation_heatmap,
    plot_feature_vs_price,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance,
)


def main():
    """Run the complete ML pipeline."""
    print("🏠 ML-Based House Price Prediction Pipeline")
    print("=" * 60)

    # ----- Step 1: Generate Dataset -----
    print("\n📁 Step 1: Generating Dataset...")
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=5000)
    df.to_csv("data/housing_data.csv", index=False)
    print(f"   Generated {df.shape[0]} samples with {df.shape[1]} features")

    # ----- Step 2: EDA Visualizations -----
    print("\n📊 Step 2: Creating EDA Visualizations...")
    os.makedirs("outputs", exist_ok=True)
    plot_price_distribution(df)
    plot_correlation_heatmap(df)

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove("Price")
    plot_feature_vs_price(df, numerical_features)

    # ----- Step 3: Data Preprocessing -----
    print("\n🔧 Step 3: Data Preprocessing...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(
        "data/housing_data.csv"
    )

    # ----- Step 4: Model Training & Evaluation -----
    print("\n🤖 Step 4: Training Models...")
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    # ----- Step 5: Save Models -----
    print("\n💾 Step 5: Saving Models...")
    os.makedirs("models", exist_ok=True)
    trainer.save_best_model()
    trainer.save_all_models()

    # ----- Step 6: Generate Evaluation Plots -----
    print("\n📈 Step 6: Generating Evaluation Plots...")
    plot_model_comparison(results)

    # Actual vs Predicted for best model
    best_model = trainer.best_model
    y_pred = best_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred, trainer.best_model_name)

    # Feature importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_names, trainer.best_model_name)

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Dataset:       data/housing_data.csv")
    print(f"🤖 Best Model:    {trainer.best_model_name}")
    print(f"📊 R² Score:      {results[trainer.best_model_name]['R2_Score']:.4f}")
    print(f"💰 RMSE:          ${results[trainer.best_model_name]['RMSE']:,.2f}")
    print(f"📈 Visualizations: outputs/")
    print(f"💾 Models:         models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
