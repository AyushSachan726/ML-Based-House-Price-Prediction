"""
Visualization Module for House Price Prediction.
Creates charts and plots for EDA and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def set_style():
    """Set consistent plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 12


def plot_price_distribution(df, save_path="outputs/price_distribution.png"):
    """Plot the distribution of house prices."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df["Price"], bins=50, color="#4CAF50", edgecolor="black", alpha=0.7)
    axes[0].set_title("House Price Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Frequency")

    # Box plot
    axes[1].boxplot(df["Price"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#2196F3", alpha=0.7))
    axes[1].set_title("House Price Box Plot", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Price ($)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_correlation_heatmap(df, save_path="outputs/correlation_heatmap.png"):
    """Plot correlation heatmap of numerical features."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_style()

    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0,
        fmt=".2f", linewidths=0.5, square=True
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_feature_vs_price(df, features, save_path="outputs/feature_vs_price.png"):
    """Plot scatter plots of top features vs price."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_style()

    n_features = min(len(features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for i in range(n_features):
        feature = features[i]
        axes[i].scatter(df[feature], df["Price"], alpha=0.4, color=colors[i], s=15)
        axes[i].set_xlabel(feature, fontsize=11)
        axes[i].set_ylabel("Price ($)", fontsize=11)
        axes[i].set_title(f"{feature} vs Price", fontsize=12, fontweight="bold")

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature vs Price Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_model_comparison(results, save_path="outputs/model_comparison.png"):
    """Plot model comparison bar chart."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_style()

    model_names = list(results.keys())
    r2_scores = [results[m]["R2_Score"] for m in model_names]
    rmse_values = [results[m]["RMSE"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # R² Score comparison
    colors = sns.color_palette("husl", len(model_names))
    bars1 = axes[0].barh(model_names, r2_scores, color=colors, edgecolor="black")
    axes[0].set_xlabel("R² Score", fontsize=12)
    axes[0].set_title("Model Comparison - R² Score", fontsize=14, fontweight="bold")
    for bar, score in zip(bars1, r2_scores):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{score:.4f}", va="center", fontweight="bold")

    # RMSE comparison
    bars2 = axes[1].barh(model_names, rmse_values, color=colors, edgecolor="black")
    axes[1].set_xlabel("RMSE ($)", fontsize=12)
    axes[1].set_title("Model Comparison - RMSE", fontsize=14, fontweight="bold")
    for bar, rmse in zip(bars2, rmse_values):
        axes[1].text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                     f"${rmse:,.0f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_actual_vs_predicted(y_actual, y_predicted, model_name,
                              save_path="outputs/actual_vs_predicted.png"):
    """Plot actual vs predicted values."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    axes[0].scatter(y_actual, y_predicted, alpha=0.4, color="#4CAF50", s=15)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
                 label="Perfect Prediction")
    axes[0].set_xlabel("Actual Price ($)", fontsize=12)
    axes[0].set_ylabel("Predicted Price ($)", fontsize=12)
    axes[0].set_title(f"Actual vs Predicted - {model_name}",
                      fontsize=14, fontweight="bold")
    axes[0].legend()

    # Residuals plot
    residuals = y_actual - y_predicted
    axes[1].scatter(y_predicted, residuals, alpha=0.4, color="#FF6B6B", s=15)
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted Price ($)", fontsize=12)
    axes[1].set_ylabel("Residuals ($)", fontsize=12)
    axes[1].set_title(f"Residual Plot - {model_name}",
                      fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_feature_importance(model, feature_names, model_name,
                             save_path="outputs/feature_importance.png"):
    """Plot feature importance for tree-based models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_style()

    if not hasattr(model, "feature_importances_"):
        print(f"⚠️  {model_name} does not support feature importance")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(feature_names))
    plt.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=colors,
        edgecolor="black"
    )
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")
