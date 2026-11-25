"""
Evaluation Module
==================
Evaluates every model on the same holdout test set and produces:
  - Metric tables (MAE, RMSE, R²)
  - Comparison charts
  - Feature importance
  - Residual analysis
  - SHAP summary (optional)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "charts")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute MAE, RMSE, R² for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every model and return a DataFrame sorted by MAE.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Test Set)")
    print("=" * 60)

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print(f"  {name:30s}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("MAE").reset_index(drop=True)
    df_results = df_results[["Model", "MAE", "RMSE", "R2"]]

    # Baseline comparison
    baseline_mae = df_results.loc[df_results["Model"] == "Dummy (Mean)", "MAE"].values
    if len(baseline_mae) > 0:
        baseline_mae = baseline_mae[0]
        df_results["Improvement_vs_Baseline_%"] = (
            (baseline_mae - df_results["MAE"]) / baseline_mae * 100
        ).round(2)
    else:
        df_results["Improvement_vs_Baseline_%"] = np.nan

    print(f"\n  Best model: {df_results.iloc[0]['Model']} (MAE={df_results.iloc[0]['MAE']:.4f})")
    print("=" * 60)
    return df_results


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_model_comparison_bar(df_results: pd.DataFrame):
    """
    Horizontal bar chart: Y-axis = model names, X-axis = MAE.
    Best model highlighted in a different colour.
    R² displayed alongside MAE at bar end.
    """
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_results) * 0.55)))

    # Sort worst → best so best is at the top visually
    df_plot = df_results.sort_values("MAE", ascending=True).reset_index(drop=True)

    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df_plot))]

    bars = ax.barh(df_plot["Model"], df_plot["MAE"], color=colors, edgecolor="white")

    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f'MAE={row["MAE"]:.2f}  R²={row["R2"]:.3f}',
                va="center", fontsize=9)

    ax.set_xlabel("Mean Absolute Error (minutes)", fontsize=12)
    ax.set_title("Model Comparison — MAE on Test Set", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "model_comparison_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metric_chart(df_results: pd.DataFrame, metric: str, title: str, filename: str):
    """Generic horizontal bar chart for a single metric."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_results) * 0.5)))
    df_plot = df_results.sort_values(metric, ascending=(metric != "R2"))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df_plot))]
    ax.barh(df_plot["Model"], df_plot[metric], color=colors, edgecolor="white")
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(row[metric] + 0.05 * df_plot[metric].max(), i,
                f'{row[metric]:.3f}', va="center", fontsize=9)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if metric != "R2":
        ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_actual_vs_predicted(model, X_test, y_test, model_name: str):
    """Scatter plot of actual vs predicted for the best model."""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, c="#3498db")
    mn = min(y_test.min(), y_pred.min()) - 2
    mx = max(y_test.max(), y_pred.max()) + 2
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Delivery Time (min)", fontsize=12)
    ax.set_ylabel("Predicted Delivery Time (min)", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "actual_vs_predicted.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals(model, X_test, y_test, model_name: str):
    """Residual distribution plot."""
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, c="#e74c3c")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Residual", fontsize=11)
    axes[0].set_title("Residuals vs Predicted", fontsize=13)

    # Distribution of residuals
    axes[1].hist(residuals, bins=50, edgecolor="white", color="#9b59b6", alpha=0.8)
    axes[1].set_xlabel("Residual (minutes)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=13)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "residual_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 20):
    """
    Feature importance for tree-based models.
    Extracts importances from the regressor step of the pipeline.
    """
    regressor = model.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        print(f"  {model_name} does not have feature_importances_ — skipping.")
        return None

    importances = regressor.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Save CSV
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"  Saved: {fi_path}")

    # Plot top N
    fi_top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(fi_top["Feature"][::-1], fi_top["Importance"][::-1], color="#f39c12", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fi_df


def plot_shap_summary(model, X_test, feature_names: list, model_name: str):
    """
    SHAP summary plot for the best model (optional).
    Falls back gracefully if SHAP is not installed.
    """
    try:
        import shap

        regressor = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X_test)

        # Use TreeExplainer for tree-based models, otherwise KernelExplainer sample
        if hasattr(regressor, "feature_importances_"):
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(X_transformed)
        else:
            # Sample for speed
            bg = shap.sample(X_transformed, min(100, len(X_transformed)))
            explainer = shap.KernelExplainer(regressor.predict, bg)
            shap_values = explainer.shap_values(X_transformed[:200])

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=feature_names[:X_transformed.shape[1]],
            show=False,
        )
        plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, "shap_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
    except ImportError:
        print("  SHAP not installed — skipping SHAP summary plot.")
    except Exception as e:
        print(f"  SHAP plot failed: {e}")


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------

def save_results(df_results: pd.DataFrame, best_model_name: str, best_model_info: dict):
    """Save model comparison CSV and JSON files."""
    # model_comparison.csv
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # model_metrics.json
    metrics_dict = df_results.set_index("Model").to_dict(orient="index")
    # Convert numpy types to native Python
    for model_name in metrics_dict:
        for key in metrics_dict[model_name]:
            val = metrics_dict[model_name][key]
            if isinstance(val, (np.integer,)):
                metrics_dict[model_name][key] = int(val)
            elif isinstance(val, (np.floating,)):
                metrics_dict[model_name][key] = float(val)
    json_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved: {json_path}")

    # best_model_info.json
    info_path = os.path.join(MODELS_DIR, "best_model_info.json")
    # Convert numpy types
    clean_info = {}
    for k, v in best_model_info.items():
        if isinstance(v, (np.integer,)):
            clean_info[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_info[k] = float(v)
        else:
            clean_info[k] = v
    with open(info_path, "w") as f:
        json.dump(clean_info, f, indent=2)
    print(f"  Saved: {info_path}")


def generate_all_charts(all_models: dict, df_results: pd.DataFrame,
                        X_test, y_test, feature_names: list):
    """Generate all required charts."""
    print("\n" + "=" * 60)
    print("GENERATING CHARTS")
    print("=" * 60)

    # 1. Main comparison bar chart
    plot_model_comparison_bar(df_results)

    # 2–4. Individual metric charts
    plot_metric_chart(df_results, "MAE", "MAE Comparison (minutes)", "mae_comparison.png")
    plot_metric_chart(df_results, "RMSE", "RMSE Comparison", "rmse_comparison.png")
    plot_metric_chart(df_results, "R2", "R² Comparison", "r2_comparison.png")

    # 5. Actual vs predicted for best model
    best_name = df_results.iloc[0]["Model"]
    best_model = all_models[best_name]
    plot_actual_vs_predicted(best_model, X_test, y_test, best_name)

    # 6. Residual analysis
    plot_residuals(best_model, X_test, y_test, best_name)

    # 7. Feature importance for best tree-based model
    # Find best tree-based model
    tree_models = ["Tuned LightGBM", "Tuned XGBoost", "LightGBM", "XGBoost",
                   "Tuned Random Forest", "Random Forest", "Tuned Gradient Boosting",
                   "Gradient Boosting", "Decision Tree"]
    best_tree_name = None
    for name in df_results["Model"]:
        if name in tree_models:
            best_tree_name = name
            break
    if best_tree_name:
        plot_feature_importance(all_models[best_tree_name], feature_names, best_tree_name)

    # 8. SHAP summary for the best model (if tree-based)
    if best_name in tree_models or best_tree_name:
        target_name = best_name if best_name in tree_models else best_tree_name
        plot_shap_summary(all_models[target_name], X_test, feature_names, target_name)

    print("=" * 60)
