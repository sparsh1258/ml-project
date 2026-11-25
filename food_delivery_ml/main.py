"""
Main Pipeline Orchestrator
============================
Runs the complete ML pipeline end-to-end:
  1. Data cleaning
  2. Feature engineering
  3. Train/test split
  4. Build preprocessing pipeline
  5. Train baseline models
  6. Hyperparameter tuning
  7. Evaluate all models
  8. Generate charts and reports
  9. Save best model

Usage:
    cd food_delivery_ml
    python main.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.preprocessing import (
    prepare_features, build_preprocessor, get_feature_names,
    TARGET, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
)
from src.train_models import get_baseline_models, train_all_models
from src.hyperparameter_tuning import run_all_tuning
from src.evaluation import (
    evaluate_all_models, generate_all_charts, save_results,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "delivery_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(RANDOM_STATE)


def main():
    print("\n" + "█" * 60)
    print("  FOOD DELIVERY TIME PREDICTION — ML PIPELINE")
    print("█" * 60)

    # ==================================================================
    # Step 1: Data Cleaning
    # ==================================================================
    df = clean_data(DATA_PATH)

    # ==================================================================
    # Step 2: Feature Engineering
    # ==================================================================
    df = engineer_features(df)

    # ==================================================================
    # Step 3: Prepare Features & Train/Test Split
    # ==================================================================
    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X, y = prepare_features(df)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # Drop rows with NaN target
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    print(f"  After dropping NaN targets: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print("=" * 60)

    # ==================================================================
    # Step 4: Build Preprocessing Pipeline
    # ==================================================================
    preprocessor = build_preprocessor()

    # ==================================================================
    # Step 5: Train Baseline Models
    # ==================================================================
    baseline_models = get_baseline_models(preprocessor)
    baseline_models = train_all_models(baseline_models, X_train, y_train)

    # Get feature names after fitting (use first fitted model's preprocessor)
    first_model = list(baseline_models.values())[0]
    fitted_preprocessor = first_model.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_preprocessor, X_train)

    # ==================================================================
    # Step 6: Hyperparameter Tuning
    # ==================================================================
    # Build a fresh preprocessor for tuning (will be re-fit inside CV)
    tuned_models = run_all_tuning(
        build_preprocessor(), X_train, y_train, cv=3,
    )

    # ==================================================================
    # Step 7: Merge all models & Evaluate
    # ==================================================================
    all_models = {**baseline_models, **tuned_models}
    df_results = evaluate_all_models(all_models, X_test, y_test)

    # ==================================================================
    # Step 8: Final Comparison Table
    # ==================================================================
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # Identify best models
    best_overall = df_results.iloc[0]
    baseline_row = df_results[df_results["Model"] == "Dummy (Mean)"]

    tuned_names = [n for n in df_results["Model"] if n.startswith("Tuned")]
    baseline_names = [n for n in df_results["Model"] if not n.startswith("Tuned") and n != "Dummy (Mean)"]

    best_baseline_df = df_results[df_results["Model"].isin(baseline_names)]
    best_tuned_df = df_results[df_results["Model"].isin(tuned_names)]

    print(f"\n  Overall best model:   {best_overall['Model']} (MAE={best_overall['MAE']:.4f})")
    if not best_baseline_df.empty:
        bb = best_baseline_df.iloc[0]
        print(f"  Best baseline model:  {bb['Model']} (MAE={bb['MAE']:.4f})")
    if not best_tuned_df.empty:
        bt = best_tuned_df.iloc[0]
        print(f"  Best tuned model:     {bt['Model']} (MAE={bt['MAE']:.4f})")
    if not baseline_row.empty:
        bl_mae = baseline_row.iloc[0]["MAE"]
        print(f"  Baseline (Dummy) MAE: {bl_mae:.4f}")
        improvement = (bl_mae - best_overall["MAE"]) / bl_mae * 100
        print(f"  Best improvement over baseline: {improvement:.2f}%")

    # ==================================================================
    # Step 9: Generate Charts
    # ==================================================================
    # Update feature names from the best model's preprocessor if tree-based
    best_name = best_overall["Model"]
    best_model = all_models[best_name]
    try:
        fn = get_feature_names(best_model.named_steps["preprocessor"], X_test)
        feature_names = fn
    except Exception:
        pass

    generate_all_charts(all_models, df_results, X_test, y_test, feature_names)

    # ==================================================================
    # Step 10: Save Best Model & Reports
    # ==================================================================
    print("\n" + "=" * 60)
    print("SAVING MODEL & REPORTS")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_delivery_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  Best model saved: {model_path}")

    # Save preprocessing pipeline separately
    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    joblib.dump(best_model.named_steps["preprocessor"], prep_path)
    print(f"  Preprocessing pipeline saved: {prep_path}")

    # Build best_model_info
    best_model_info = {
        "model_name": best_name,
        "MAE": float(best_overall["MAE"]),
        "RMSE": float(best_overall["RMSE"]),
        "R2": float(best_overall["R2"]),
        "improvement_vs_baseline_pct": float(best_overall.get("Improvement_vs_Baseline_%", 0)),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_features": int(len(feature_names)),
        "random_state": RANDOM_STATE,
    }
    save_results(df_results, best_name, best_model_info)

    # ==================================================================
    # Verify prediction function
    # ==================================================================
    print("\n" + "=" * 60)
    print("PREDICTION FUNCTION VERIFICATION")
    print("=" * 60)

    # Pick a sample from the test set and predict
    sample_row = X_test.iloc[0].to_dict()
    pred = best_model.predict(X_test.iloc[[0]])
    actual = y_test.iloc[0]
    print(f"  Sample prediction: {pred[0]:.1f} min (actual: {actual:.0f} min)")

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE!")
    print("█" * 60)
    print(f"\n  Best Model: {best_name}")
    print(f"  Test MAE:   {best_overall['MAE']:.4f} minutes")
    print(f"  Test RMSE:  {best_overall['RMSE']:.4f}")
    print(f"  Test R²:    {best_overall['R2']:.4f}")
    print(f"\n  Model saved to: {model_path}")
    print(f"  Reports in:     {os.path.abspath(os.path.join(MODELS_DIR, '..', 'reports'))}")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
