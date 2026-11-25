# Food Delivery Time Prediction — ML Pipeline

A complete, reproducible machine learning pipeline that predicts food delivery time in minutes using multiple regression algorithms.

## Project Overview

This project trains and compares **11 regression models** (9 baseline + 2 tuned) on a food delivery dataset containing ~45,000 orders from India. The pipeline covers data cleaning, feature engineering, model training, hyperparameter tuning, evaluation, and model serialization.

### Models Implemented

| # | Model | Type |
|---|-------|------|
| 1 | Dummy Regressor (Mean) | Baseline |
| 2 | Linear Regression | Baseline |
| 3 | Ridge Regression | Baseline |
| 4 | Lasso Regression | Baseline |
| 5 | KNN (Manhattan distance) | Baseline |
| 6 | Decision Tree | Baseline |
| 7 | Random Forest | Baseline |
| 8 | Gradient Boosting | Baseline |
| 9 | XGBoost | Baseline |
| 10 | LightGBM | Baseline |
| 11 | Tuned KNN | Tuned |
| 12 | Tuned Random Forest | Tuned |
| 13 | Tuned Gradient Boosting | Tuned |
| 14 | Tuned XGBoost | Tuned |
| 15 | Tuned LightGBM | Tuned |

### Key Features Engineered

- **Haversine distance** (great-circle distance in km)
- **Manhattan distance** (coordinate-based city-block distance in km)
- **Latitude/longitude differences**
- **Order hour, minute, day of week, weekend indicator, month**
- **Preparation time** (minutes between order placement and pickup)
- **Time period** (morning/afternoon/evening/night)
- Delivery person age, ratings, vehicle condition
- Weather, traffic, festival, city classification

## Installation

```bash
# Navigate to the project directory
cd food_delivery_ml

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- lightgbm, xgboost
- joblib, shap, optuna

## Usage

### Run the Full Pipeline

```bash
cd food_delivery_ml
python main.py
```

This will:
1. Clean and preprocess the dataset
2. Engineer features
3. Split data 80:20 (train/test)
4. Train all baseline models
5. Tune hyperparameters (RandomizedSearchCV, 3-fold CV)
6. Evaluate all models on the test set
7. Generate comparison charts and reports
8. Save the best model

### Make Predictions

```python
from src.prediction import predict_delivery_time

sample = {
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.5,
    "Restaurant_latitude": 22.745,
    "Restaurant_longitude": 75.892,
    "Delivery_location_latitude": 22.765,
    "Delivery_location_longitude": 75.912,
    "Order_Date": "19-03-2022",
    "Time_Orderd": "11:30:00",
    "Time_Order_picked": "11:45:00",
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "High",
    "Vehicle_condition": 2,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 0,
    "Festival": "No",
    "City": "Urban",
}

result = predict_delivery_time(sample)
print(f"Predicted delivery time: {result[0]:.1f} minutes")
```

## Project Structure

```
food_delivery_ml/
├── dataset/
│   └── delivery_data.csv          # Training data (45,593 rows)
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA notebook
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py           # Data loading, parsing, cleaning
│   ├── feature_engineering.py     # Haversine, Manhattan, time features
│   ├── preprocessing.py           # ColumnTransformer, imputation, scaling
│   ├── train_models.py            # Baseline model definitions
│   ├── hyperparameter_tuning.py   # RandomizedSearchCV for top models
│   ├── evaluation.py              # Metrics, charts, SHAP, reports
│   └── prediction.py              # Clean prediction function
├── models/
│   ├── best_delivery_model.pkl    # Saved best pipeline
│   ├── preprocessing_pipeline.pkl # Saved preprocessor
│   ├── model_metrics.json         # All model metrics
│   └── best_model_info.json       # Best model summary
├── reports/
│   ├── model_comparison.csv       # Full comparison table
│   ├── feature_importance.csv     # Feature importance rankings
│   └── charts/
│       ├── model_comparison_bar.png
│       ├── mae_comparison.png
│       ├── rmse_comparison.png
│       ├── r2_comparison.png
│       ├── actual_vs_predicted.png
│       ├── residual_analysis.png
│       ├── feature_importance.png
│       └── shap_summary.png
├── main.py                        # Pipeline orchestrator
├── requirements.txt
└── README.md
```

## Evaluation Metrics

Models are evaluated on exactly the same holdout test set using:

- **MAE** (Mean Absolute Error) — primary metric (in minutes)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Data Leakage Prevention

- Target variable (`Time_taken(min)`) is never used as an input feature
- Preprocessing (imputation, scaling, encoding) is fitted only on training data
- Hyperparameter tuning uses cross-validation on training data only
- Test set is held out until final evaluation
- No future-aware features (all features are available at prediction time)

## Dataset Notes

- **Source:** Indian food delivery platform, March–April 2022
- **Target:** `Time_taken(min)` — parsed from format `(min) 24` → integer 24
- **Known issues:**
  - ~3,640 rows with zero/negative geographic coordinates (flagged, distances set to NaN)
  - Categorical NaNs encoded as literal string `"NaN"` (cleaned during preprocessing)
  - Trailing whitespace in string columns (stripped)
  - `Weatherconditions` has prefix `"conditions "` (removed)

## License

This project is for educational and research purposes.
