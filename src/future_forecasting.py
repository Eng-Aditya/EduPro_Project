
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = ROOT / "models"

INPUT_FILE = OUT / "feature_engineered_monthly.csv"
OUTPUT_FILE = OUT / "future_forecasts.csv"

# ---------------- CONFIGURATION ----------------
FORECAST_MONTHS = 3

# ---------------- LOAD DATA ----------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Run src/feature_engineering.py first."
    )

df = pd.read_csv(INPUT_FILE, parse_dates=["Month"])
df = df.sort_values(["CourseID", "Month"]).copy()

required_columns = [
    "CourseID",
    "CourseName",
    "Month",
    "Demand",
    "Revenue",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------------- LOAD TRAINED MODELS ----------------
demand_model_path = MODELS / "demand_model.joblib"
revenue_model_path = MODELS / "revenue_model.joblib"

if not demand_model_path.exists() or not revenue_model_path.exists():
    raise FileNotFoundError(
        "Trained models were not found. "
        "Check the filenames inside the models folder."
    )

demand_model = joblib.load(demand_model_path)
revenue_model = joblib.load(revenue_model_path)

# ---------------- FEATURES ----------------
features = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "MonthNumber",
    "Quarter",
    "PreviousMonthDemand",
    "PreviousMonthRevenue",
    "DemandMovingAverage3",
    "RevenueMovingAverage3",
]

missing_features = [c for c in features if c not in df.columns]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

# ---------------- RECURSIVE FORECASTING ----------------
forecast_rows = []

for course_id, group in df.groupby("CourseID", sort=False):
    history = group.sort_values("Month").copy()

    if history.empty:
        continue

    latest = history.iloc[-1]
    course_name = latest["CourseName"]

    # Keep historical values for rolling features.
    demand_history = history["Demand"].astype(float).tolist()
    revenue_history = history["Revenue"].astype(float).tolist()

    for step in range(1, FORECAST_MONTHS + 1):
        forecast_month = (
            latest["Month"] + pd.DateOffset(months=step)
        )

        previous_demand = demand_history[-1]
        previous_revenue = revenue_history[-1]

        demand_average = float(
            np.mean(demand_history[-3:])
        )
        revenue_average = float(
            np.mean(revenue_history[-3:])
        )

        row = {
            "CoursePrice": latest["CoursePrice"],
            "CourseDuration": latest["CourseDuration"],
            "CourseRating": latest["CourseRating"],
            "MonthNumber": forecast_month.month,
            "Quarter": forecast_month.quarter,
            "PreviousMonthDemand": previous_demand,
            "PreviousMonthRevenue": previous_revenue,
            "DemandMovingAverage3": demand_average,
            "RevenueMovingAverage3": revenue_average,
        }

        X = pd.DataFrame([row], columns=features)

        predicted_demand = float(
            demand_model.predict(X)[0]
        )
        predicted_revenue = float(
            revenue_model.predict(X)[0]
        )

        # Demand represents a non-negative transaction count.
        predicted_demand = max(0, int(round(predicted_demand)))

        # Revenue is constrained to non-negative values.
        predicted_revenue = max(0.0, predicted_revenue)

        forecast_rows.append({
            "ForecastMonth": forecast_month.strftime("%Y-%m-%d"),
            "CourseID": course_id,
            "CourseName": course_name,
            "PredictedDemand": predicted_demand,
            "PredictedRevenue": round(predicted_revenue, 2),
        })

        # Use predictions as the next month's lag values.
        demand_history.append(predicted_demand)
        revenue_history.append(predicted_revenue)

# ---------------- SAVE RESULTS ----------------
if not forecast_rows:
    raise ValueError("No forecasts were generated.")

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(OUTPUT_FILE, index=False)

print("\nFuture forecasting completed!")
print(f"Forecast rows: {len(forecast_df)}")
print(f"Forecast months: {FORECAST_MONTHS}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nForecast summary by month:")
print(
    forecast_df.groupby("ForecastMonth", as_index=False)
    .agg(
        TotalPredictedDemand=("PredictedDemand", "sum"),
        TotalPredictedRevenue=("PredictedRevenue", "sum"),
    )
    .to_string(index=False)
)

print("\nFirst 10 course forecasts:")
print(forecast_df.head(10).to_string(index=False))
