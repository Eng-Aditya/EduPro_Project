
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

df = pd.read_csv(
    OUT / "feature_engineered_monthly.csv",
    parse_dates=["Month"]
)

df = df.sort_values(["CourseID", "Month"]).copy()

# Features known before the month being predicted
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

# Evaluate October, November, and December separately.
# For each month, train only on earlier months.
results = []

for forecast_month in pd.date_range(
    "2025-10-01", "2025-12-01", freq="MS"
):
    train = df[df["Month"] < forecast_month].copy()
    test = df[df["Month"] == forecast_month].copy()

    # Historical features must be available for both datasets.
    train = train.dropna(subset=features)
    test = test.dropna(subset=features)

    if train.empty or test.empty:
        print("Skipping month:", forecast_month.date())
        continue

    for target in ["Demand", "Revenue"]:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        model.fit(train[features], train[target])
        predictions = model.predict(test[features])

        mae = mean_absolute_error(test[target], predictions)
        rmse = np.sqrt(
            mean_squared_error(test[target], predictions)
        )

        results.append({
            "ForecastMonth": forecast_month,
            "Target": target,
            "MAE": mae,
            "RMSE": rmse,
            "TrainingRows": len(train),
            "TestingRows": len(test)
        })

        print(
            f"{forecast_month:%Y-%m} | {target} | "
            f"MAE={mae:.2f} | RMSE={rmse:.2f}"
        )

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUT / "sequential_evaluation.csv",
    index=False
)

print("\nSequential evaluation saved to outputs/sequential_evaluation.csv")
 