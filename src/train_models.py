
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load the monthly course dataset
df = pd.read_csv(
    OUT / "feature_engineered_monthly.csv",
    parse_dates=["Month"]
)

# Features available at prediction time
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

targets = ["Demand", "Revenue"]

# Keep only rows with all required historical features
df = df.dropna(subset=features).copy()

# Chronological split: train through September, test October–December
train = df[df["Month"] <= "2025-09-01"].copy()
test = df[df["Month"] >= "2025-10-01"].copy()

print("Training rows:", len(train))
print("Testing rows:", len(test))

results = []

for target in targets:
    # Separate model for each target
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
    rmse = np.sqrt(mean_squared_error(test[target], predictions))

    print(f"\n{target} model")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    results.append({
        "Target": target,
        "MAE": mae,
        "RMSE": rmse
    })

    joblib.dump(model, MODEL_DIR / f"{target.lower()}_model.joblib")

    test[f"Actual_{target}"] = test[target]
    test[f"Predicted_{target}"] = predictions

# Save evaluation metrics and test predictions
pd.DataFrame(results).to_csv(
    OUT / "model_evaluation.csv", index=False
)

test.to_csv(OUT / "test_predictions.csv", index=False)

print("\nModels saved in:", MODEL_DIR)
print("Metrics saved to outputs/model_evaluation.csv")
print("Predictions saved to outputs/test_predictions.csv")
