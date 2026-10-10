
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

df = pd.read_csv(
    OUT / "feature_engineered_monthly.csv",
    parse_dates=["Month"]
)

df = df.sort_values(["CourseID", "Month"]).copy()

# Use only months with a previous month's observation
df = df.dropna(subset=["PreviousMonthDemand", "PreviousMonthRevenue"])

# Hold out the final three months
train = df[df["Month"] < "2025-10-01"]
test = df[df["Month"] >= "2025-10-01"].copy()

# Naive baseline: predict the next month's value
# using the previous month's actual value.
for target, lag in [
    ("Demand", "PreviousMonthDemand"),
    ("Revenue", "PreviousMonthRevenue")
]:
    actual = test[target]
    predicted = test[lag]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n{target} — Previous-month baseline")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

print("\nTraining months:", train["Month"].nunique())
print("Testing months:", test["Month"].nunique())
print("Baseline evaluation completed.")
