
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Load transaction and course data
transactions = pd.read_csv(
    OUT / "transactions_clean.csv",
    parse_dates=["TransactionDate"]
)

courses = pd.read_csv(OUT / "courses_clean.csv")

transactions["Month"] = (
    transactions["TransactionDate"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Aggregate observed monthly demand and revenue
observed = (
    transactions.groupby(["Month", "CourseID"])
    .agg(
        Demand=("TransactionID", "nunique"),
        Revenue=("Amount", "sum"),
        UniqueLearners=("UserID", "nunique")
    )
    .reset_index()
)

# Create every course-month combination
months = pd.date_range(
    transactions["Month"].min(),
    transactions["Month"].max(),
    freq="MS"
)

panel = pd.MultiIndex.from_product(
    [months, courses["CourseID"].tolist()],
    names=["Month", "CourseID"]
).to_frame(index=False)

# Add course information
panel = panel.merge(
    courses,
    on="CourseID",
    how="left",
    validate="many_to_one"
)

# Add observed monthly outcomes
panel = panel.merge(
    observed,
    on=["Month", "CourseID"],
    how="left",
    validate="one_to_one"
)

# Missing observed activity means zero recorded transactions
for col in ["Demand", "Revenue", "UniqueLearners"]:
    panel[col] = panel[col].fillna(0)

# Calendar features
panel["Year"] = panel["Month"].dt.year
panel["MonthNumber"] = panel["Month"].dt.month
panel["Quarter"] = panel["Month"].dt.quarter

# Ensure correct chronological ordering
panel = panel.sort_values(
    ["CourseID", "Month"]
).reset_index(drop=True)

# Historical features: previous calendar month
panel["PreviousMonthDemand"] = (
    panel.groupby("CourseID")["Demand"].shift(1)
)

panel["PreviousMonthRevenue"] = (
    panel.groupby("CourseID")["Revenue"].shift(1)
)

# Previous three calendar months' averages
panel["DemandMovingAverage3"] = (
    panel.groupby("CourseID")["Demand"]
    .transform(lambda s: s.shift(1).rolling(3).mean())
)

panel["RevenueMovingAverage3"] = (
    panel.groupby("CourseID")["Revenue"]
    .transform(lambda s: s.shift(1).rolling(3).mean())
)

# Save the modeling dataset
output_file = OUT / "feature_engineered_monthly.csv"
panel.to_csv(output_file, index=False)

print("Feature engineering completed successfully!")
print("Total rows:", len(panel))
print("Number of courses:", panel["CourseID"].nunique())
print("Number of months:", panel["Month"].nunique())
print("Date range:", panel["Month"].min(), "to", panel["Month"].max())

print("\nMissing values in historical features:")
print(
    panel[
        ["PreviousMonthDemand", "PreviousMonthRevenue",
         "DemandMovingAverage3", "RevenueMovingAverage3"]
    ].isna().sum()
)

print("\nFirst 10 rows:")
print(panel.head(10).to_string(index=False))
print("\nSaved to:", output_file)
