
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"

# Load our prepared data
courses = pd.read_csv(OUT / "course_summary.csv")
categories = pd.read_csv(OUT / "category_summary.csv")
monthly = pd.read_csv(OUT / "monthly_overall_summary.csv")

# Top 10 courses by demand
print("\nTOP 10 COURSES BY DEMAND")
print(
    courses.nlargest(10, "Demand")[
        ["CourseName", "Demand", "Revenue"]
    ].to_string(index=False)
)

# Top 10 courses by revenue
print("\nTOP 10 COURSES BY REVENUE")
print(
    courses.nlargest(10, "Revenue")[
        ["CourseName", "Demand", "Revenue"]
    ].to_string(index=False)
)

# Category performance
print("\nCATEGORY PERFORMANCE")
print(
    categories.sort_values(
        "Revenue", ascending=False
    )[["CourseCategory", "Transactions", "Revenue"]]
    .to_string(index=False)
)

# Monthly trends
monthly["Month"] = pd.to_datetime(monthly["Month"])
print("\nMONTHLY DEMAND AND REVENUE")
print(monthly.to_string(index=False))

# Create a demand vs revenue chart
FIG.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 6))
plt.scatter(courses["Demand"], courses["Revenue"])
plt.xlabel("Course Demand")
plt.ylabel("Revenue")
plt.title("Course Demand vs Revenue")
plt.tight_layout()
plt.savefig(FIG / "demand_vs_revenue.png", dpi=150)
plt.close()

print("\nEDA completed successfully!")
print("Check outputs/figures for the new chart.")


# Step 2: Investigate zero-revenue courses

course_details = pd.read_csv(OUT / "courses_clean.csv")

zero_revenue = courses[courses["Revenue"] == 0].merge(
    course_details[["CourseID", "CourseType", "CoursePrice"]],
    on="CourseID",
    how="left"
)

print("\nZERO-REVENUE COURSE INVESTIGATION")

print(
    zero_revenue[
        ["CourseName", "Demand", "CourseType", "CoursePrice"]
    ].to_string(index=False)
)

print("\nCOURSE TYPE SUMMARY")

print(
    course_details.groupby("CourseType")
    .agg(
        CourseCount=("CourseID", "count"),
        AveragePrice=("CoursePrice", "mean")
    )
    .sort_values("CourseCount", ascending=False)
    .to_string()
)


# Step 3: Analyze paid courses

course_details = pd.read_csv(OUT / "courses_clean.csv")

paid_courses = course_details[
    course_details["CoursePrice"] > 0
]

print("\nPAID COURSE ANALYSIS")

print("Number of paid courses:", len(paid_courses))

paid_analysis = courses.merge(
    paid_courses[["CourseID", "CoursePrice", "CourseType"]],
    on="CourseID",
    how="inner"
)

print("\nTOP 10 PAID COURSES BY REVENUE")

print(
    paid_analysis.nlargest(10, "Revenue")[
        ["CourseName", "CoursePrice", "Demand", "Revenue"]
    ].to_string(index=False)
)

print("\nTOP 10 PAID COURSES BY DEMAND")

print(
    paid_analysis.nlargest(10, "Demand")[
        ["CourseName", "CoursePrice", "Demand", "Revenue"]
    ].to_string(index=False)
)


# Step 4: Category demand and revenue analysis

category_analysis = (
    courses.groupby("CourseCategory")
    .agg(
        TotalDemand=("Demand", "sum"),
        TotalRevenue=("Revenue", "sum"),
        NumberOfCourses=("CourseID", "count")
    )
    .sort_values("TotalRevenue", ascending=False)
    .reset_index()
)

print("\nCATEGORY-WISE DEMAND AND REVENUE")
print(category_analysis.to_string(index=False))

# Save the results
category_analysis.to_csv(
    OUT / "category_demand_revenue_analysis.csv",
    index=False
)

# Plot the top 10 categories by revenue
top_categories = category_analysis.head(10).sort_values("TotalRevenue")

plt.figure(figsize=(10, 6))
plt.barh(
    top_categories["CourseCategory"],
    top_categories["TotalRevenue"]
)
plt.xlabel("Total Revenue")
plt.ylabel("Course Category")
plt.title("Top 10 Course Categories by Revenue")
plt.tight_layout()
plt.savefig(
    FIG / "top_10_categories_revenue.png",
    dpi=150
)
plt.close()

print("\nCategory analysis completed!")
