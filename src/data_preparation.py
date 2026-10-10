"""
EduPro: Predictive Modeling for Course Demand and Revenue Forecasting
Step 1 — Load, validate, clean, integrate, and explore the workbook.

Expected input:
    data/EduPro Online Platform.xlsx

Run from the project root:
    python src/data_preparation.py
"""
from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "EduPro Online Platform.xlsx"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

REQUIRED_COLUMNS = {
    "Users": {"UserID", "Age", "Gender"},
    "Teachers": {"TeacherID", "Age", "Gender", "Expertise",
                 "YearsOfExperience", "TeacherRating"},
    "Courses": {"CourseID", "CourseName", "CourseCategory", "CourseType",
                "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"},
    "Transactions": {"TransactionID", "UserID", "CourseID", "TransactionDate",
                     "Amount", "PaymentMethod", "TeacherID"},
}

def require_columns(name, df):
    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

def clean_id_column(df, column):
    if column in df.columns:
        df[column] = df[column].astype("string").str.strip()

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Workbook not found: {DATA_PATH}\n"
            "Copy 'EduPro Online Platform.xlsx' into the project's data/ folder."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all workbook sheets
    sheets = pd.read_excel(DATA_PATH, sheet_name=None)
    expected_sheets = set(REQUIRED_COLUMNS)
    missing_sheets = expected_sheets - set(sheets)
    if missing_sheets:
        raise ValueError(f"Workbook is missing sheets: {sorted(missing_sheets)}")

    users = sheets["Users"].copy()
    teachers = sheets["Teachers"].copy()
    courses = sheets["Courses"].copy()
    transactions = sheets["Transactions"].copy()

    frames = {
        "Users": users, "Teachers": teachers,
        "Courses": courses, "Transactions": transactions
    }
    for name, df in frames.items():
        require_columns(name, df)

    # 2. Standardize text and identifiers
    id_columns = {
        "Users": ["UserID"],
        "Teachers": ["TeacherID"],
        "Courses": ["CourseID"],
        "Transactions": ["TransactionID", "UserID", "CourseID", "TeacherID"],
    }
    for name, columns in id_columns.items():
        for column in columns:
            clean_id_column(frames[name], column)

    # Refresh references after in-place updates
    users, teachers, courses, transactions = (
        frames["Users"], frames["Teachers"], frames["Courses"], frames["Transactions"]
    )
    for df in [users, teachers, courses, transactions]:
        for column in df.select_dtypes(include=["object", "string"]).columns:
            df[column] = df[column].astype("string").str.strip()

    # 3. Convert types and remove exact duplicate rows
    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"], errors="coerce"
    )
    numeric_columns = {
        "Users": ["Age"],
        "Teachers": ["Age", "YearsOfExperience", "TeacherRating"],
        "Courses": ["CoursePrice", "CourseDuration", "CourseRating"],
        "Transactions": ["Amount"],
    }
    for name, columns in numeric_columns.items():
        df = frames[name]
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    duplicate_counts = {name: int(df.duplicated().sum()) for name, df in frames.items()}
    for name, df in frames.items():
        frames[name] = df.drop_duplicates().copy()
    users, teachers, courses, transactions = (
        frames["Users"], frames["Teachers"], frames["Courses"], frames["Transactions"]
    )

    # Validate essential fields. Do not delete valid zero-amount/free-course transactions.
    essential = ["TransactionID", "UserID", "CourseID", "TeacherID",
                 "TransactionDate", "Amount"]
    invalid_essential = transactions[essential].isna().any(axis=1)
    invalid_transaction_rows = int(invalid_essential.sum())
    transactions = transactions.loc[~invalid_essential].copy()

    negative_amounts = int((transactions["Amount"] < 0).sum())
    if negative_amounts:
        warnings.warn(
            f"Found {negative_amounts} negative transaction amounts; inspect these records."
        )

    # 4. Check referential integrity before merging
    unknown_users = int((~transactions["UserID"].isin(users["UserID"])).sum())
    unknown_courses = int((~transactions["CourseID"].isin(courses["CourseID"])).sum())
    unknown_teachers = int((~transactions["TeacherID"].isin(teachers["TeacherID"])).sum())

    # 5. Integrate transaction, course, user, and teacher information
    merged = transactions.merge(
        courses, on="CourseID", how="left", validate="many_to_one",
        suffixes=("", "_Course")
    )
    merged = merged.merge(
        users[["UserID", "Age", "Gender"]].rename(
            columns={"Age": "UserAge", "Gender": "UserGender"}
        ),
        on="UserID", how="left", validate="many_to_one"
    )
    merged = merged.merge(
        teachers[["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]]
        .rename(columns={"Expertise": "TeacherExpertise"}),
        on="TeacherID", how="left", validate="many_to_one"
    )

    merged["Month"] = merged["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    merged["Year"] = merged["TransactionDate"].dt.year
    merged["MonthNumber"] = merged["TransactionDate"].dt.month

    # Save clean and integrated data
    users.to_csv(OUTPUT_DIR / "users_clean.csv", index=False)
    teachers.to_csv(OUTPUT_DIR / "teachers_clean.csv", index=False)
    courses.to_csv(OUTPUT_DIR / "courses_clean.csv", index=False)
    transactions.to_csv(OUTPUT_DIR / "transactions_clean.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "integrated_transactions.csv", index=False)

    # 6. Build monthly course-level targets for forecasting
    monthly = (
        merged.groupby(["Month", "CourseID", "CourseName", "CourseCategory",
                        "CourseType"], dropna=False)
        .agg(
            Demand=("TransactionID", "nunique"),
            Revenue=("Amount", "sum"),
            UniqueLearners=("UserID", "nunique"),
            AverageTransactionAmount=("Amount", "mean"),
        )
        .reset_index()
        .sort_values(["Month", "CourseID"])
    )
    monthly.to_csv(OUTPUT_DIR / "monthly_course_targets.csv", index=False)

    # Include zero-transaction course-month combinations as demand=0 and revenue=0.
    # This creates a complete course-by-month panel for the observed calendar year.
    if not merged.empty:
        months = pd.date_range(
            merged["TransactionDate"].min().to_period("M").to_timestamp(),
            merged["TransactionDate"].max().to_period("M").to_timestamp(),
            freq="MS"
        )
        course_info = courses[["CourseID", "CourseName", "CourseCategory", "CourseType"]]
        panel = pd.MultiIndex.from_product(
            [months, course_info["CourseID"].tolist()],
            names=["Month", "CourseID"]
        ).to_frame(index=False)
        panel = panel.merge(course_info, on="CourseID", how="left", validate="many_to_one")
        panel = panel.merge(
            monthly[["Month", "CourseID", "Demand", "Revenue", "UniqueLearners",
                     "AverageTransactionAmount"]],
            on=["Month", "CourseID"], how="left", validate="one_to_one"
        )
        for col in ["Demand", "Revenue", "UniqueLearners"]:
            panel[col] = panel[col].fillna(0)
        # Average transaction amount is undefined when there are no transactions.
        panel.to_csv(OUTPUT_DIR / "monthly_course_panel.csv", index=False)

    # 7. EDA summary tables
    monthly_overall = (
        merged.groupby("Month")
        .agg(Demand=("TransactionID", "nunique"), Revenue=("Amount", "sum"))
        .reset_index()
        .sort_values("Month")
    )
    monthly_overall.to_csv(OUTPUT_DIR / "monthly_overall_summary.csv", index=False)

    category_summary = (
        merged.groupby("CourseCategory")
        .agg(
            Transactions=("TransactionID", "nunique"),
            Revenue=("Amount", "sum"),
            UniqueLearners=("UserID", "nunique"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    category_summary.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)

    course_summary = (
        merged.groupby(["CourseID", "CourseName", "CourseCategory"])
        .agg(
            Demand=("TransactionID", "nunique"),
            Revenue=("Amount", "sum"),
            UniqueLearners=("UserID", "nunique"),
        )
        .reset_index()
        .sort_values("Demand", ascending=False)
    )
    course_summary.to_csv(OUTPUT_DIR / "course_summary.csv", index=False)

    # 8. Generate charts using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_overall["Month"], monthly_overall["Demand"], marker="o")
    plt.title("Monthly Course Demand — EduPro (2025)")
    plt.xlabel("Month")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=45)
    save_chart("monthly_demand.png")

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_overall["Month"], monthly_overall["Revenue"], marker="o")
    plt.title("Monthly Revenue — EduPro (2025)")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    save_chart("monthly_revenue.png")

    top_categories = category_summary.head(10).sort_values("Revenue")
    plt.figure(figsize=(10, 6))
    plt.barh(top_categories["CourseCategory"], top_categories["Revenue"])
    plt.title("Revenue by Course Category")
    plt.xlabel("Revenue")
    plt.ylabel("Course Category")
    save_chart("category_revenue.png")

    top_courses = course_summary.head(10).sort_values("Demand")
    plt.figure(figsize=(10, 6))
    plt.barh(top_courses["CourseName"], top_courses["Demand"])
    plt.title("Top 10 Courses by Transaction Demand")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Course")
    save_chart("top_courses_by_demand.png")

    plt.figure(figsize=(9, 5))
    plt.hist(merged["Amount"], bins=30)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    save_chart("transaction_amount_distribution.png")

    # 9. Produce a compact quality report
    report = [
        "EDUPRO DATA PREPARATION REPORT",
        "==============================",
        f"Users rows: {len(users):,}",
        f"Teachers rows: {len(teachers):,}",
        f"Courses rows: {len(courses):,}",
        f"Transactions rows after cleaning: {len(transactions):,}",
        f"Integrated transaction rows: {len(merged):,}",
        f"Transaction date range: {merged['TransactionDate'].min()} to {merged['TransactionDate'].max()}",
        f"Missing cells by source sheet before cleaning: "
        f"{ {name: int(df.isna().sum().sum()) for name, df in sheets.items()} }",
        f"Exact duplicate rows found before removal: {duplicate_counts}",
        f"Rows removed due to missing essential transaction fields: {invalid_transaction_rows}",
        f"Transactions referencing unknown users: {unknown_users}",
        f"Transactions referencing unknown courses: {unknown_courses}",
        f"Transactions referencing unknown teachers: {unknown_teachers}",
        f"Zero-amount transactions retained: {int((merged['Amount'] == 0).sum())}",
        f"Negative transaction amounts: {negative_amounts}",
        f"Total revenue (sum of transaction Amount): {merged['Amount'].sum():,.2f}",
        f"Monthly course panel rows: {len(months) * len(courses) if not merged.empty else 0:,}",
        "",
        "Important: zero-amount transactions were retained because the workbook includes free courses.",
        "Forecasting must use chronological train/test splits to avoid temporal leakage.",
    ]
    (OUTPUT_DIR / "data_quality_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Charts saved to: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
