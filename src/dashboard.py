 
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- PAGE CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(
    page_title="EduPro Analytics",
    page_icon="📚",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    courses = pd.read_csv(OUT / "course_summary.csv")
    categories = pd.read_csv(OUT / "category_summary.csv")
    monthly = pd.read_csv(OUT / "monthly_overall_summary.csv")
    details = pd.read_csv(OUT / "courses_clean.csv")
    evaluation_path = OUT / "model_evaluation.csv"

    if evaluation_path.exists():
        evaluation = pd.read_csv(evaluation_path)
    else:
        evaluation = pd.DataFrame()

    monthly["Month"] = pd.to_datetime(monthly["Month"])

    return courses, categories, monthly, details, evaluation

courses, categories, monthly, details, evaluation = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 EduPro")
st.sidebar.caption("Course Demand & Revenue Forecasting")

course_types = ["All"] + sorted(
    courses["CourseCategory"].dropna().unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Select course category",
    course_types
)

filtered_courses = courses.copy()
filtered_categories = categories.copy()

if selected_category != "All":
    filtered_courses = courses[
        courses["CourseCategory"] == selected_category
    ]
    filtered_categories = categories[
        categories["CourseCategory"] == selected_category
    ]

# ---------------- HEADER ----------------
st.title("📊 EduPro Analytics Dashboard")
st.write(
    "Explore course demand, recorded revenue, category performance, "
    "and preliminary machine learning evaluation."
)

st.info(
    "Historical analysis uses 2025 transaction data. "
    "Model evaluation is preliminary and is not a guaranteed future forecast."
)

# ---------------- KEY METRICS ----------------
total_demand = int(filtered_courses["Demand"].sum())
total_revenue = float(filtered_courses["Revenue"].sum())
number_courses = int(filtered_courses["CourseID"].nunique())

c1, c2, c3 = st.columns(3)

c1.metric("Recorded Transactions", f"{total_demand:,}")
c2.metric("Recorded Revenue", f"₹{total_revenue:,.2f}")
c3.metric("Courses", f"{number_courses:,}")

st.divider()

# ---------------- MONTHLY TRENDS ----------------
st.subheader("📈 Monthly Demand and Revenue")

left, right = st.columns(2)

with left:
    fig_demand = px.line(
        monthly,
        x="Month",
        y="Demand",
        markers=True,
        title="Monthly Course Demand"
    )
    fig_demand.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Transactions"
    )
    st.plotly_chart(fig_demand, width="stretch")

with right:
    fig_revenue = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Recorded Revenue"
    )
    fig_revenue.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue"
    )
    st.plotly_chart(fig_revenue, width="stretch")

# ---------------- CATEGORY ANALYSIS ----------------
st.subheader("🏆 Category Performance")

if not filtered_categories.empty:
    fig_category = px.bar(
        filtered_categories.sort_values("Revenue", ascending=True),
        x="Revenue",
        y="CourseCategory",
        orientation="h",
        title="Revenue by Course Category",
        hover_data=["Transactions", "UniqueLearners"]
    )
    st.plotly_chart(fig_category, width="stretch")

# ---------------- COURSE ANALYSIS ----------------
st.subheader("🎯 Course Performance")

sort_option = st.radio(
    "Rank courses by",
    ["Demand", "Revenue"],
    horizontal=True
)

top_courses = filtered_courses.nlargest(10, sort_option)

fig_courses = px.bar(
    top_courses.sort_values(sort_option, ascending=True),
    x=sort_option,
    y="CourseName",
    orientation="h",
    color=sort_option,
    title=f"Top 10 Courses by {sort_option}"
)
st.plotly_chart(fig_courses, width="stretch")

st.dataframe(
    filtered_courses.sort_values(sort_option, ascending=False),
    width="stretch",
    hide_index=True
)

# ---------------- FREE VS PAID ----------------
st.subheader("💳 Free and Paid Courses")

type_summary = (
    details.groupby("CourseType")
    .agg(
        NumberOfCourses=("CourseID", "nunique"),
        AveragePrice=("CoursePrice", "mean")
    )
    .reset_index()
)

left, right = st.columns(2)

with left:
    fig_types = px.pie(
        type_summary,
        names="CourseType",
        values="NumberOfCourses",
        title="Course Type Distribution",
        hole=0.45
    )
    st.plotly_chart(fig_types, width="stretch")

with right:
    st.dataframe(type_summary, width="stretch", hide_index=True)



# ---------------- SEQUENTIAL MODEL EVALUATION ----------------
st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
st.subheader("🤖 Sequential Forecasting Evaluation")

sequential_path = OUT / "sequential_evaluation.csv"

if sequential_path.exists():
    sequential = pd.read_csv(sequential_path)
    sequential["ForecastMonth"] = pd.to_datetime(
        sequential["ForecastMonth"]
    )

    st.caption(
        "Rolling one-month-ahead evaluation for October–December 2025. "
        "Actual prior-month results are assumed available before each forecast."
    )

    metric_choice = st.selectbox(
        "Select evaluation metric",
        ["MAE", "RMSE"],
        key="sequential_metric"
    )

    demand_results = sequential[
        sequential["Target"] == "Demand"
    ].sort_values("ForecastMonth")

    revenue_results = sequential[
        sequential["Target"] == "Revenue"
    ].sort_values("ForecastMonth")

    left, right = st.columns(2)

    with left:
        fig_demand_error = px.line(
            demand_results,
            x="ForecastMonth",
            y=metric_choice,
            markers=True,
            title=f"Demand {metric_choice} (transactions)"
        )
        st.plotly_chart(
            fig_demand_error,
            width="stretch"
        )

    with right:
        fig_revenue_error = px.line(
            revenue_results,
            x="ForecastMonth",
            y=metric_choice,
            markers=True,
            title=f"Revenue {metric_choice} (currency units)"
        )
        st.plotly_chart(
            fig_revenue_error,
            width="stretch"
        )

    st.dataframe(
        sequential,
        width="stretch",
        hide_index=True
    )
else:
    st.warning(
        "Run src/sequential_evaluation.py to generate evaluation results."
    )



# ---------------- FUTURE FORECASTING ----------------
st.subheader("🔮 Future Demand & Revenue Forecast")

forecast_path = OUT / "future_forecasts.csv"

if forecast_path.exists():
    forecasts = pd.read_csv(forecast_path)
    
    # Add each course's category to the forecast data
    category_lookup = (
        courses[["CourseID", "CourseCategory"]]
        .drop_duplicates(subset=["CourseID"])
        .copy()
    )

    forecasts["CourseID"] = forecasts["CourseID"].astype(str)
    category_lookup["CourseID"] = (
        category_lookup["CourseID"].astype(str)
    )

    forecasts = forecasts.merge(
        category_lookup,
        on="CourseID",
        how="left"
    )

    # Apply the sidebar category selection
    if selected_category != "All":
        forecasts = forecasts[
            forecasts["CourseCategory"] == selected_category
        ].copy()

    forecasts["ForecastMonth"] = pd.to_datetime(
        forecasts["ForecastMonth"]
    )

    st.caption(
        "Predicted course demand and revenue for the next "
        "three months, generated by the trained ML models."
    )

    forecast_summary = (
        forecasts.groupby("ForecastMonth", as_index=False)
        .agg(
            TotalPredictedDemand=("PredictedDemand", "sum"),
            TotalPredictedRevenue=("PredictedRevenue", "sum")
        )
    )

    left, right = st.columns(2)


# ---------------- FORECAST KPI CARDS ----------------
st.subheader("📌 Forecast Summary")

if not forecast_summary.empty:
    first_month = forecast_summary.iloc[0]

    next_demand = int(first_month["TotalPredictedDemand"])
    next_revenue = float(first_month["TotalPredictedRevenue"])
    three_month_revenue = float(
        forecast_summary["TotalPredictedRevenue"].sum()
    )

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Next Month's Demand",
        f"{next_demand:,} transactions"
    )

    k2.metric(
        "Next Month's Revenue",
        f"₹{next_revenue:,.2f}"
    )

    k3.metric(
        "Three-Month Revenue",
        f"₹{three_month_revenue:,.2f}"
    )

    with left:
        fig_future_demand = px.line(
            forecast_summary,
            x="ForecastMonth",
            y="TotalPredictedDemand",
            markers=True,
            title="Forecasted Monthly Demand"
        )
        
        fig_future_demand.update_layout(
            xaxis_title="Forecast Month",
            yaxis_title="Predicted Transactions",
            hovermode="x unified"
        )

        fig_future_demand.update_traces(
            line=dict(width=3),
            marker=dict(size=9),
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Predicted Demand: %{y:.0f}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(
            fig_future_demand,
            width="stretch"
        )

    with right:
        fig_future_revenue = px.line(
            forecast_summary,
            x="ForecastMonth",
            y="TotalPredictedRevenue",
            markers=True,
            title="Forecasted Monthly Revenue"
        )
        
        fig_future_revenue.update_layout(
            xaxis_title="Forecast Month",
            yaxis_title="Predicted Revenue (₹)",
            hovermode="x unified"
        )

        fig_future_revenue.update_traces(
            line=dict(width=3),
            marker=dict(size=9),
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Predicted Revenue: ₹%{y:,.2f}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(
            fig_future_revenue,
            width="stretch"
        )


# ---------------- INDIVIDUAL COURSE FORECAST ----------------
    st.subheader("🎯 Explore an Individual Course")

    course_options = (
        forecasts[["CourseID", "CourseName"]]
        .drop_duplicates()
        .sort_values("CourseName")
    )

    if not course_options.empty:
        selected_course_name = st.selectbox(
            "Select a course to explore",
            course_options["CourseName"].tolist(),
            key="future_course_selector"
        )

        selected_forecast = forecasts[
            forecasts["CourseName"] == selected_course_name
        ].sort_values("ForecastMonth")

        course_k1, course_k2 = st.columns(2)

        course_k1.metric(
            "Total Predicted Demand",
            f"{int(selected_forecast['PredictedDemand'].sum()):,}"
        )

        course_k2.metric(
            "Total Predicted Revenue",
            f"₹{selected_forecast['PredictedRevenue'].sum():,.2f}"
        )

        course_chart_left, course_chart_right = st.columns(2)

        with course_chart_left:
            fig_course_demand = px.line(
                selected_forecast,
                x="ForecastMonth",
                y="PredictedDemand",
                markers=True,
                title=f"Predicted Demand — {selected_course_name}"
            )
            st.plotly_chart(
                fig_course_demand,
                width="stretch"
            )

        with course_chart_right:
            fig_course_revenue = px.line(
                selected_forecast,
                x="ForecastMonth",
                y="PredictedRevenue",
                markers=True,
                title=f"Predicted Revenue — {selected_course_name}"
            )
            fig_course_revenue.update_layout(
                yaxis_title="Predicted Revenue (₹)"
            )
            st.plotly_chart(
                fig_course_revenue,
                width="stretch"
            )
    else:
        st.info("No forecasts available for the selected category.")
        
    st.subheader("Course-Level Forecast Details")

    # Display only the most useful forecast columns
    display_columns = [
        "ForecastMonth",
        "CourseID",
        "CourseName",
        "CourseCategory",
        "PredictedDemand",
        "PredictedRevenue"
    ]

    available_columns = [
        col for col in display_columns
        if col in forecasts.columns
    ]

    
    st.dataframe(
        forecasts[available_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "ForecastMonth": st.column_config.DateColumn(
                "Forecast Month",
                format="MMM YYYY"
            ),
            "PredictedDemand": st.column_config.NumberColumn(
                "Predicted Demand",
                format="%d"
            ),
            "PredictedRevenue": st.column_config.NumberColumn(
                "Predicted Revenue (₹)",
                format="₹%.2f"
            )
        }
    )



    forecast_csv = forecasts.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Future Forecasts CSV",
        data=forecast_csv,
        file_name="future_forecasts.csv",
        mime="text/csv"
    )

else:
    st.warning(
        "Future forecast file not found. "
        "Run: python src/future_forecasting.py"
    )

st.divider()

# ---------------- DOWNLOAD ----------------
st.subheader("📥 Download Data")

csv_data = filtered_courses.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download course analysis CSV",
    data=csv_data,
    file_name="edupro_course_analysis.csv",
    mime="text/csv"
)

st.caption("EduPro | Course Demand and Revenue Forecasting Project")
