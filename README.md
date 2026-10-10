# EduPro — Course Demand & Revenue Forecasting

## Step 1: Data loading, cleaning, integration, and EDA

### Project structure
```text
EduPro_Project_Step1/
├── data/
│   └── EduPro Online Platform.xlsx   # copy your supplied workbook here
├── src/
│   └── data_preparation.py
├── outputs/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

### 1. Set up Python
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Copy `EduPro Online Platform.xlsx` into the `data` folder.

### 4. Run the preparation pipeline
From the `EduPro_Project_Step1` root folder:
```bash
python src/data_preparation.py
```

### 5. Key outputs
- `outputs/data_quality_report.txt` — data validation summary
- `outputs/integrated_transactions.csv` — joined transaction, course, user, and teacher data
- `outputs/monthly_course_targets.csv` — monthly course targets for months with observed transactions
- `outputs/monthly_course_panel.csv` — complete course-month panel including zero-transaction months
- `outputs/monthly_overall_summary.csv` — monthly demand and revenue
- `outputs/category_summary.csv` — category performance
- `outputs/course_summary.csv` — course-level demand and revenue
- `outputs/figures/` — EDA charts

## Modeling notes
- Demand is initially defined as the number of transactions per course per month.
- Revenue is defined as the sum of transaction `Amount` per course per month.
- Zero-amount transactions are retained because free courses exist in the workbook.
- The monthly course panel is a forecasting dataset, not proof that zero observed purchases mean zero real-world interest.
- Use chronological train/test splits for all forecasting experiments.
- The supplied data spans 2025; a full-year seasonal forecast is difficult to validate reliably with only one annual cycle. Model evaluation should report this limitation clearly.

## Running the Dashboard

The EduPro dashboard displays course demand, revenue analysis, model evaluation, and future forecasts.

### 1. Generate Future Forecasts

From the project root directory, run:

```bash
python src/future_forecasting.py
```

This generates the forecast results in `outputs/future_forecasts.csv`.

### 2. Launch the Streamlit Dashboard

Run:

```bash
streamlit run src/dashboard.py
```

Open the local URL displayed in the terminal to view the dashboard.

### Dashboard Features

- Course category filtering
- Monthly course demand and revenue analysis
- Course-level performance summaries
- Model evaluation results
- Future demand and revenue forecasts
- Individual course forecast selection
- Forecast results available for CSV download

### Important Notes

- Forecasts for January–March 2026 are model predictions, not verified actual results.
- The original dataset and trained model files are excluded from GitHub by default.
- To reproduce the results, provide the authorized dataset and run the required data preparation and model training scripts.