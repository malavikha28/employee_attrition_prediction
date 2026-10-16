# Employee Attrition Prediction — Python Data Science Mini Project

**Project owner:** Malavikha G  
**Domain:** Human Resources Analytics  
**Core libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## What this project does
- Loads an HR employee dataset from CSV.
- Cleans and summarizes the data.
- Explores attrition by job role, satisfaction, overtime, tenure and salary.
- Produces a collection of analytics charts (bar, line, scatter, histogram, box, heatmap, pie and count plots).
- Trains a classification model to estimate whether an employee may leave.
- Reports accuracy, precision, recall, F1 score, confusion matrix and ROC-AUC where available.
- Saves predictions and a text model report into `outputs/`.

## Important dataset note
`data/employee_attrition.csv` is a **synthetic example dataset created for this project**. It is not real company employee data and should not be described as a real-world collected dataset. The relationships are illustrative. For a real deployment, use an authorized HR dataset with consent, access controls, privacy protections, validation and human review.

## Setup (Windows PowerShell)
1. Install Python 3.10 or later.
2. Extract the ZIP.
3. Open PowerShell in the extracted `employee_attrition_prediction` folder.
4. Create and activate a virtual environment (recommended):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install packages:

```powershell
py -m pip install -r requirements.txt
```

6. Run the project:

```powershell
py main.py
```

Charts are saved in `outputs/graphs/`, employee predictions in `outputs/employee_attrition_predictions.csv`, and model metrics in `outputs/model_report.txt`.

If PowerShell blocks environment activation, you can skip activation and run:
`py -m pip install -r requirements.txt` and then `py main.py`.

## Project flow
CSV dataset → pandas/NumPy cleaning and analysis → visualizations → preprocessing → train/test split → Random Forest model → evaluation → prediction CSV/report.

## Suggested real-time application
This is a **prototype for HR decision support**, not an automated decision maker. An HR analyst could upload a current employee CSV, review aggregate patterns and use the model output to prioritize voluntary, confidential check-ins. Never use predictions alone for firing, promotion, compensation or other consequential employment decisions. Validate fairness and performance across groups, and consult HR/legal/privacy teams before deployment.

## Dataset columns
- `EmployeeID`: synthetic employee identifier
- `Age`: age in years
- `JobRole`: job category
- `MonthlyIncome`: monthly income in dataset currency units
- `YearsAtCompany`: company tenure
- `YearsInRole`: tenure in current role
- `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`: ordinal survey scores (1–4)
- `OverTime`: overtime indicator
- `DistanceFromHome`: illustrative distance value (1–40)
- `YearsSinceLastPromotion`: years since promotion
- `TrainingTimesLastYear`: training count
- `PerformanceRating`: rating (2–4)
- `BusinessTravel`: travel frequency
- `Attrition`: target label (`Yes` or `No`)

## Files
- `main.py`: complete analysis and modeling pipeline
- `data/employee_attrition.csv`: example dataset
- `requirements.txt`: dependencies
- `outputs/`: results are written here when you run the program
- `src/`: reserved for future modularization


## Display charts automatically
The updated `main.py` uses Matplotlib's interactive backend. When you run `py main.py`, each graph opens automatically in a plot window. Close each graph window to continue to the next chart. PNG copies are also saved in `outputs/graphs/` as a backup; you do not need to open that folder to view them.


## NEW: Interactive Streamlit dashboard

This version adds `app.py`, an interactive browser-based dashboard while keeping the original `main.py` automatic Matplotlib graph program.

### Start the dashboard on Windows
1. Extract the ZIP file.
2. Open the project folder.
3. Double-click `RUN_DASHBOARD.bat`, or run these commands in PowerShell from the project folder:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

4. Streamlit opens the dashboard in your browser (usually at `http://localhost:8501`). If it does not open automatically, copy the Local URL shown in the terminal into your browser.

### Dashboard tabs
- **HR Analytics:** interactive charts, filters by role/attrition/tenure, line chart, scatter plot, salary distribution, satisfaction analysis, and correlation heatmap.
- **Predict an Employee:** enter employee details and receive a model-estimated attrition probability; download the result as CSV.
- **Model Performance:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve.
- **Dataset:** view and download the included CSV.

### Original automatic graph program
To continue running the original desktop graphs, use `py main.py`. It displays the Matplotlib charts one at a time and saves backup images in `outputs/graphs/`.

**Data notice:** The included 1,200-row dataset is synthetic sample data, not real company records. Predictions are educational estimates and must not be used alone to make employment decisions. Use only authorized, appropriately anonymized company data for real-world analysis.
