"""
Employee Attrition Prediction mini project.
Run: python main.py
Dataset: data/employee_attrition.csv (synthetic teaching dataset).
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    RocCurveDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "employee_attrition.csv"
OUT = ROOT / "outputs"
GRAPH_DIR = OUT / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save_plot(filename):
    """Save each chart and display it immediately in a separate plot window."""
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    print("=" * 68)
    print("EMPLOYEE ATTRITION PREDICTION — HR ANALYTICS")
    print("=" * 68)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Basic cleaning
    df = df.drop_duplicates().copy()
    df["MonthlyIncome"] = pd.to_numeric(df["MonthlyIncome"], errors="coerce")
    df["YearsAtCompany"] = pd.to_numeric(df["YearsAtCompany"], errors="coerce")
    df = df.dropna(subset=["Attrition"])
    df["AttritionFlag"] = (df["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Summary analysis
    overall_rate = df["AttritionFlag"].mean() * 100
    print(f"\nOverall attrition rate: {overall_rate:.2f}%")
    print("\nAttrition by job role (%):")
    print((df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False).round(2)).to_string())
    print("\nAverage monthly income by attrition:")
    print(df.groupby("Attrition")["MonthlyIncome"].mean().round(2).to_string())
    print("\nAverage tenure by attrition:")
    print(df.groupby("Attrition")["YearsAtCompany"].mean().round(2).to_string())

    # Charts 1: target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Attrition", order=["No", "Yes"], color="steelblue")
    plt.title("Employee Attrition Distribution")
    plt.xlabel("Attrition")
    plt.ylabel("Number of employees")
    save_plot("01_attrition_distribution.png")

    # Chart 2: attrition by job role
    role_rate = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values()
    plt.figure(figsize=(10, 6))
    role_rate.plot(kind="barh")
    plt.title("Attrition Rate by Job Role")
    plt.xlabel("Attrition rate (%)")
    plt.ylabel("Job role")
    save_plot("02_attrition_by_job_role.png")

    # Chart 3: line chart - attrition rate by tenure
    tenure_rate = df.groupby("YearsAtCompany")["AttritionFlag"].mean().mul(100).sort_index()
    plt.figure(figsize=(9, 5))
    plt.plot(tenure_rate.index, tenure_rate.values, marker="o")
    plt.title("Attrition Rate by Years at Company (Line Chart)")
    plt.xlabel("Years at company")
    plt.ylabel("Attrition rate (%)")
    save_plot("03_line_attrition_by_tenure.png")

    # Chart 4: scatter plot salary vs tenure, colored by attrition
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="YearsAtCompany", y="MonthlyIncome", hue="Attrition", alpha=.65)
    plt.title("Monthly Income vs Tenure (Scatter Plot)")
    plt.xlabel("Years at company")
    plt.ylabel("Monthly income")
    save_plot("04_scatter_income_tenure.png")

    # Chart 5: satisfaction vs attrition
    satisfaction = df.groupby("JobSatisfaction")["AttritionFlag"].mean().mul(100)
    plt.figure(figsize=(7, 5))
    satisfaction.plot(kind="bar")
    plt.title("Attrition Rate by Job Satisfaction")
    plt.xlabel("Job satisfaction (1 = low, 4 = high)")
    plt.ylabel("Attrition rate (%)")
    save_plot("05_satisfaction_vs_attrition.png")

    # Chart 6: overtime
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="OverTime", y="AttritionFlag", errorbar=None, color="steelblue")
    plt.title("Attrition Rate by Overtime")
    plt.xlabel("Overtime")
    plt.ylabel("Attrition rate")
    save_plot("06_overtime_vs_attrition.png")

    # Chart 7: salary distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="MonthlyIncome", hue="Attrition", kde=True, element="step")
    plt.title("Monthly Income Distribution by Attrition")
    plt.xlabel("Monthly income")
    save_plot("07_income_distribution.png")

    # Chart 8: box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Attrition", y="YearsAtCompany")
    plt.title("Tenure Distribution by Attrition")
    plt.xlabel("Attrition")
    plt.ylabel("Years at company")
    save_plot("08_tenure_boxplot.png")

    # Chart 9: correlation heatmap for numeric features
    numeric = df.select_dtypes(include=np.number).drop(columns=["AttritionFlag"], errors="ignore")
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric HR Features")
    save_plot("09_correlation_heatmap.png")

    # Chart 10: attrition by business travel
    travel_rate = df.groupby("BusinessTravel")["AttritionFlag"].mean().mul(100).sort_values()
    plt.figure(figsize=(8, 5))
    travel_rate.plot(kind="bar")
    plt.title("Attrition Rate by Business Travel")
    plt.xlabel("Business travel")
    plt.ylabel("Attrition rate (%)")
    save_plot("10_business_travel.png")

    # Modeling: exclude ID and target; use a stratified split.
    features = df.drop(columns=["EmployeeID", "Attrition", "AttritionFlag"])
    target = df["AttritionFlag"]
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
            min_samples_leaf=3
        ))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 score": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
    except ValueError:
        pass

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted No", "Predicted Yes"],
                yticklabels=["Actual No", "Actual Yes"])
    plt.title("Model Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    save_plot("11_confusion_matrix.png")

    try:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC Curve")
        save_plot("12_roc_curve.png")
    except ValueError:
        pass

    # Save test-set predictions with a human-readable risk band.
    predictions = X_test.copy()
    predictions["ActualAttrition"] = np.where(y_test.to_numpy() == 1, "Yes", "No")
    predictions["PredictedAttrition"] = np.where(y_pred == 1, "Yes", "No")
    predictions["AttritionProbability"] = np.round(y_prob, 4)
    predictions["RiskBand"] = pd.cut(
        y_prob, bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["Lower model score", "Moderate model score", "Higher model score"]
    ).astype(str)
    predictions.to_csv(OUT / "employee_attrition_predictions.csv", index=False)

    report_lines = [
        "EMPLOYEE ATTRITION PREDICTION — MODEL REPORT",
        "NOTE: The data is synthetic and metrics are educational only.",
        f"Rows used: {len(df)}",
        f"Training rows: {len(X_train)}",
        f"Test rows: {len(X_test)}",
        f"Overall attrition rate: {overall_rate:.2f}%",
        "",
        "Evaluation metrics:"
    ]
    report_lines.extend([f"{k}: {v:.4f}" for k, v in metrics.items()])
    report_lines += ["", "Classification report:", classification_report(
        y_test, y_pred, target_names=["No attrition", "Attrition"], zero_division=0
    )]
    (OUT / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nModel evaluation:")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No attrition", "Attrition"], zero_division=0))
    print(f"\nDone! Charts saved to: {GRAPH_DIR}")
    print(f"Predictions saved to: {OUT / 'employee_attrition_predictions.csv'}")
    print(f"Model report saved to: {OUT / 'model_report.txt'}")
    print("\nReminder: this is a teaching prototype using synthetic data, not a validated HR decision tool.")


if __name__ == "__main__":
    main()
