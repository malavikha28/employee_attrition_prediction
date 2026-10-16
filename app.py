from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "employee_attrition.csv"

st.set_page_config(page_title="Employee Attrition | HR Analytics", page_icon="📊", layout="wide")
st.title("📊 Employee Attrition Prediction")
st.caption("HR analytics dashboard • Explore employee trends and test a machine-learning prediction")
st.info("Educational prototype using synthetic employee data. Predictions are estimates, not facts, and should never be used alone for employment decisions.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().dropna(subset=["Attrition"]).copy()
    df["AttritionFlag"] = df["Attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=["AttritionFlag"])
    df["AttritionFlag"] = df["AttritionFlag"].astype(int)
    return df

@st.cache_resource
def train_model(data):
    feature_cols = [c for c in data.columns if c not in ["EmployeeID", "Attrition", "AttritionFlag"]]
    X = data[feature_cols]
    y = data["AttritionFlag"]
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=3))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1 score": f1_score(y_test, pred, zero_division=0)}
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_test, prob)
    except ValueError:
        metrics["ROC-AUC"] = float("nan")
    return model, feature_cols, metrics

df = load_data()
model, feature_cols, metrics = train_model(df)

with st.sidebar:
    st.header("Dashboard filters")
    roles = sorted(df["JobRole"].dropna().unique().tolist())
    chosen_roles = st.multiselect("Job role", roles, default=roles)
    attrition_options = ["No", "Yes"]
    chosen_attrition = st.multiselect("Attrition status", attrition_options, default=attrition_options)
    min_tenure, max_tenure = int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())
    tenure_range = st.slider("Years at company", min_tenure, max_tenure, (min_tenure, max_tenure))
    st.caption("Filters affect analytics charts. Prediction uses the employee details entered on the Prediction tab.")

filtered = df[df["JobRole"].isin(chosen_roles) & df["Attrition"].isin(chosen_attrition) & df["YearsAtCompany"].between(tenure_range[0], tenure_range[1])].copy()

analytics_tab, predict_tab, model_tab, data_tab = st.tabs(["📈 HR Analytics", "🧑‍💼 Predict an Employee", "🤖 Model Performance", "🗂️ Dataset"])

with analytics_tab:
    if filtered.empty:
        st.warning("No records match these filters. Please change the sidebar selections.")
    else:
        rate = filtered["AttritionFlag"].mean() * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Employees in view", f"{len(filtered):,}")
        c2.metric("Attrition rate", f"{rate:.1f}%")
        c3.metric("Average monthly income", f"{filtered['MonthlyIncome'].mean():,.0f}")
        c4.metric("Average tenure", f"{filtered['YearsAtCompany'].mean():.1f} years")

        left, right = st.columns(2)
        with left:
            counts = filtered["Attrition"].value_counts().reindex(["No", "Yes"], fill_value=0).rename_axis("Attrition").reset_index(name="Employees")
            st.plotly_chart(px.bar(counts, x="Attrition", y="Employees", color="Attrition", title="Employees who stayed vs left", text="Employees"), use_container_width=True)
        with right:
            role_rate = filtered.groupby("JobRole", as_index=False)["AttritionFlag"].mean()
            role_rate["Attrition rate (%)"] = role_rate["AttritionFlag"] * 100
            st.plotly_chart(px.bar(role_rate.sort_values("Attrition rate (%)"), x="Attrition rate (%)", y="JobRole", orientation="h", title="Attrition rate by job role", text=role_rate.sort_values("Attrition rate (%)")["Attrition rate (%)"].round(1)), use_container_width=True)

        left, right = st.columns(2)
        with left:
            tenure = filtered.groupby("YearsAtCompany", as_index=False)["AttritionFlag"].mean()
            tenure["Attrition rate (%)"] = tenure["AttritionFlag"] * 100
            st.plotly_chart(px.line(tenure, x="YearsAtCompany", y="Attrition rate (%)", markers=True, title="Tenure vs attrition rate"), use_container_width=True)
        with right:
            st.plotly_chart(px.scatter(filtered, x="YearsAtCompany", y="MonthlyIncome", color="Attrition", hover_data=["JobRole", "Age", "JobSatisfaction"], title="Salary vs tenure (scatter plot)"), use_container_width=True)

        left, right = st.columns(2)
        with left:
            sat = filtered.groupby("JobSatisfaction", as_index=False)["AttritionFlag"].mean()
            sat["Attrition rate (%)"] = sat["AttritionFlag"] * 100
            st.plotly_chart(px.bar(sat, x="JobSatisfaction", y="Attrition rate (%)", title="Job satisfaction vs attrition", text=sat["Attrition rate (%)"].round(1)), use_container_width=True)
        with right:
            ot = filtered.groupby("OverTime", as_index=False)["AttritionFlag"].mean()
            ot["Attrition rate (%)"] = ot["AttritionFlag"] * 100
            st.plotly_chart(px.bar(ot, x="OverTime", y="Attrition rate (%)", title="Attrition rate by overtime", text=ot["Attrition rate (%)"].round(1)), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.histogram(filtered, x="MonthlyIncome", color="Attrition", marginal="box", barmode="overlay", title="Monthly income distribution"), use_container_width=True)
        with right:
            travel = filtered.groupby("BusinessTravel", as_index=False)["AttritionFlag"].mean()
            travel["Attrition rate (%)"] = travel["AttritionFlag"] * 100
            st.plotly_chart(px.bar(travel, x="BusinessTravel", y="Attrition rate (%)", title="Business travel vs attrition", text=travel["Attrition rate (%)"].round(1)), use_container_width=True)

        numeric_cols = ["Age", "MonthlyIncome", "YearsAtCompany", "YearsInRole", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "DistanceFromHome", "YearsSinceLastPromotion", "TrainingTimesLastYear", "PerformanceRating", "AttritionFlag"]
        corr = filtered[numeric_cols].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric feature correlation heatmap"), use_container_width=True)

with predict_tab:
    st.subheader("Enter employee details")
    st.write("Complete the fields and select **Predict attrition**. The model was trained on the included sample dataset.")
    # Defaults are reasonable sample values; they are not a real employee record.
    with st.form("prediction_form"):
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
            job_role = st.selectbox("Job role", sorted(df["JobRole"].dropna().unique()))
            income = st.number_input("Monthly income", min_value=0, max_value=1000000, value=int(df["MonthlyIncome"].median()), step=100)
            tenure = st.number_input("Years at company", min_value=0, max_value=50, value=3, step=1)
            years_role = st.number_input("Years in current role", min_value=0, max_value=40, value=2, step=1)
        with b:
            job_sat = st.select_slider("Job satisfaction (1 low – 4 high)", options=[1,2,3,4], value=3)
            env_sat = st.select_slider("Environment satisfaction (1–4)", options=[1,2,3,4], value=3)
            worklife = st.select_slider("Work-life balance (1–4)", options=[1,2,3,4], value=3)
            overtime = st.selectbox("Works overtime?", ["No", "Yes"])
            distance = st.number_input("Distance from home (km / dataset units)", min_value=0, max_value=100, value=10, step=1)
        with c:
            promotion = st.number_input("Years since last promotion", min_value=0, max_value=40, value=1, step=1)
            training = st.number_input("Training times last year", min_value=0, max_value=20, value=3, step=1)
            performance = st.select_slider("Performance rating", options=[1,2,3,4,5], value=3)
            travel = st.selectbox("Business travel", sorted(df["BusinessTravel"].dropna().unique()))
        submitted = st.form_submit_button("🔍 Predict attrition", type="primary", use_container_width=True)
    if submitted:
        record = {
            "Age": age, "JobRole": job_role, "MonthlyIncome": income,
            "YearsAtCompany": tenure, "YearsInRole": years_role,
            "JobSatisfaction": job_sat, "EnvironmentSatisfaction": env_sat,
            "WorkLifeBalance": worklife, "OverTime": overtime,
            "DistanceFromHome": distance, "YearsSinceLastPromotion": promotion,
            "TrainingTimesLastYear": training, "PerformanceRating": performance,
            "BusinessTravel": travel
        }
        sample = pd.DataFrame([{col: record[col] for col in feature_cols}])
        probability = float(model.predict_proba(sample)[0, 1])
        predicted = int(model.predict(sample)[0])
        st.markdown("### Prediction result")
        p1, p2 = st.columns([1,2])
        with p1:
            st.metric("Model-estimated attrition probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        with p2:
            if probability >= 0.66:
                st.warning("Higher model score: consider a confidential, supportive human review of workload, satisfaction, and employee feedback.")
            elif probability >= 0.33:
                st.info("Moderate model score: consider checking in with the employee and reviewing relevant workplace factors.")
            else:
                st.success("Lower model score: the model estimates a lower likelihood of attrition for this profile.")
            st.caption(f"Classifier label: {'Attrition predicted' if predicted == 1 else 'No attrition predicted'}. This is a model estimate, not a certainty.")
        st.download_button("Download this prediction as CSV", pd.DataFrame([{**record, "PredictedAttrition": "Yes" if predicted else "No", "AttritionProbability": round(probability, 4)}]).to_csv(index=False).encode("utf-8"), file_name="employee_attrition_prediction.csv", mime="text/csv")

with model_tab:
    st.subheader("Model evaluation")
    st.caption("Metrics use a stratified 80/20 split of the included synthetic dataset. Results can change with a different dataset.")
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}" if pd.notna(value) else "N/A")
    test_data = df[feature_cols + ["AttritionFlag"]]
    X_train, X_test, y_train, y_test = train_test_split(test_data[feature_cols], test_data["AttritionFlag"], test_size=0.2, random_state=42, stratify=test_data["AttritionFlag"])
    pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix, roc_curve
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["Actual: Stayed", "Actual: Left"], columns=["Predicted: Stayed", "Predicted: Left"])
    st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Confusion matrix"), use_container_width=True)
    prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Random Forest"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line={"dash":"dash"}))
    roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(roc_fig, use_container_width=True)
    st.warning("A high score is not a guarantee. Validate the model on representative, authorized company data before any real-world use.")

with data_tab:
    st.subheader("Employee dataset")
    st.write(f"Rows: **{len(df):,}** · Columns: **{df.shape[1]}**")
    st.dataframe(df.drop(columns=["AttritionFlag"], errors="ignore"), use_container_width=True, height=400)
    st.download_button("Download dataset CSV", df.drop(columns=["AttritionFlag"], errors="ignore").to_csv(index=False).encode("utf-8"), file_name="employee_attrition.csv", mime="text/csv")
    st.caption("The supplied dataset is synthetic and intended for learning/demo purposes.")

st.divider()
st.caption("Employee Attrition Prediction • Python · pandas · NumPy · scikit-learn · Plotly · Streamlit")
