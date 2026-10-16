
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "ai4i2020.csv"
DB_PATH = ROOT / "maintenance.db"

TARGET = "Machine failure"

NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL = ["Type"]

ENGINEERED_NUMERIC = NUMERIC + [
    "Temperature difference [K]",
    "Estimated mechanical power [W]",
    "Wear torque index",
]

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Maintenance for Machinery")
st.caption(
    "Machine Learning | Failure Risk Prediction | "
    "Anomaly Detection | Maintenance Analytics"
)


# =========================================================
# 2. DATA LOADING AND FEATURE ENGINEERING
# =========================================================

def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    required = set(NUMERIC + CATEGORICAL)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUMERIC + CATEGORICAL)

    return df.reset_index(drop=True)


def engineer_features(df):
    x = df[NUMERIC + CATEGORICAL].copy()

    x["Temperature difference [K]"] = (
        x["Process temperature [K]"]
        - x["Air temperature [K]"]
    )

    x["Estimated mechanical power [W]"] = (
        x["Torque [Nm]"]
        * x["Rotational speed [rpm]"]
        * (2 * np.pi / 60)
    )

    x["Wear torque index"] = (
        x["Tool wear [min]"] * x["Torque [Nm]"]
    )

    return x


def build_pipeline(estimator):
    preprocessing = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            ENGINEERED_NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL,
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", estimator),
    ])


# =========================================================
# 3. SQLITE DATABASE
# =========================================================

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                machine_type TEXT,
                risk REAL,
                prediction INTEGER,
                health TEXT,
                action TEXT
            )
        """)


def save_prediction(machine_type, risk, prediction, health, action):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO prediction_history
            (created_at, machine_type, risk, prediction, health, action)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            str(machine_type),
            float(risk),
            int(prediction),
            health,
            action,
        ))


def get_history():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY id DESC",
            con,
        )


init_db()


# =========================================================
# 4. MODEL TRAINING AND EVALUATION
# =========================================================

@st.cache_resource(show_spinner="Training machine learning models...")
def train_models(file_path, modified_time):
    # modified_time helps invalidate the cache when the CSV changes.
    del modified_time

    df = load_data(file_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Training CSV must contain the '{TARGET}' column."
        )

    df = df[df[TARGET].isin([0, 1])].copy()

    if df[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    y = df[TARGET].astype(int)
    x = engineer_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    models = {}
    results = {}

    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)

        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "PR-AUC": average_precision_score(y_test, probabilities),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }

        models[name] = pipeline

    # PR-AUC is useful for imbalanced failure datasets.
    best_name = max(
        results,
        key=lambda name: results[name]["PR-AUC"],
    )

    best_model = models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "failure_model.joblib",
    )

    metrics = pd.DataFrame(results).T
    metrics.index.name = "Model"
    metrics.to_csv(REPORT_DIR / "model_metrics.csv")

    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    # Anomaly detector is separate from the failure classifier.
    anomaly_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=42,
        )),
    ])

    anomaly_model.fit(x_train[ENGINEERED_NUMERIC])

    joblib.dump(
        anomaly_model,
        MODEL_DIR / "anomaly_model.joblib",
    )

    return df, best_model, anomaly_model, best_name, metrics, matrix


# =========================================================
# 5. SIDEBAR AND DATASET
# =========================================================

st.sidebar.title("⚙️ System Settings")

uploaded_training = st.sidebar.file_uploader(
    "Upload training dataset (CSV)",
    type=["csv"],
    help="Training data must include Machine failure labels.",
)

if uploaded_training is not None:
    training_bytes = uploaded_training.getvalue()
    digest = hashlib.sha256(training_bytes).hexdigest()[:16]
    training_path = DATA_DIR / f"uploaded_{digest}.csv"
    training_path.write_bytes(training_bytes)

elif DATA_PATH.exists():
    training_path = DATA_PATH

else:
    st.warning(
        "Dataset not found. Download the AI4I 2020 Predictive "
        "Maintenance Dataset and save it as data/ai4i2020.csv, "
        "or upload a labeled CSV using the sidebar."
    )
    st.markdown(
        "[Download dataset from UCI](https://archive.ics.uci.edu/"
        "dataset/601/ai4i+2020+predictive+maintenance+dataset)"
    )
    st.stop()

try:
    (
        df,
        model,
        anomaly_model,
        best_name,
        metrics,
        matrix,
    ) = train_models(
        str(training_path),
        training_path.stat().st_mtime,
    )

except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

st.sidebar.success(f"Active model: {best_name}")

threshold = st.sidebar.slider(
    "Failure alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Machine Prediction",
        "Batch Prediction",
        "Model Evaluation",
        "Maintenance History",
    ],
)


# =========================================================
# 6. RISK CLASSIFICATION
# =========================================================

def classify_health(risk, cutoff):
    if risk >= cutoff:
        return (
            "Critical",
            "Prioritize inspection and follow established safety procedures.",
        )

    if risk >= cutoff * 0.6:
        return (
            "Warning",
            "Schedule an inspection and verify the sensor readings.",
        )

    return (
        "Normal",
        "Continue routine monitoring and scheduled maintenance.",
    )


# =========================================================
# 7. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    total = len(df)
    failure_count = int(df[TARGET].sum())
    failure_rate = failure_count / max(total, 1) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{total:,}")
    c2.metric("Recorded Failures", f"{failure_count:,}")
    c3.metric("Historical Failure Rate", f"{failure_rate:.2f}%")
    c4.metric("Selected Model", best_name)

    left, right = st.columns(2)

    with left:
        st.subheader("Failure Distribution")

        distribution = pd.DataFrame({
            "Outcome": ["No Failure", "Failure"],
            "Count": [
                int((df[TARGET] == 0).sum()),
                int((df[TARGET] == 1).sum()),
            ],
        })

        fig = px.pie(
            distribution,
            names="Outcome",
            values="Count",
            hole=0.45,
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Machine Operating Conditions")

        fig = px.scatter(
            df,
            x="Torque [Nm]",
            y="Rotational speed [rpm]",
            color=TARGET,
            hover_data=["Type", "Tool wear [min]"],
            title="Torque vs Rotational Speed",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensor Statistics")
    st.dataframe(
        df[NUMERIC].describe().round(2),
        use_container_width=True,
    )

    st.info(
        "This dashboard analyzes historical dataset records. "
        "It does not receive real-time industrial sensor signals."
    )


# =========================================================
# 8. SINGLE MACHINE PREDICTION
# =========================================================

elif page == "Machine Prediction":

    st.subheader("Machine Failure Risk Prediction")

    with st.form("machine_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            machine_type = st.selectbox(
                "Machine Type",
                ["L", "M", "H"],
            )
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=290.0,
                max_value=310.0,
                value=300.0,
            )

        with col2:
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=295.0,
                max_value=320.0,
                value=310.0,
            )
            speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=500,
                max_value=4000,
                value=1500,
            )

        with col3:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=1.0,
                max_value=100.0,
                value=40.0,
            )
            wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=300,
                value=100,
            )

        submitted = st.form_submit_button(
            "Predict Failure Risk",
            type="primary",
        )

    if submitted:

        if process_temp < air_temp:
            st.warning(
                "Process temperature is below air temperature. "
                "Verify that this input is correct."
            )

        record = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }])

        features = engineer_features(record)

        risk = float(model.predict_proba(features)[0, 1])
        prediction = int(risk >= threshold)

        health, action = classify_health(risk, threshold)

        c1, c2, c3 = st.columns(3)

        c1.metric("Estimated Failure Risk", f"{risk:.1%}")
        c2.metric("Machine Health", health)
        c3.metric(
            "Failure Alert",
            "YES" if prediction else "NO",
        )

        st.progress(risk)

        st.write("**Maintenance recommendation:**", action)

        anomaly_result = anomaly_model.predict(
            features[ENGINEERED_NUMERIC]
        )[0]

        if anomaly_result == -1:
            st.warning(
                "Anomaly detected: these readings differ from "
                "typical training observations. This is not proof "
                "of an actual machine failure."
            )
        else:
            st.info("No anomaly flagged by the anomaly detector.")

        st.caption(
            "The risk score is a model estimate, not a guarantee "
            "of future failure. Validate probability calibration "
            "before operational use."
        )

        save_prediction(
            machine_type,
            risk,
            prediction,
            health,
            action,
        )


# =========================================================
# 9. BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.subheader("Batch Machine Risk Analysis")

    batch_file = st.file_uploader(
        "Upload machine sensor CSV",
        type=["csv"],
        key="batch_upload",
    )

    st.caption(
        "Required columns: Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], "
        "Torque [Nm], Tool wear [min]. A failure target is optional."
    )

    if batch_file is not None:

        try:
            batch = load_data(batch_file)
            features = engineer_features(batch)

            probabilities = model.predict_proba(features)[:, 1]

            batch["Failure Risk"] = probabilities
            batch["Predicted Failure"] = (
                probabilities >= threshold
            ).astype(int)

            batch["Health"] = [
                classify_health(float(r), threshold)[0]
                for r in probabilities
            ]

            batch["Recommended Action"] = [
                classify_health(float(r), threshold)[1]
                for r in probabilities
            ]

            st.metric(
                "Machines Flagged",
                int(batch["Predicted Failure"].sum()),
            )

            st.dataframe(
                batch.sort_values(
                    "Failure Risk",
                    ascending=False,
                ),
                use_container_width=True,
            )

            fig = px.histogram(
                batch,
                x="Failure Risk",
                color="Health",
                nbins=25,
                title="Machine Failure Risk Distribution",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Prediction Report",
                batch.to_csv(index=False).encode("utf-8"),
                file_name="machine_risk_report.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error(f"Batch prediction failed: {error}")


# =========================================================
# 10. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.subheader("Machine Learning Model Comparison")

    st.dataframe(
        metrics.style.format("{:.3f}"),
        use_container_width=True,
    )

    chart_data = metrics.reset_index()

    fig = px.bar(
        chart_data,
        x="Model",
        y=["Precision", "Recall", "F1 Score", "PR-AUC"],
        barmode="group",
        title="Model Performance Comparison",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Failure"],
        columns=["Predicted Normal", "Predicted Failure"],
    )

    st.dataframe(cm_df, use_container_width=True)

    st.download_button(
        "Download Evaluation Metrics",
        metrics.to_csv().encode("utf-8"),
        file_name="model_metrics.csv",
        mime="text/csv",
    )

    st.info(
        "Metrics are calculated on a held-out test set at a "
        "classification threshold of 0.50. The dashboard alert "
        "threshold is independently configurable."
    )


# =========================================================
# 11. MAINTENANCE HISTORY
# =========================================================

elif page == "Maintenance History":

    st.subheader("Prediction History")

    history = get_history()

    if history.empty:
        st.info(
            "No prediction records yet. Run a machine prediction first."
        )

    else:
        st.dataframe(
            history,
            use_container_width=True,
        )

        st.download_button(
            "Export Maintenance History",
            history.to_csv(index=False).encode("utf-8"),
            file_name="maintenance_history.csv",
            mime="text/csv",
        )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "CODETECH Internship Project | Predictive Maintenance "
    "for Machinery | Educational prototype. Not certified "
    "for safety-critical industrial control."
)
