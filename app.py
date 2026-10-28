from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).parent
DEFAULT_CSV = APP_DIR / "india_housing_prices.csv"
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REG_MODEL_PATH = ARTIFACT_DIR / "regression_model.joblib"
CLF_MODEL_PATH = ARTIFACT_DIR / "classification_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_schema.json"

st.set_page_config(
    page_title="India Housing Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#132f56,#1e4978); color: white; margin-bottom: 1rem;}
.hero h1 {margin:0 0 .4rem 0; font-size:2.1rem;}
.hero p {margin:0; opacity:.9;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 14px; border-radius: 12px;}
.small-note {font-size:.88rem; opacity:.8;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>🏠 India Housing Intelligence</h1>
      <p>Housing price estimation · Model evaluation · Investment classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Data utilities ----------
@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    frame = pd.read_csv(BytesIO(file_bytes))
    frame.columns = frame.columns.astype(str).str.strip()
    for col in frame.select_dtypes(include=["object"]).columns:
        frame[col] = frame[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return frame


@st.cache_data(show_spinner=False)
def read_csv_path(path: str, modified_time: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    for col in frame.select_dtypes(include=["object"]).columns:
        frame[col] = frame[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return frame


def load_default_data() -> pd.DataFrame | None:
    if DEFAULT_CSV.exists():
        return read_csv_path(str(DEFAULT_CSV), DEFAULT_CSV.stat().st_mtime)
    return None


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    mapped = series.astype("string").str.strip().str.lower().map({
        "good investment": 1, "not a good investment": 0,
        "not good investment": 0, "good": 1, "bad": 0,
        "yes": 1, "no": 0, "true": 1, "false": 0,
        "1": 1, "0": 0
    })
    return mapped


def clean_features(frame: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    drop_cols = {"ID"}
    if target:
        drop_cols.add(target)
    X = frame.drop(columns=[c for c in drop_cols if c in frame.columns]).copy()

    # Remove target-derived or direct leakage columns from feature inputs.
    leakage_names = {
        "price_per_sqft", "price_per_sqft_calc",
        "price_after_5_years", "good_investment"
    }
    X = X.drop(columns=[c for c in X.columns if c.strip().lower() in leakage_names], errors="ignore")

    # Avoid unsupported types and normalize booleans/categorical columns.
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("string")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    keep = numeric + categorical
    X = X[keep]
    return X, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5, max_categories=100)),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable numeric or categorical input features were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def compact_categories(X: pd.DataFrame, categorical: list[str], max_categories: int = 100) -> pd.DataFrame:
    """Bound feature expansion by grouping less-common category values."""
    X = X.copy()
    for col in categorical:
        counts = X[col].value_counts(dropna=True)
        if len(counts) > max_categories:
            keep = set(counts.head(max_categories - 1).index)
            X[col] = X[col].where(X[col].isin(keep), "Other")
    return X


def money_lakhs(value: float) -> str:
    return f"₹{value:,.2f} lakh"


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def build_prediction_row(
    schema: dict, numeric_values: dict[str, float], categorical_values: dict[str, str]
) -> pd.DataFrame:
    row = {}
    for col in schema["features"]:
        if col in numeric_values:
            row[col] = numeric_values[col]
        elif col in categorical_values:
            row[col] = categorical_values[col]
        else:
            row[col] = np.nan
    return pd.DataFrame([row], columns=schema["features"])


# ---------- Sidebar / data ----------
with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload housing CSV", type=["csv"], help="Upload india_housing_prices.csv or your cleaned CSV.")
    st.caption("For a deployed app, include the CSV in the GitHub repository only if you are permitted to publish it.")
    st.divider()
    st.header("Navigation")
    page = st.radio(
        "Choose a view",
        ["Overview", "Predict price", "Train & evaluate", "Model metrics", "Data explorer"],
        label_visibility="collapsed",
    )

try:
    if uploaded is not None:
        df = read_csv_bytes(uploaded.getvalue())
        data_source_label = f"Uploaded file: {uploaded.name}"
    else:
        df = load_default_data()
        data_source_label = "Repository CSV" if df is not None else "No CSV loaded"
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df is None:
    st.warning("No dataset found yet. Upload a CSV in the sidebar, or add `india_housing_prices.csv` to the app repository.")
    st.info("Expected columns for the core workflow include `Price_in_Lakhs`, `Size_in_SqFt`, and optionally `Good_Investment`.")
    st.stop()

df.columns = df.columns.astype(str).str.strip()
st.caption(f"Data source: {data_source_label} · {len(df):,} rows · {len(df.columns)} columns")

price_col = "Price_in_Lakhs" if "Price_in_Lakhs" in df.columns else None
size_col = "Size_in_SqFt" if "Size_in_SqFt" in df.columns else None
state_col = "State" if "State" in df.columns else None
city_col = "City" if "City" in df.columns else None

# ---------- Views ----------
if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties", f"{len(df):,}")
    c2.metric("Features", f"{len(df.columns):,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    if price_col:
        prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
        if not prices.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Median asking price", money_lakhs(float(prices.median())))
            m2.metric("Mean asking price", money_lakhs(float(prices.mean())))
            m3.metric("Price range", f"{money_lakhs(float(prices.min()))} – {money_lakhs(float(prices.max()))}")
            st.plotly_chart(px.histogram(
                df, x=price_col, nbins=35, title="Property price distribution",
                labels={price_col: "Price (₹ lakh)"}
            ), use_container_width=True)
    if state_col and price_col:
        state_means = df.assign(_price=pd.to_numeric(df[price_col], errors="coerce")).groupby(state_col, dropna=False)["_price"].mean().dropna().sort_values(ascending=False).head(20).reset_index()
        if not state_means.empty:
            st.plotly_chart(px.bar(state_means, x=state_col, y="_price", title="Top 20 states by average listed price", labels={"_price": "Average price (₹ lakh)"}), use_container_width=True)
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

elif page == "Predict price":
    st.subheader("Estimate a property's price")
    if not REG_MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.warning("No saved regression model is available yet. Go to **Train & evaluate** and train a model first.")
        st.stop()

    try:
        model = joblib.load(REG_MODEL_PATH)
        schema = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
        st.stop()

    st.caption("Predictions are estimates based on the training dataset, not a professional valuation or investment guarantee.")
    numeric_values: dict[str, float] = {}
    categorical_values: dict[str, str] = {}
    with st.form("prediction_form"):
        left, right = st.columns(2)
        numeric_features = schema.get("numeric_features", [])
        categorical_features = schema.get("categorical_features", [])
        for i, col in enumerate(numeric_features):
            values = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else pd.Series(dtype=float)
            default = float(values.median()) if not values.empty else 0.0
            low, high = (float(values.min()), float(values.max())) if not values.empty else (0.0, max(1.0, default * 2))
            if not math.isfinite(low): low = 0.0
            if not math.isfinite(high) or high < low: high = max(low + 1.0, default)
            # Use number input to avoid slider issues for wide ranges.
            with (left if i % 2 == 0 else right):
                numeric_values[col] = st.number_input(
                    col.replace("_", " "), value=default, format="%.3f",
                    help=f"Training-data median: {default:.3f}"
                )
        for i, col in enumerate(categorical_features):
            series = df[col].dropna().astype(str) if col in df else pd.Series(dtype=str)
            options = sorted(series.unique().tolist())[:100]
            if "Other" not in options:
                options.append("Other")
            if not options:
                options = ["Unknown"]
            with (left if i % 2 == 0 else right):
                categorical_values[col] = st.selectbox(col.replace("_", " "), options=options)
        submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

    if submitted:
        try:
            input_frame = build_prediction_row(schema, numeric_values, categorical_values)
            prediction = float(model.predict(input_frame)[0])
            st.success(f"Estimated current price: {money_lakhs(prediction)}")
            st.metric("Estimated price (₹ lakh)", f"{prediction:,.2f}")
            assumption = st.number_input(
                "Optional annual appreciation assumption (%)",
                min_value=-50.0, max_value=100.0, value=7.0, step=0.5,
                help="Scenario calculation only. It is not a model-validated five-year forecast."
            )
            future = prediction * (1 + assumption / 100) ** 5
            st.metric("Five-year scenario value", money_lakhs(future))
            st.caption("Five-year value is a scenario using your selected annual rate, not an observed-price forecast.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Train & evaluate":
    st.subheader("Train and evaluate models")
    st.warning("Training runs on the app server. For large datasets, train offline and deploy saved artifacts instead.")
    st.markdown("#### Regression")
    if not price_col:
        st.error("The dataset needs a `Price_in_Lakhs` column to create the regression target.")
    else:
        target_choice = st.radio(
            "Regression target",
            ["Current price (Price_in_Lakhs)", "Simulated five-year price (7% annual growth)"],
            help="The simulated target is formula-generated and cannot validate actual future prices.",
        )
        reg_target = "Price_in_Lakhs" if target_choice.startswith("Current") else "Price_After_5_Years"
        reg_frame = df.copy()
        simulated = reg_target == "Price_After_5_Years"
        if simulated:
            reg_frame[reg_target] = pd.to_numeric(reg_frame[price_col], errors="coerce") * (1.07 ** 5)

        if st.button("Train regression models", type="primary"):
            try:
                y_reg = pd.to_numeric(reg_frame[reg_target], errors="coerce")
                valid = y_reg.notna() & np.isfinite(y_reg) & (y_reg > 0)
                working = reg_frame.loc[valid].copy()
                X, numeric, categorical = clean_features(working, reg_target)
                # Prevent the current-price target from appearing as an input.
                if reg_target == "Price_in_Lakhs":
                    X = X.drop(columns=["Price_in_Lakhs"], errors="ignore")
                    numeric = [c for c in numeric if c != "Price_in_Lakhs"]
                X = compact_categories(X, categorical)
                if X.empty:
                    raise ValueError("No usable features remain after excluding leakage columns.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_reg.loc[valid], test_size=0.2, random_state=42
                )
                pre = make_preprocessor(numeric, categorical)
                models = {
                    "Linear Regression": Pipeline([("preprocessor", pre), ("model", LinearRegression())]),
                    "Random Forest Regressor": Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(
                        n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1
                    ))]),
                }
                metrics = {"target": reg_target, "target_is_simulated": simulated, "models": {}, "features": X.columns.tolist()}
                progress = st.progress(0, text="Preparing regression training...")
                status = st.empty()
                for i, (name, pipeline) in enumerate(models.items(), start=1):
                    status.info(f"Training {name}...")
                    pipeline.fit(X_train, y_train)
                    pred = pipeline.predict(X_test)
                    model_metrics = {
                        "MAE": float(mean_absolute_error(y_test, pred)),
                        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                        "R2": float(r2_score(y_test, pred)),
                        "train_rows": int(len(X_train)),
                        "test_rows": int(len(X_test)),
                    }
                    metrics["models"][name] = model_metrics
                    progress.progress(i / len(models), text=f"Completed {name}")
                # Save Random Forest by default for interactive estimates.
                selected_model = models["Random Forest Regressor"]
                joblib.dump(selected_model, REG_MODEL_PATH)
                schema = {
                    "features": X.columns.tolist(),
                    "numeric_features": numeric,
                    "categorical_features": categorical,
                    "target": reg_target,
                    "target_is_simulated": simulated,
                }
                save_json(FEATURES_PATH, schema)
                all_metrics = read_json(METRICS_PATH)
                all_metrics["regression"] = metrics
                save_json(METRICS_PATH, all_metrics)
                status.success("Regression training complete. Metrics and model saved in the current app environment.")
                st.dataframe(pd.DataFrame([
                    {"Model": name, **values}
                    for name, values in metrics["models"].items()
                ]), use_container_width=True)
                if simulated:
                    st.warning("These scores measure reproduction of the simulated 7% growth formula, not real-world five-year forecast performance.")
                st.rerun()
            except Exception as exc:
                st.error(f"Regression training failed: {exc}")

    st.divider()
    st.markdown("#### Investment classification")
    if "Good_Investment" not in df.columns:
        st.info("No `Good_Investment` target exists in this CSV. Classification training is disabled until you add and validate this label.")
    else:
        st.caption("Only use labels derived from a defensible, documented investment criterion. Avoid using future outcomes or target-derived fields as input features.")
        if st.button("Train classification models"):
            try:
                y_cls = normalize_target(df["Good_Investment"])
                valid = y_cls.notna()
                working = df.loc[valid].copy()
                y_cls = y_cls.loc[valid].astype(int)
                if y_cls.nunique() != 2:
                    raise ValueError("Good_Investment must contain exactly two valid classes.")
                X, numeric, categorical = clean_features(working, "Good_Investment")
                X = compact_categories(X, categorical)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
                )
                pre = make_preprocessor(numeric, categorical)
                models = {
                    "Logistic Regression": Pipeline([("preprocessor", pre), ("model", LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=42
                    ))]),
                    "Random Forest Classifier": Pipeline([("preprocessor", pre), ("model", RandomForestClassifier(
                        n_estimators=100, max_depth=15, min_samples_leaf=2,
                        class_weight="balanced", random_state=42, n_jobs=-1
                    ))]),
                }
                cls_metrics = {"models": {}, "target": "Good_Investment"}
                progress = st.progress(0, text="Training classification models...")
                for i, (name, pipeline) in enumerate(models.items(), start=1):
                    pipeline.fit(X_train, y_train)
                    pred = pipeline.predict(X_test)
                    cls_metrics["models"][name] = {
                        "Accuracy": float(accuracy_score(y_test, pred)),
                        "Precision": float(precision_score(y_test, pred, zero_division=0)),
                        "Recall": float(recall_score(y_test, pred, zero_division=0)),
                        "F1": float(f1_score(y_test, pred, zero_division=0)),
                        "Confusion matrix": confusion_matrix(y_test, pred).tolist(),
                        "Classification report": classification_report(y_test, pred, zero_division=0, output_dict=True),
                    }
                    progress.progress(i / len(models), text=f"Completed {name}")
                joblib.dump(models["Random Forest Classifier"], CLF_MODEL_PATH)
                all_metrics = read_json(METRICS_PATH)
                all_metrics["classification"] = cls_metrics
                save_json(METRICS_PATH, all_metrics)
                st.success("Classification training complete.")
                st.dataframe(pd.DataFrame([
                    {"Model": name, **{k: v for k, v in values.items() if k not in ["Confusion matrix", "Classification report"]}}
                    for name, values in cls_metrics["models"].items()
                ]), use_container_width=True)
                st.rerun()
            except Exception as exc:
                st.error(f"Classification training failed: {exc}")

elif page == "Model metrics":
    st.subheader("Saved model metrics")
    metrics = read_json(METRICS_PATH)
    if not metrics:
        st.info("No saved metrics found. Use **Train & evaluate** to train models and generate metrics.")
    else:
        reg = metrics.get("regression")
        if reg:
            st.markdown("#### Regression results")
            if reg.get("target_is_simulated"):
                st.warning("Regression target is simulated using a fixed 7% annual appreciation assumption. These metrics are not evidence of real-world future-price accuracy.")
            st.caption(f"Target: `{reg.get('target', 'unknown')}`")
            reg_df = pd.DataFrame([{"Model": n, **v} for n, v in reg.get("models", {}).items()])
            st.dataframe(reg_df, use_container_width=True)
            if not reg_df.empty and "RMSE" in reg_df:
                st.plotly_chart(px.bar(reg_df, x="Model", y=["MAE", "RMSE"], barmode="group", title="Regression error comparison"), use_container_width=True)
            if not reg_df.empty and "R2" in reg_df:
                st.plotly_chart(px.bar(reg_df, x="Model", y="R2", title="Regression R² comparison"), use_container_width=True)
        cls = metrics.get("classification")
        if cls:
            st.markdown("#### Classification results")
            cls_df = pd.DataFrame([
                {"Model": n, **{k: v for k, v in vals.items() if k not in ["Confusion matrix", "Classification report"]}}
                for n, vals in cls.get("models", {}).items()
            ])
            st.dataframe(cls_df, use_container_width=True)
            if not cls_df.empty:
                st.plotly_chart(px.bar(cls_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1"], barmode="group", title="Classification metric comparison"), use_container_width=True)
            for model_name, vals in cls.get("models", {}).items():
                with st.expander(f"{model_name}: confusion matrix"):
                    cm = vals.get("Confusion matrix")
                    if cm:
                        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), use_container_width=True)
        st.download_button(
            "Download metrics JSON",
            data=json.dumps(metrics, indent=2),
            file_name="housing_model_metrics.json",
            mime="application/json",
        )

elif page == "Data explorer":
    st.subheader("Data explorer")
    with st.expander("Column types and missing values", expanded=True):
        summary = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "unique": df.nunique(dropna=True),
        }).sort_values(["missing", "unique"], ascending=False)
        st.dataframe(summary, use_container_width=True)
    if price_col and state_col:
        state_options = ["All"] + sorted(df[state_col].dropna().astype(str).unique().tolist())
        selected_state = st.selectbox("Filter by state", state_options)
        filtered = df if selected_state == "All" else df[df[state_col].astype(str) == selected_state]
    else:
        filtered = df
    if city_col:
        city_options = ["All"] + sorted(filtered[city_col].dropna().astype(str).unique().tolist())
        selected_city = st.selectbox("Filter by city", city_options)
        if selected_city != "All":
            filtered = filtered[filtered[city_col].astype(str) == selected_city]
    st.write(f"Showing {len(filtered):,} rows")
    st.dataframe(filtered, use_container_width=True, height=500)
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="housing_data_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    '<p class="small-note">For research and decision support only. Estimates depend on the data and assumptions supplied; they are not financial advice, an appraisal, or a guarantee of investment returns.</p>',
    unsafe_allow_html=True,
)
