# Bike Rental Demand Prediction - Streamlit App
# Run with: streamlit run app.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 Bike Rental Demand Prediction")
st.caption("Explore the dataset, clean the data, visualize rental patterns, and evaluate a regression model.")

NUMERIC_COLUMNS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
]
MODEL_FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday", "weathersit"]


def load_data(uploaded_file):
    """Load the uploaded CSV, or try Dataset.csv beside this app."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    default_path = Path(__file__).with_name("Dataset.csv")
    if default_path.is_file():
        return pd.read_csv(default_path)
    return None


def clean_data(raw_df):
    """Return a cleaned copy while keeping date parsing separate."""
    data = raw_df.copy()
    data.columns = data.columns.astype(str).str.strip()

    # Treat question marks as missing values, as in the original notebook.
    data = data.replace(r"^\s*\?\s*$", np.nan, regex=True)

    # Convert only columns that actually exist in this CSV.
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    before = len(data)
    data = data.dropna().copy()
    dropped = before - len(data)

    # Parse dates without dropping otherwise useful rows from the modeling data.
    invalid_dates = 0
    if "dteday" in data.columns:
        try:
            data["dteday"] = pd.to_datetime(
                data["dteday"], errors="coerce", format="mixed"
            )
        except (TypeError, ValueError):
            data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce")
        invalid_dates = int(data["dteday"].isna().sum())

    return data, before, dropped, invalid_dates


def make_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


uploaded_file = st.sidebar.file_uploader("Upload your bike rental CSV", type=["csv"])
st.sidebar.markdown(
    "If no file is uploaded, the app looks for **Dataset.csv** in the same folder as `app.py`."
)

try:
    raw_df = load_data(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if raw_df is None:
    st.info("Upload your bike rental CSV using the sidebar, or place `Dataset.csv` beside `app.py`.")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

df, rows_before_cleaning, rows_dropped, invalid_dates = clean_data(raw_df)

required_for_eda = {"hr", "cnt"}
missing_eda = required_for_eda - set(df.columns)
if missing_eda:
    st.error(
        "This CSV is missing required columns for the EDA charts: "
        + ", ".join(sorted(missing_eda))
        + ". Please upload the Bike Sharing hourly dataset with columns such as hr and cnt."
    )
    st.stop()

# Overview
st.header("1. Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Original rows", f"{len(raw_df):,}")
m2.metric("Rows after cleaning", f"{len(df):,}")
m3.metric("Rows removed", f"{rows_dropped:,}")
m4.metric("Columns", f"{raw_df.shape[1]:,}")

with st.expander("Preview original data", expanded=True):
    st.dataframe(raw_df.head(10), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Dataset dimensions")
    st.write(f"**Rows:** {raw_df.shape[0]:,}")
    st.write(f"**Columns:** {raw_df.shape[1]:,}")
    st.write("**Column names:**")
    st.write(list(raw_df.columns))
with right:
    st.subheader("Data types")
    dtype_df = pd.DataFrame({
        "Column": raw_df.columns.astype(str),
        "Data type": raw_df.dtypes.astype(str).values,
    })
    st.dataframe(dtype_df, use_container_width=True)

st.subheader("Missing values and duplicates (original data)")
missing_df = raw_df.isna().sum().rename("Missing values").to_frame()
missing_df["Data type"] = raw_df.dtypes.astype(str)
st.dataframe(missing_df, use_container_width=True)
st.write(f"**Duplicate rows:** {int(raw_df.duplicated().sum()):,}")

with st.expander("Summary statistics"):
    st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)

# Cleaning
st.header("2. Data cleaning")
c1, c2, c3 = st.columns(3)
c1.metric("Rows before dropna", f"{rows_before_cleaning:,}")
c2.metric("Rows removed for missing values", f"{rows_dropped:,}")
c3.metric("Missing values remaining", f"{int(df.isna().sum().sum()):,}")
st.caption("Question marks are treated as missing values. Rows with missing values are removed from the analysis copy; the original uploaded data is not modified.")

if invalid_dates:
    st.warning(
        f"{invalid_dates:,} row(s) have an invalid or unrecognised date in `dteday`. "
        "They are kept for non-date analysis and excluded only from the daily time-series chart."
    )

# EDA visualizations
st.header("3. Exploratory data analysis")

hourly = df.groupby("hr", dropna=True)["cnt"].mean().sort_index()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(hourly.index, hourly.values, marker="o")
ax.set(title="Average Bike Rentals by Hour", xlabel="Hour of the Day", ylabel="Average Number of Rentals")
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
fig.tight_layout()
make_plot(fig)

col1, col2 = st.columns(2)
with col1:
    if "season" in df.columns:
        season_avg = df.groupby("season", dropna=True)["cnt"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(season_avg.index.astype(str), season_avg.values)
        ax.set(title="Average Bike Rentals by Season", xlabel="Season", ylabel="Average Rentals")
        fig.tight_layout()
        make_plot(fig)
    else:
        st.info("The `season` column is not available.")

with col2:
    if "weathersit" in df.columns:
        weather_avg = df.groupby("weathersit", dropna=True)["cnt"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(weather_avg.index.astype(str), weather_avg.values)
        ax.set(title="Average Bike Rentals by Weather Condition", xlabel="Weather Condition", ylabel="Average Rentals")
        fig.tight_layout()
        make_plot(fig)
    else:
        st.info("The `weathersit` column is not available.")

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(data=df, x="cnt", bins=30, ax=ax)
ax.set(title="Distribution of Bike Rentals", xlabel="Total Rentals per Hour", ylabel="Frequency")
fig.tight_layout()
make_plot(fig)

numeric_df = df.select_dtypes(include="number")
if numeric_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Variables")
    fig.tight_layout()
    make_plot(fig)
else:
    st.info("Not enough numeric columns are available to create a correlation heatmap.")

if "dteday" in df.columns:
    dated_df = df.dropna(subset=["dteday"]).copy()
    if not dated_df.empty:
        daily_rentals = dated_df.groupby("dteday")["cnt"].sum().sort_index()
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(daily_rentals.index, daily_rentals.values)
        ax.set(title="Daily Bike Rentals Over Time", xlabel="Date", ylabel="Total Bike Rentals per Day")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        make_plot(fig)
    else:
        st.warning("No valid dates were found in `dteday`, so the daily time-series chart cannot be shown.")
else:
    st.info("The `dteday` column is not available, so the daily time-series chart is skipped.")

# Regression model
st.header("4. Multiple Linear Regression")
st.write("The model predicts `cnt` using the original notebook's selected weather, calendar, and hour features. `casual` and `registered` are intentionally excluded because they add up to `cnt` and would cause target leakage.")

missing_model = [column for column in MODEL_FEATURES + ["cnt"] if column not in df.columns]
if missing_model:
    st.warning("Model training skipped because these required columns are missing: " + ", ".join(missing_model))
else:
    model_df = df[MODEL_FEATURES + ["cnt"]].copy()
    for column in MODEL_FEATURES:
        if column not in CATEGORICAL_FEATURES:
            model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
    model_df["cnt"] = pd.to_numeric(model_df["cnt"], errors="coerce")
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_df) < 10:
        st.warning("At least 10 complete rows are recommended before training the model.")
    else:
        X = model_df[MODEL_FEATURES].copy()
        y = model_df["cnt"].copy()
        X = pd.get_dummies(X, columns=[c for c in CATEGORICAL_FEATURES if c in X.columns], drop_first=True, dtype=float)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)

        test_size = st.sidebar.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan")
        k1, k2, k3 = st.columns(3)
        k1.metric("Training rows", f"{len(X_train):,}")
        k2.metric("RMSE", f"{rmse:.2f}")
        k3.metric("R² score", f"{r2:.3f}" if np.isfinite(r2) else "Not available")

        eval_left, eval_right = st.columns(2)
        with eval_left:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
            lower = min(float(y_test.min()), float(np.min(y_pred)))
            upper = max(float(y_test.max()), float(np.max(y_pred)))
            ax.plot([lower, upper], [lower, upper], "r--", linewidth=2)
            ax.set(title="Actual vs. Predicted Bike Rentals", xlabel="Actual Count", ylabel="Predicted Count")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            make_plot(fig)

        with eval_right:
            residuals = y_test.to_numpy() - y_pred
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.histplot(residuals, kde=True, bins=30, ax=ax)
            ax.axvline(x=0, color="red", linestyle="--")
            ax.set(title="Residuals (Prediction Errors)", xlabel="Actual - Predicted", ylabel="Frequency")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            make_plot(fig)

        st.subheader("Model coefficients")
        coefficients = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_,
        }).sort_values("Coefficient", ascending=False)
        st.dataframe(coefficients, use_container_width=True)
        st.download_button(
            "Download coefficients as CSV",
            data=coefficients.to_csv(index=False).encode("utf-8"),
            file_name="bike_rental_model_coefficients.csv",
            mime="text/csv",
        )

# Feature engineering section from the original notebook
st.header("5. Feature engineering")
if {"weekday", "hr", "workingday", "temp"}.issubset(df.columns):
    engineered = df.copy()
    engineered["weekday"] = pd.to_numeric(engineered["weekday"], errors="coerce")
    engineered["hr"] = pd.to_numeric(engineered["hr"], errors="coerce")
    engineered["workingday"] = pd.to_numeric(engineered["workingday"], errors="coerce")
    engineered["temp_numeric"] = pd.to_numeric(engineered["temp"], errors="coerce")
    engineered["is_weekend"] = engineered["weekday"].isin([0, 6]).astype(int)
    engineered["is_rush_hour"] = (
        (engineered["hr"].between(7, 9) | engineered["hr"].between(16, 18))
        & (engineered["workingday"] == 1)
    ).astype(int)
    engineered["temp_category"] = pd.cut(
        engineered["temp_numeric"],
        bins=[-np.inf, 0.3, 0.6, np.inf],
        labels=["Cold", "Mild", "Hot"],
        right=False,
    )
    show_cols = ["hr", "weekday", "workingday", "temp", "is_weekend", "is_rush_hour", "temp_category"]
    st.dataframe(engineered[show_cols].head(10), use_container_width=True)
else:
    st.info("Feature engineering preview needs the weekday, hr, workingday, and temp columns.")

st.success("Analysis complete. Use the sidebar to upload a different CSV or change the test-set percentage.")
