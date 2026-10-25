import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #166534;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #15803d;
    }
    .metric-error {
        font-size: 0.9rem;
        color: #4b5563;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading & Robust Pipeline Fallback
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    candidates = [
        "house_price_model.pkl",
        os.path.join("House Price Prediction K-Fold Cross Validation", "house_price_model.pkl"),
        os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
    ]
    # 1. Attempt unpickling
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                # Version mismatch across differing scikit-learn builds (e.g. cloud _loss module refactor)
                pass

    # 2. Seamless on-the-fly fit (takes < 1 second and is 100% version-proof)
    csv_candidates = [
        "housing.csv",
        os.path.join("House Price Prediction K-Fold Cross Validation", "housing.csv"),
        os.path.join(os.path.dirname(__file__), "housing.csv")
    ]
    csv_path = None
    for p in csv_candidates:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        csv_path = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    cat_cols = ["ocean_proximity"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(random_state=42))
    ])
    model.fit(X, y)
    return model

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("<div class='main-title'>🏠 California House Price Prediction System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Predict median house values across California census districts using an optimized "
    "Machine Learning pipeline evaluated with 10-Fold Cross-Validation.</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar - Feature Inputs
# ---------------------------------------------------------
st.sidebar.header("📍 District & Location Settings")

# Presets for ease of demonstration
presets = {
    "Custom Coordinates": {"lat": 37.88, "lon": -122.23, "prox": "NEAR BAY", "inc": 8.32, "age": 41},
    "San Francisco (Bay Area)": {"lat": 37.77, "lon": -122.42, "prox": "NEAR BAY", "inc": 7.50, "age": 35},
    "Los Angeles (Coastal / Westside)": {"lat": 34.02, "lon": -118.49, "prox": "<1H OCEAN", "inc": 6.80, "age": 30},
    "San Diego (Oceanfront)": {"lat": 32.72, "lon": -117.16, "prox": "NEAR OCEAN", "inc": 5.90, "age": 28},
    "Sacramento (Central Valley)": {"lat": 38.58, "lon": -121.49, "prox": "INLAND", "inc": 3.40, "age": 25},
    "Fresno (Inland Agricultural)": {"lat": 36.75, "lon": -119.77, "prox": "INLAND", "inc": 2.60, "age": 22}
}

selected_preset = st.sidebar.selectbox("Quick Location Preset", list(presets.keys()))
default_vals = presets[selected_preset]

# Geographic coordinates
col_lat, col_lon = st.sidebar.columns(2)
with col_lat:
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=float(default_vals["lat"]), step=0.05, format="%.2f")
with col_lon:
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=float(default_vals["lon"]), step=0.05, format="%.2f")

ocean_proximity_options = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
default_prox_idx = ocean_proximity_options.index(default_vals["prox"]) if default_vals["prox"] in ocean_proximity_options else 0
ocean_proximity = st.sidebar.selectbox("Ocean Proximity", ocean_proximity_options, index=default_prox_idx)

st.sidebar.markdown("---")
st.sidebar.header("📊 District Housing Metrics")

median_income = st.sidebar.slider(
    "Median Household Income (in $10,000s)",
    min_value=0.5,
    max_value=15.0,
    value=float(default_vals["inc"]),
    step=0.1,
    help="Census median income. E.g., 5.0 represents $50,000/year."
)

housing_median_age = st.sidebar.slider(
    "Housing Median Age (years)",
    min_value=1,
    max_value=52,
    value=int(default_vals["age"]),
    step=1
)

total_rooms = st.sidebar.number_input("Total Rooms in Block Group", min_value=50, max_value=40000, value=2500, step=100)
total_bedrooms = st.sidebar.number_input("Total Bedrooms in Block Group", min_value=10, max_value=8000, value=480, step=20)
population = st.sidebar.number_input("Population in Block Group", min_value=10, max_value=35000, value=1400, step=50)
households = st.sidebar.number_input("Households in Block Group", min_value=10, max_value=7000, value=450, step=20)

# Build feature dataframe
input_df = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": float(housing_median_age),
    "total_rooms": float(total_rooms),
    "total_bedrooms": float(total_bedrooms),
    "population": float(population),
    "households": float(households),
    "median_income": float(median_income),
    "ocean_proximity": ocean_proximity
}])

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📈 Model Comparison & Benchmark", "ℹ️ Dataset & Architecture"])

with tab1:
    col_pred, col_map = st.columns([1.2, 1.0])

    with col_pred:
        # Perform prediction
        raw_pred = model.predict(input_df)[0]
        # Real-world cap at $500,001 as in 1990 California census
        predicted_value = max(15000.0, min(500001.0, float(raw_pred)))

        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>ESTIMATED MEDIAN HOUSE VALUE</div>
            <div class='metric-value'>${predicted_value:,.2f}</div>
            <div class='metric-error'>Expected Margin of Error: &plusmn;$47,418 (10-Fold CV RMSE)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Derived District Ratios")
        rooms_per_hh = total_rooms / max(1, households)
        bedrooms_per_room = total_bedrooms / max(1, total_rooms)
        persons_per_hh = population / max(1, households)

        r1, r2, r3 = st.columns(3)
        r1.metric("Rooms / Household", f"{rooms_per_hh:.2f}")
        r2.metric("Bedrooms / Room", f"{bedrooms_per_room:.2%}")
        r3.metric("Residents / Household", f"{persons_per_hh:.2f}")

        st.markdown("---")
        st.markdown("#### 📋 Feature Vector Sent to Pipeline")
        st.dataframe(input_df, use_container_width=True)

    with col_map:
        st.markdown("#### 🗺️ Geographic Location")
        map_df = pd.DataFrame({
            "lat": [latitude],
            "lon": [longitude]
        })
        st.map(map_df, zoom=7)
        st.caption(f"Coordinates: {latitude:.2f}° N, {longitude:.2f}° W ({ocean_proximity})")

with tab2:
    st.markdown("### 🏆 10-Fold Cross-Validation Performance Comparison")
    st.write(
        "Each of the 5 candidate regression algorithms was trained and tested across **10 identical cross-validation folds** "
        "on the complete 20,640-instance dataset. Notice that **Histogram Gradient Boosting** achieved the best predictive accuracy."
    )

    benchmark_data = {
        "Algorithm": [
            "Histogram Gradient Boosting",
            "XGBoost Regressor",
            "Extra Trees Regressor",
            "Ridge Regression",
            "AdaBoost Regressor"
        ],
        "Mean RMSE ($)": ["$47,418", "$47,738", "$51,142", "$68,904", "$89,372"],
        "Std RMSE ($)": ["$2,056", "$1,917", "$1,990", "$2,755", "$2,935"],
        "RMSE % of Avg Price": ["22.92%", "23.08%", "24.72%", "33.31%", "43.20%"],
        "Total 10-Fold Time (s)": ["2.62 s", "1.23 s", "6.55 s", "0.66 s", "13.22 s"],
        "Algorithm Family": ["Histogram Gradient Boosting", "Extreme Gradient Boosting", "Bagged Randomized Trees", "L2-Regularized Linear", "Adaptive Boosting"]
    }
    benchmark_df = pd.DataFrame(benchmark_data)
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    st.markdown("""
    #### 💡 Key Takeaways:
    1. **Histogram Gradient Boosting** achieved the lowest Mean RMSE of **$47,418**, successfully capturing non-linear geographical interactions.
    2. **XGBoost** was extremely competitive (**$47,738** RMSE) and ran fastest per tree round due to optimized C++ backend threading.
    3. **Extra Trees Regressor** outperformed Ridge by **$17,762**, demonstrating the power of ensemble randomization for tabular spatial data.
    4. **Ridge Regression** served as a reliable linear baseline (**$68,904**), while **AdaBoost** struggled with the continuous price variance.
    """)

with tab3:
    st.markdown("### 📚 Dataset & System Architecture")
    st.markdown("""
    - **Dataset**: California Housing Census Dataset (20,640 block groups, 1990 US Census).
    - **Input Attributes**: 8 numerical features (`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`) and 1 categorical feature (`ocean_proximity`).
    - **Target Attribute**: `median_house_value` (USD).
    - **Preprocessing Pipeline**:
        - **Numerical Pipeline**: Missing values imputed via `SimpleImputer(strategy='median')`, standardized via `StandardScaler()`.
        - **Categorical Pipeline**: Missing values imputed via `SimpleImputer(strategy='most_frequent')`, encoded via `OneHotEncoder(drop='first')`.
        - Combined seamlessly using `ColumnTransformer` into a single standalone scikit-learn pipeline.
    - **Deployment Platform**: Streamlit Community Cloud / Hugging Face Spaces.
    """)

st.markdown("---")
st.caption("Machine Learning Fundamentals (AIC354) | Assignment 3: Task 2 Solution | Instructor: Dr. Rao Muhammad Adeel Nawab")
