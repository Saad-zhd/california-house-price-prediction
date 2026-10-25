# California House Price Prediction System 🏠
### AIC354: Machine Learning Fundamentals — Assignment 3 (Task 2)
**Instructor:** Dr. Rao Muhammad Adeel Nawab  
**Institution:** COMSATS University Islamabad, Lahore Campus  

---

## 📌 Project Overview
This project presents an end-to-end Machine Learning solution that predicts the median house value of California census block groups from demographic, geographic, and structural features. It adheres to the half-cooked pedagogical structure demonstrated in the Car Price Prediction System, evaluating **5 distinct machine learning algorithms** using **10-Fold Cross-Validation** and deploying the best model as an interactive cloud web application.

---

## 📂 Repository Structure
```
House Price Prediction K-Fold Cross Validation/
├── housing.csv                                   # Full real-world dataset (20,640 records)
├── house_price_kfold_full_dataset_10fold.ipynb   # Complete 10-Fold CV notebook on full dataset
├── house_price_kfold_100_instances_3fold.ipynb   # Fast 3-Fold CV notebook (100 instances for sanity check)
├── house_price_model.pkl                         # Trained production pipeline (preprocessor + model)
├── app.py                                        # Streamlit web application
├── requirements.txt                              # Python library dependencies
├── .streamlit/
│   └── config.toml                               # Streamlit UI theme configuration
└── README.md                                     # Project documentation & deployment guide
```

---

## 📊 Dataset Description (`housing.csv`)
- **Instances:** 20,640 census block groups (1990 U.S. Census for California).
- **Target Attribute ($y$):** `median_house_value` (USD).
- **Features ($X$):**
  1. `longitude` (Float): East-west geographic coordinate.
  2. `latitude` (Float): North-south geographic coordinate.
  3. `housing_median_age` (Float): Median age of houses in block group.
  4. `total_rooms` (Float): Total count of rooms across all households.
  5. `total_bedrooms` (Float): Total count of bedrooms (contains 207 missing values imputed via median).
  6. `population` (Float): Total resident population in block group.
  7. `households` (Float): Total number of households.
  8. `median_income` (Float): Median household income (in units of $10,000 USD).
  9. `ocean_proximity` (Categorical): Location relative to the Pacific coast (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`).

---

## ⚙️ Pre-Processing Pipeline
All features are transformed using scikit-learn's `ColumnTransformer`:
- **Numeric Pipeline:**
  - `SimpleImputer(strategy='median')`: Imputes missing `total_bedrooms` robustly without data leakage.
  - `StandardScaler()`: Normalizes variables to $\mu = 0, \sigma = 1$.
- **Categorical Pipeline:**
  - `SimpleImputer(strategy='most_frequent')`: Safety imputation for missing categorical tags.
  - `OneHotEncoder(drop='first', handle_unknown='ignore')`: Converts categorical levels into binary indicator columns.

---

## 🔬 Algorithm Evaluation (10-Fold Cross-Validation)
The following **5 regression algorithms** (all completely distinct from those used in the Car Price Prediction System) were evaluated across 10 identical folds on all 20,640 instances:

| Rank | Algorithm | Category | Mean RMSE ($) | Std RMSE ($) | RMSE % of Avg Price | Total 10-Fold Time |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Histogram Gradient Boosting** | Histogram Binned Boosting | **$47,418** | ± $2,056 | **22.92%** | 2.62 s |
| 🥈 | **XGBoost Regressor** | Extreme Gradient Boosting | **$47,738** | ± $1,917 | **23.08%** | 1.23 s |
| 🥉 | **Extra Trees Regressor** | Bagging Ensemble | **$51,142** | ± $1,990 | **24.72%** | 6.55 s |
| 4 | **Ridge Regression** | Regularized Linear (L2) | **$68,904** | ± $2,755 | **33.31%** | 0.66 s |
| 5 | **AdaBoost Regressor** | Adaptive Boosting | **$89,372** | ± $2,935 | **43.20%** | 13.22 s |

**Selected Production Model:** **Histogram Gradient Boosting** was selected for final persistence into `house_price_model.pkl` due to its superior RMSE, handling of non-linear spatial interactions, and zero-dependency portability.

---

## 🚀 How to Run Locally

### 1. Run the Streamlit Web Application
```bash
# Navigate to the project directory
cd "House Price Prediction K-Fold Cross Validation"

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Jupyter Notebooks
Open either notebook in VS Code or Jupyter Lab:
- `house_price_kfold_100_instances_3fold.ipynb` (quick run in ~2 seconds)
- `house_price_kfold_full_dataset_10fold.ipynb` (full 10-fold CV on all 20,640 instances)

---

## ☁️ How to Deploy to the Cloud (Free Live Web App)

### Option A: Streamlit Community Cloud (Recommended & Free)
1. Push this folder to a GitHub repository (e.g., `https://github.com/<your-username>/california-house-price-prediction`).
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New App"**.
4. Select your repository, branch (`main`), and set Main file path to `app.py` (or `House Price Prediction K-Fold Cross Validation/app.py`).
5. Click **"Deploy"**.
6. Once live, copy your public URL (e.g., `https://california-house-price-prediction.streamlit.app`) and paste it at the top of your notebook in Cell 0.

### Option B: Hugging Face Spaces (Instant & Free)
1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. Go to **Spaces → Create new Space**.
3. Select SDK: **Streamlit**, Hardware: **Free (CPU basic)**.
4. Upload `app.py`, `house_price_model.pkl`, and `requirements.txt`.
5. Your app builds automatically and is hosted live with a permanent URL!

---

## 📦 Google Classroom Submission Checklist
- [x] Full dataset file: `housing.csv`
- [x] Trained model file: `house_price_model.pkl`
- [x] Jupyter Notebook file with pre-computed outputs: `house_price_kfold_full_dataset_10fold.ipynb`
- [x] Deployment link included at the top of the notebook file.
- [x] Interactive web application: `app.py` with requirements.
