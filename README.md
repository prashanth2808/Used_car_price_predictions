# Used Car Price Prediction 🚗

A complete end-to-end **Machine Learning project** that predicts the **selling price of used cars** based on vehicle characteristics, using the [CarDekho dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

---

## 📌 Project Overview

This project builds and compares multiple regression models to estimate used car prices. It covers the full data science pipeline — from raw data exploration to model evaluation and individual car price prediction.

---

## 📂 Project Structure

```
Used_car_price_prediction/
│
├── CAR DETAILS FROM CAR DEKHO.csv    # Raw dataset from CarDekho
├── Used_cars_price_prediction.ipynb  # Main Jupyter Notebook (126 cells)
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**Source:** CarDekho Used Car Dataset

| Feature | Description |
|---|---|
| `name` | Car make and model name |
| `year` | Manufacturing year |
| `selling_price` | Target variable — car's selling price |
| `km_driven` | Total kilometers driven |
| `fuel` | Fuel type (Petrol / Diesel / CNG / LPG / Electric) |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `transmission` | Manual / Automatic |
| `owner` | Ownership history (First / Second / Third / Fourth+ Owner) |

---

## ✅ What Has Been Done

### 1. 📥 Data Loading & Initial Understanding
- Loaded the CarDekho CSV dataset using `pandas`
- Inspected dataset shape, column names, data types (`df.info()`)
- Reviewed basic statistics for numerical and categorical columns (`df.describe()`)
- Checked for **missing values** and **duplicate records**
- Counted unique values in each column to understand cardinality
- Identified **independent features (X)** and **target variable (y = selling_price)**

---

### 2. 🔍 Exploratory Data Analysis (EDA)

#### Univariate Analysis
- Plotted **histograms with KDE** for:
  - `selling_price` — to understand price distribution
  - `km_driven` — to understand mileage distribution
  - `year` — to understand manufacturing year spread
- Plotted **count plots** for all categorical features:
  - `fuel`, `seller_type`, `transmission`, `owner`
- Printed **value counts and percentage distribution** for each categorical column

#### Outlier Analysis (Box Plots)
- Visualized outliers using **box plots** for:
  - `selling_price`
  - `km_driven`
  - `year`

#### Bivariate Analysis
- **Numerical vs Numerical:**
  - Car Age vs Selling Price (scatter plot)
  - Kilometers Driven vs Selling Price (scatter plot)
- **Categorical vs Numerical:**
  - Fuel Type vs Selling Price
  - Transmission vs Selling Price
  - Ownership Type vs Selling Price
  - Seller Type vs Selling Price

#### Multivariate Analysis
- Fuel Type + Car Age + Selling Price (scatter with hue)
- Fuel Type + Transmission + Selling Price (box plot with hue)
- Seller Type + Ownership + Selling Price (box plot with hue)

#### Correlation Analysis
- Computed **correlation matrix** for numerical features
- Visualized correlations using a **heatmap** (`coolwarm` colormap)

---

### 3. 🧹 Data Cleaning & Preprocessing
- Created a clean working copy of the original dataset
- **Handled missing values**: Checked all columns — no nulls found
- **Removed duplicate records** using `drop_duplicates()` and reset the index
- **Stripped whitespace** from all categorical string columns
- **Outlier Treatment** using IQR (Interquartile Range) method:
  - Calculated Q1, Q3, and IQR for `selling_price`
  - Identified outliers (above/below 1.5×IQR bounds)
  - Applied **capping (clipping)** to handle outliers — values clipped to lower and upper bounds
  - Visualized `selling_price` before and after capping
- **One-Hot Encoding** applied to all categorical columns using `pd.get_dummies()` (drop_first=True)
- **Feature Scaling** using `StandardScaler` on numerical features (`year`, `km_driven`)
- **Train-Test Split**: 80% training / 20% testing using `train_test_split(random_state=42)`

---

### 4. ⚙️ Feature Engineering & Feature Selection

| New Feature | Description |
|---|---|
| `brand` | Extracted from the first word of the `name` column |
| `model` | Extracted from the second word of the `name` column |
| `car_age` | Calculated as `2026 - year` |
| `car_age_category` | Binned into: **New** (≤3 yrs), **Moderate** (4–7 yrs), **Old** (8–12 yrs), **Very Old** (12+ yrs) |
| `km_driven_log` | Log-transformed `km_driven` using `np.log1p()` to reduce skewness |

- Analyzed **top 15 car brands** and **top 20 car models** by frequency
- Computed **feature importance** using a preliminary Random Forest model
- Visualized top 15 most important features using a bar chart
- **Final selected features** for model training:
  `brand`, `model`, `car_age`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`

---

### 5. 🤖 Model Building & Training

Built an **sklearn Pipeline** for each model with:
- **Numerical Transformer**: `SimpleImputer(median)` → `StandardScaler`
- **Categorical Transformer**: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- **ColumnTransformer** to apply the above to respective feature types

Five regression models were trained and compared:

| Model | Key Hyperparameters |
|---|---|
| **Linear Regression** | Default |
| **Decision Tree Regressor** | max_depth=8, min_samples_split=5, min_samples_leaf=2 |
| **Random Forest Regressor** | n_estimators=200 |
| **Gradient Boosting Regressor** | Default |
| **XGBoost Regressor** | Default |

---

### 6. 📈 Model Evaluation

Each model was evaluated on the test set using three metrics:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average absolute difference between actual and predicted |
| **RMSE** | Root Mean Squared Error — penalizes large errors more heavily |
| **R² Score** | Coefficient of Determination — how well the model explains variance |

- Plotted bar charts comparing all models by **MAE**, **RMSE**, and **R² Score**
- **Results sorted by RMSE (ascending)** to identify the best model
- Best performing model was selected automatically and printed

---

### 7. 💾 Model Saving
- Best model saved using `joblib.dump()` to `../models/best_car_price_model.pkl`

---

### 8. 🔮 Individual Car Price Predictions
- Tested the best trained model on individual samples from the test set
- Compared **actual selling price vs predicted selling price** for real data points

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and feature engineering |
| `numpy` | Numerical operations and log transformation |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Preprocessing, pipelines, model training, and evaluation |
| `xgboost` | Gradient boosted trees regression |
| `joblib` | Model serialization |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/prashanth2808/Used_car_price_predictions.git
cd Used_car_price_predictions

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter joblib

# Launch the notebook
jupyter notebook Used_cars_price_prediction.ipynb
```

---

## 📄 License

This project is for educational purposes.
