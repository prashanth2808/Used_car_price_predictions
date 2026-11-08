# Used Car Price Prediction 🚗

A Machine Learning project to predict the **selling price of used cars** based on vehicle characteristics, using the [CarDekho dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

---

## 📌 Project Overview

This project builds and compares multiple regression models to estimate used car prices. It covers the full data science pipeline — from exploratory data analysis to model evaluation and prediction.

---

## 📂 Project Structure

```
Used_car_price_prediction/
│
├── CAR DETAILS FROM CAR DEKHO.csv    # Raw dataset
├── Used_cars_price_prediction.ipynb  # Main Jupyter Notebook
├── .gitignore
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `name` | Car make and model |
| `year` | Manufacturing year |
| `selling_price` | Target variable — car's selling price |
| `km_driven` | Total kilometers driven |
| `fuel` | Fuel type (Petrol / Diesel / CNG / LPG / Electric) |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `transmission` | Manual / Automatic |
| `owner` | Ownership history |

---

## 🔬 Notebook Walkthrough

1. **Data Loading & Exploration** — shape, types, nulls, duplicates
2. **EDA** — univariate, bivariate and multivariate analysis with visualizations
3. **Feature Engineering** — brand/model extraction, car age, age categorization, log transformation
4. **Model Building** — 5 regression models trained using sklearn Pipelines:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
5. **Model Evaluation** — MAE, RMSE, R² Score comparison
6. **Prediction** — Individual car price predictions on new inputs

---

## 🛠️ Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Used_car_price_prediction.git
cd Used_car_price_prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# Launch the notebook
jupyter notebook Used_cars_price_prediction.ipynb
```

---

## 📄 License

This project is for educational purposes.
