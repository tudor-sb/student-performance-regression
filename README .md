# Student Performance Prediction

Predicting students' exam scores and finding which factors matter most, based on a dataset of ~6,600 records.

---

## Results

* R2 = 0.82 (RMSE = 1.55, MAE = 0.41) - the model explains 82% of the variance in scores
* Linear Regression chosen over Random Forest (R2 = 0.67), which overfitted
* OLS validation with statsmodels for p-values and interpretable coefficients
* Most influential controllable factors: attendance (+0.20/%), study hours (+0.29/hr), internet access (+0.98)

---

## What I worked with

* pandas: loading, cleaning (missing values / outliers), dropna, filtering, map, get_dummies for ordinal and one-hot encoding
* seaborn / matplotlib: histograms, skew/kurtosis, correlation heatmap, residual plot
* scikit-learn: train_test_split, LinearRegression, RandomForestRegressor, metrics (r2_score, RMSE, MAE), overfitting diagnosis
* statsmodels: OLS, reading p-values and coefficients, avoiding the dummy variable trap

---

## Tech stack

Python, pandas, scikit-learn, statsmodels, matplotlib, seaborn
