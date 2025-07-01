# 🎯 Predictive Modeling with Decision Trees & Gradient Boosting

This project explores customer behavior prediction using decision trees and gradient boosting. It started as a standard classification task and quickly evolved into a lesson in **evaluating imbalanced data**, **thoughtful feature engineering**, and **tuning for performance** beyond raw accuracy.

## 📂 Project Overview

- Dataset: Demographic and socioeconomic customer data  
- Goal: Predict the target class label accurately and fairly, even under class imbalance  
- Tools: Python, Jupyter Notebook, pandas, scikit-learn, seaborn, NumPy

## 🛠️ Workflow Summary

### 🔹 Data Cleaning & Preprocessing
- Dropped irrelevant columns (`Unnamed: 18`)
- Imputed missing values using median and constant strategies
- Removed leading/trailing whitespaces from column names

### 🔹 Feature Engineering
- Created new features:
  - `Age`: derived from `Birthday_count`
  - `Employment_years`: computed from `Employed_days`
- One-hot encoding for all categorical variables

### 🔹 Model Building
- Used `DecisionTreeClassifier` with entropy criterion
- Tuned hyperparameters using `GridSearchCV` and `RandomizedSearchCV`

### 🔹 Addressing Class Imbalance
- Identified skewed label distribution: Class `1` was underrepresented
- Switched to `HistGradientBoostingClassifier` with `class_weight='balanced'` for better recall
- Evaluated with precision, recall, F1-score, and confusion matrix

## ✅ Key Learnings

- Accuracy is **not enough** in imbalanced datasets—confusion matrices and F1 scores matter
- Feature engineering (like age and employment length) added real predictive value
- Balancing the data and choosing the right metric dramatically improved the model’s relevance

## 🔮 Next Steps

- Implement **Logistic Regression** with **Recursive Feature Elimination (RFE)** for interpretability
- Impute remaining missing values for SMOTE compatibility
- Explore **SMOTE** to synthetically oversample minority class data
- Build clearer model interpretability plots (e.g., coefficient analysis)

## 📊 Final Performance Snapshot (Gradient Boosting)
- **Test Accuracy**: ~90%
- **F1-Score on minority class**: Improved significantly
- **Confusion Matrix**: Balanced predictions with reduced false negatives

---

Made with 🌱 curiosity and 💻 code.

