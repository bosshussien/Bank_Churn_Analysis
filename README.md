# Customer Churn Analysis & Prediction System

## Objective of the Project

The purpose of this project is to analyze customer behavior and transform this information into actionable business intelligence using exploratory analysis, predictive machine learning, and behavioral segmentation techniques.

## Data Source :
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

This project aims to:

* Perform Exploratory Data Analysis (EDA) to uncover customer demographics, financial characteristics, and behavioral patterns.
* Detect structural churn patterns by comparing loyal and attrited customers across transactional and engagement features.
* Handle skewed financial distributions and preserve meaningful banking outliers through statistical preprocessing techniques.
* Build machine learning classification models capable of predicting whether a customer is likely to churn.
* Address class imbalance using SMOTENC oversampling for improved minority-class learning.
* Evaluate model performance using classification metrics such as precision, recall, F1-score, confusion matrix, and ROC-AUC.
* Identify the most influential churn-driving features using feature importance analysis.
* Create reduced-feature predictive models focused on the strongest behavioral indicators.
* Generate business-focused interpretations explaining the lifecycle and progression of customer attrition.
* Provide actionable insights that financial institutions can use for customer retention strategies.

---

# Project Workflow

## Phase 1 — Data Preparation & Cleaning

The dataset was loaded and cleaned by:

* Removing redundant identifier and probabilistic Naive Bayes columns.
* Standardizing categorical text formatting.
* Preserving meaningful banking outliers rather than blindly removing them.
* Applying log transformations to highly skewed financial variables.
* Encoding categorical variables for machine learning compatibility.

---

# Phase 2 — Exploratory Data Analysis (EDA)

EDA was conducted to investigate:

* Customer demographics
* Income distributions
* Card category behavior
* Customer engagement patterns
* Financial activity trends
* Transaction frequency and utilization behavior

The analysis revealed several strong churn indicators:

* Higher inactivity periods
* Increased support contact frequency
* Reduced transaction counts
* Lower revolving balances
* Declining credit utilization

---

# Phase 3 — Predictive Machine Learning

A supervised machine learning pipeline was developed using a Random Forest classifier to predict customer attrition.

## Modeling Steps

* One-hot encoding categorical variables
* Train-test split using stratified sampling
* Training baseline Random Forest models
* Handling class imbalance using SMOTENC
* Evaluating classification performance
* Comparing balanced vs unbalanced training behavior
* Extracting feature importance rankings
* Building a reduced-feature optimized model

---

# Phase 6 — Model Performance

## Baseline Random Forest

* Accuracy: 95%
* Churn Precision: 92%
* Churn Recall: 77%
* ROC-AUC: High discriminatory performance

## SMOTENC Balanced Model

After balancing the minority churn class:

* Accuracy remained stable at 95%
* Churn Recall improved significantly
* More balanced minority-class detection was achieved
* ROC-AUC reached approximately 0.91

This demonstrates that oversampling improved the model’s ability to detect churned customers without sacrificing overall accuracy.

---

# Most Important Churn Predictors

Feature importance analysis revealed that the strongest churn indicators were:

1. Total Transaction Count (`Total_Trans_Ct`)
2. Total Transaction Amount (`Total_Trans_Amt`)
3. Total Revolving Balance (`Total_Revolving_Bal`)
4. Transaction Count Change (`Total_Ct_Chng_Q4_Q1`)
5. Average Utilization Ratio (`Avg_Utilization_Ratio`)

These variables strongly support the behavioral disengagement theory discovered during EDA.

---

# Final Business Interpretation

The project demonstrates that customer churn is not random.

Attrition follows a measurable behavioral deterioration process:

* Customers reduce activity frequency
* Transaction spending declines
* Credit utilization drops
* Balances are intentionally cleared
* Customer support interactions increase
* Long inactivity periods emerge before final exit

By detecting these signals early, banks can proactively intervene with retention campaigns before customers fully disengage.

---

# Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTENC)
* Joblib

---

# Machine Learning Techniques Used

* Random Forest Classification
* Feature Importance Analysis
* SMOTENC Oversampling
* One-Hot Encoding
* Stratified Train/Test Splitting
* Correlation Analysis
* Log Transformation

Otherwise the entire `churn` column becomes NaN in the preprocessing notebook.
