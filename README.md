# Telecom Customer Churn Analysis Using ML Models
## Executive Summary
**End-to-end ML project using Python**, built to predict customer churn for a telecom company. The project features two models - Logistic Regression, used as an interpretable baseline, and Random Forest, chosen to capture non-linear relationships between features and churn. The results show that both models can predict customer churn with 77% accuracy.
## Business Problem
Customer churn is a major challenge for telecom companies, where the marketing teams need to answer questions like: 
- Which customers are most likely to churn?
- What factors are driving customers to leave?
- Are certain services associated with higher churn rates?

Instead of reacting after the customer has already left, predicting churn allows companies to proactively retain customers — for example, through personalized offers. This project also uses exploratory data analysis to identify patterns and factors associated with customer churn.  

In this project, I use machine learning classification models ( `Logistic Regression and Random Forest` ) to predict customer churn based on account information, service usage, and billing details.
## Dataset
This project uses the publicly available Telco Customer Churn Dataset from Kaggle.

For licensing reasons, **the dataset is not included in this repository**. Please download it from the link below and place Telco-churn-dataset.csv in the project's root directory (or update the file path in the notebook accordingly).

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

The dataset contains information about 7,043 telecom customers, described by 21 columns. The project primarily focuses on `tenure, MonthlyCharges, TotalCharges, and service/contract-related categorical features`, as these provide the clearest signals of customer behavior and churn risk.
## Project Workflow
```
Load Dataset
        ↓
Explore the Data
        ↓
Data Cleaning & Preparation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering & Encoding
        ↓
Train/Test Split & Feature Scaling
        ↓
Handle Class Imbalance (SMOTE)
        ↓
Train Classification Models (Logistic Regression, Random Forest)
        ↓
Evaluate Model Performance
        ↓
Generate Business Recommendations
```
## Exploratory Data Analysis
Before building the classification models, I explored and cleaned the dataset to better understand customer characteristics and prepare the data for modeling.

The analysis included:

1. Checking for missing values and duplicate records
2. Reviewing data types and unique values across categorical columns
3. Dropping irrelevant columns
4. Fixing data type and formatting issues
5. Generating descriptive statistics
6. Exploring relationships between variables
7. Creating visualisations to identify patterns associated with churn

### Contract Type
![Churn by Contract type](https://github.com/nkorlov-data/Customer-Churn-Analysis-for-Telco/blob/main/images/churn_by_contract_type.png)
**Customers on month-to-month contracts churn at a much higher rate** than those on one- or two-year contracts. Long-term contracts appear to significantly reduce the likelihood of churn.
### Internet Service Type
![Churn by Internet Service type](https://github.com/nkorlov-data/Customer-Churn-Analysis-for-Telco/blob/main/images/churn_by_internet_service.png)
**Customers with Fiber optic internet churn considerably more often** than those with DSL or no internet service, suggesting possible dissatisfaction with this service (e.g. pricing or service quality).
### Payment Method
![Churn by Payment Method](https://github.com/nkorlov-data/Customer-Churn-Analysis-for-Telco/blob/main/images/churn_by_payment_method.png)
**Customers paying via Electronic check show a noticeably higher churn rate** compared to those using automatic payment methods (bank transfer or credit card), which may reflect lower engagement or commitment to the service.
### Monthly Charges
![KDE for MonthlyCharges](https://github.com/nkorlov-data/Customer-Churn-Analysis-for-Telco/blob/main/images/kde_monthly_charges.png)
Customers **who churned tend to have higher monthly charges overall**, with the churned group showing a pronounced density peak between ~$80–$100, while retained customers are more concentrated in the lower charge range.
## Building the Model
Before training, the dataset was split into training and test sets, numerical features (tenure, MonthlyCharges, TotalCharges) were scaled using `StandardScaler`, and class imbalance in the training set was addressed using `SMOTE`, since churned customers made up a minority of the dataset. Both models were trained on the resampled training data and evaluated on the original, untouched test set to reflect real-world class distribution.
## Results
### Random Forest
Classification Report:
```
              precision    recall  f1-score   support

           0       0.86      0.82      0.84      1027
           1       0.57      0.64      0.61       382

    accuracy                           0.77      1409
   macro avg       0.72      0.73      0.72      1409
weighted avg       0.78      0.77      0.78      1409
```
The Random Forest model identified `tenure`, `MonthlyCharges`, and `TotalCharges` as the three most important predictors of churn, followed by `Contract_Month-to-month` and `InternetService`.
![Feature Importance](https://github.com/nkorlov-data/Customer-Churn-Analysis-for-Telco/blob/main/images/feature_importance.png)
This aligns with the patterns observed in the exploratory analysis. Customers with shorter tenure, higher monthly charges, and month-to-month contracts are the most likely to churn. Demographic features such as `gender` and service add-ons like `StreamingTV` or `DeviceProtection` had comparatively little influence on the model's predictions.
### Logistic Regression
Classification Report:
```
              precision    recall  f1-score   support

           0       0.89      0.78      0.83      1027
           1       0.56      0.74      0.64       382

    accuracy                           0.77      1409
   macro avg       0.73      0.76      0.74      1409
weighted avg       0.80      0.77      0.78      1409
```
