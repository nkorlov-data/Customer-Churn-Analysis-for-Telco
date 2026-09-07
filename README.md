# Telecom Customer Churn Analysis Using ML Models
## Executive Summary
**End-to-end ML project using Python**, built to predict customer churn for a telecom company. The project features two models - Logistic Regression, used as an interpretable baseline, and Random Forest, chosen to capture non-linear relationships between features and churn. The results show that both models can predict customer churn with 77% accuracy.
## Business problem
Customer churn is a major challenge for telecom companies, where the marketing teams need to answer questions like: 
- Which customers are most likely to churn?
- What factors are driving customers to leave?
- Are certain services associated with higher churn rates?

Instead of reacting after the customer has already left, predicting churn allows companies to proactively retain customers — for example, through personalized offers. This project also uses exploratory data analysis to identify patterns and factors associated with customer churn.  

In this project, I use machine learning classification models **( Logistic Regression and Random Forest )** to predict customer churn based on account information, service usage, and billing details.
## Dataset
This project uses the publicly available Telco Customer Churn Dataset from Kaggle.

For licensing reasons, **the dataset is not included in this repository**. Please download it from the link below and place Telco-churn-dataset.csv in the project's root directory (or update the file path in the notebook accordingly).

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

The dataset contains information about 7,043 telecom customers, described by 21 columns. The project **primarily focuses on tenure, MonthlyCharges, TotalCharges, and service/contract-related categorical features**, as these provide the clearest signals of customer behavior and churn risk.
