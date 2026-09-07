# Telecom Customer Churn Analysis Using ML Models
## Executive Summary
**End-to-end ML project using Python**, built to predict customer churn for a telecom company. The project features two models - Logistic Regression, used as an interpretable baseline, and Random Forest, chosen to capture non-linear relationships between features and churn. The results show that both models can predict customer churn with 77% accuracy.
## Business Problem
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

