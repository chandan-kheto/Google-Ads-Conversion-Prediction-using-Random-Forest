# 📊 Google Ads Conversion Prediction using Random Forest

## 📝 Project Overview
This project focuses on predicting whether a Google Ads campaign will **lead to a conversion** (sale/sign-up) based on historical advertising data. The dataset is **raw and uncleaned**, simulating real-world scenarios where data cleaning, feature engineering, and preprocessing are essential.

The main goal is to build an **end-to-end Machine Learning pipeline** that:
- Cleans messy marketing data.
- Performs feature engineering to create business-driven features.
- Trains a **Random Forest Classifier** to predict conversions.
- Evaluates the model using relevant performance metrics.

---

## ✅ **Dataset Description**
The dataset contains raw advertising campaign data from Google Ads with columns like:
- **Ad_ID** → Unique campaign ID  
- **Campaign_Name** → Name of the campaign (with typos)  
- **Clicks** → Number of ad clicks  
- **Impressions** → Number of impressions  
- **Cost** → Campaign cost (₹ / $ with mixed symbols)  
- **Leads** → Number of leads generated  
- **Conversions** → Number of conversions (sales/signups)  
- **Conversion Rate** → Ratio of conversions to clicks (to be dropped to avoid leakage)  
- **Sale_Amount** → Revenue generated  
- **Ad_Date** → Date (in inconsistent formats)  
- **Device** → Device type (Mobile/Desktop/Tablet)  
- **Location** → City (with case variations)  
- **Keyword** → Keyword triggering the ad  

---

## ✅ **Key Challenges in the Data**
✔ Inconsistent date formats  
✔ Spelling errors and typos in text columns  
✔ Missing values in numeric fields  
✔ Mixed symbols in currency columns  
✔ Irregular casing in categorical fields  
✔ Duplicate rows  

---

## ✅ **Project Workflow**
### **1. Data Cleaning**
- Removed duplicates
- Handled missing values (Cost, Sale Amount, Clicks, Conversions)
- Converted currency fields to numeric
- Fixed inconsistent date formats
- Standardized text columns (Device, Location, Campaign Name)

### **2. Feature Engineering**
- Created binary target:  
  ```python
  Converted = 1 if Conversions > 0 else 0
Added business features:

CPC (Cost per Click) = Cost / Clicks

CTR (Click-Through Rate) = Clicks / Impressions

Dropped Conversion Rate to prevent data leakage

3. Encoding
Applied Label Encoding on categorical features (Device, Location, Campaign Name)

4. Model Building
Used Random Forest Classifier for prediction

Key hyperparameters:


n_estimators=200, max_depth=10, random_state=42
5. Model Evaluation
Accuracy

Precision

Recall

F1 Score

ROC-AUC

Feature Importance Visualization


✅ Model Performance
Random Forest Classifier Performance:
Accuracy: 0.9712
Precision: 0.9712
Recall: 1.0000
F1 Score: 0.9854
ROC AUC: 0.5228 

(Scores may vary based on dataset size and cleaning method)

✅ Visualizations
Feature Importance Chart

ROC Curve

✅ Tech Stack
Python

Pandas, NumPy for data preprocessing

Matplotlib, Seaborn for visualization

Scikit-learn for ML pipeline

Random Forest Classifier for prediction

