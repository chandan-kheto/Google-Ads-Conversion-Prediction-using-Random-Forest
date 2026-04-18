# 📊 Google Ads Performance Prediction (Data Science Project)

## 📝 Project Overview

This project focuses on predicting **high-performing Google Ads campaigns** using real-world marketing data.

Instead of simply predicting whether a conversion occurred, the problem is reframed to:

> 🎯 **Classify ads into high vs low performers based on conversion rate**

This approach aligns with real business goals, where companies aim to **optimize campaign performance and maximize ROI**.

---

## 🎯 Business Objective

* Identify **high-performing ad campaigns**
* Understand **key factors influencing conversion performance**
* Enable **data-driven marketing decisions**

---

## 📁 Dataset Description

The dataset contains Google Ads campaign data with features such as:

* **Ad_ID** → Unique campaign identifier
* **Campaign_Name** → Campaign name (with inconsistencies)
* **Clicks / Impressions** → User engagement metrics
* **Cost** → Advertising spend (₹ / $ mixed formats)
* **Leads / Conversions** → Conversion indicators
* **Conversion Rate** → Used to define target variable
* **Sale_Amount** → Revenue generated
* **Device / Location / Keyword** → Campaign attributes

---

## ⚠️ Real-World Data Challenges

* Inconsistent date formats
* Missing values in numeric columns
* Currency symbols in cost fields
* Typos and inconsistent text formatting
* Duplicate records
* Risk of **data leakage**

---

## 🔧 Project Workflow

### 1️⃣ Data Cleaning

* Removed duplicates
* Handled missing values using median and zero imputation
* Cleaned currency columns (₹, $ removed)
* Standardized categorical features (Device, Location, Campaign Name)
* Converted date column to datetime

---

### 2️⃣ Feature Engineering

Created business-driven features:

* **CTR (Click-Through Rate)** = Clicks / Impressions
* **CPC (Cost Per Click)** = Cost / Clicks

Defined target variable:

```python
High_Conversion = 1 if Conversion_Rate > 0.05 else 0
```

Removed leakage features:

* Conversions
* Conversion Rate

---

### 3️⃣ Exploratory Data Analysis (EDA)

* Conversion distribution analysis
* CTR vs Conversion
* CPC vs Conversion
* Correlation heatmap

📌 **Key Insights:**

* Higher **CTR → better conversion performance**
* Higher **CPC → higher-quality traffic**
* No single feature dominates → multi-factor influence

---

### 4️⃣ Data Preprocessing

* Applied **One-Hot Encoding** for categorical features
* Handled remaining missing values
* Used **train-test split with stratification**

---

### 5️⃣ Model Building

Models used:

* Logistic Regression (baseline)
* Random Forest Classifier

---

### 6️⃣ Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## 📈 Final Model Performance

### 🔹 Logistic Regression

* Accuracy: **78.08%**
* Precision: **0.60**
* Recall: **0.84**
* F1 Score: **0.70**
* ROC-AUC: **0.83**

### 🔹 Random Forest

* Accuracy: **78.27%**
* Precision: **0.63**
* Recall: **0.70**
* F1 Score: **0.66**
* ROC-AUC: **0.82**

---

## 🧠 Model Insights

* Logistic Regression performs better in **recall**, making it ideal for identifying high-performing ads
* Random Forest provides more balanced predictions
* **CTR, CPC, and Clicks** are the most influential features
* Campaign performance depends on **multiple interacting factors**

---

## 💡 Business Impact

* Identifies **high ROI campaigns early**
* Helps optimize **advertising budget allocation**
* Supports better **targeting strategies**

---

## 📊 Visualizations

* Conversion Distribution
* CTR vs Conversion
* CPC vs Conversion
* Feature Importance
* Correlation Heatmap

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🚀 Key Learnings

* Handling **real-world messy data**
* Avoiding **data leakage**
* Importance of **feature engineering (CTR, CPC)**
* Trade-offs between **precision and recall**

---

## 📌 Conclusion

This project demonstrates how machine learning can be applied to **real-world marketing data** to generate actionable insights and improve campaign performance.

---

## 📎 How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run project
python your_script.py
```
