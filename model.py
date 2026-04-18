
# ==============================
# 📦 STEP 0: Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# ==============================
# 📂 STEP 1: Load Dataset
# ==============================
df = pd.read_csv("GoogleAds.csv")

print("Initial Shape:", df.shape)
print(df.head())


# ==============================
# 🧹 STEP 2: Data Cleaning
# ==============================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Clean currency columns
df['Cost'] = df['Cost'].replace('[\₹$,]', '', regex=True).astype(float)
df['Sale_Amount'] = df['Sale_Amount'].replace('[\₹$,]', '', regex=True).astype(float)

# Handle missing values
df['Cost'].fillna(df['Cost'].median(), inplace=True)
df['Sale_Amount'].fillna(df['Sale_Amount'].median(), inplace=True)
df['Clicks'].fillna(0, inplace=True)
df['Impressions'].fillna(0, inplace=True)
df['Conversions'].fillna(0, inplace=True)

# Convert date column
df['Ad_Date'] = pd.to_datetime(df['Ad_Date'], errors='coerce')

# Clean categorical columns
df['Device'] = df['Device'].str.strip().str.title()
df['Location'] = df['Location'].str.strip().str.title()
df['Campaign_Name'] = df['Campaign_Name'].str.strip().str.title()


# ==============================
# ⚙️ STEP 3: Feature Engineering
# ==============================

# Target variable (Conversion: Yes/No)
df['High_Conversion'] = df['Conversion Rate'].apply(lambda x: 1 if x > 0.05 else 0)

# Create new features
df['CPC'] = df.apply(lambda row: row['Cost'] / row['Clicks'] if row['Clicks'] > 0 else 0, axis=1)
df['CTR'] = df.apply(lambda row: row['Clicks'] / row['Impressions'] if row['Impressions'] > 0 else 0, axis=1)

# Drop leakage columns
df.drop(['Conversion', 'Conversion Rate'], axis=1, errors='ignore', inplace=True)


# ==============================
# 📊 STEP 4: Exploratory Data Analysis (EDA)
# ==============================

# Conversion distribution
sns.countplot(x='High_Conversion', data=df)
plt.title("Conversion Distribution")
plt.show()

# CTR vs Conversion
sns.boxplot(x='High_Conversion', y='CTR', data=df)
plt.title("CTR vs Conversion")
plt.show()

# CPC vs Conversion
sns.boxplot(x='High_Conversion', y='CPC', data=df)
plt.title("CPC vs Conversion")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()


# ==============================
# 🔤 STEP 5: Encoding Categorical Variables
# ==============================

# One-Hot Encoding (better than LabelEncoder)
df = pd.get_dummies(df, columns=['Campaign_Name', 'Device', 'Location'], drop_first=True)
# ==============================
# 🧹 FINAL NaN CHECK & FIX
# ==============================

print("\nMissing values before fix:\n", df.isnull().sum())

# Fill any remaining NaN with 0
df.fillna(0, inplace=True)

print("\nMissing values after fix:\n", df.isnull().sum())

# ==============================
# 🧪 STEP 6: Train-Test Split
# ==============================

X = df.drop(['Ad_ID', 'Ad_Date', 'Keyword', 'High_Conversion'], axis=1)
y = df['High_Conversion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# 🤖 STEP 7: Train Models
# ==============================

# Logistic Regression (Baseline Model)
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Random Forest (Advanced Model)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)


# ==============================
# 📈 STEP 8: Model Evaluation
# ==============================

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n🔹 {model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(pd.Series(y_pred).value_counts())

# Evaluate both models
evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
evaluate_model(rf_model, X_test, y_test, "Random Forest")


# ==============================
# 📊 STEP 9: Feature Importance (Random Forest)
# ==============================

importances = rf_model.feature_importances_
features = X.columns

feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# ==============================
# 🔮 STEP 10: Prediction Function
# ==============================

def predict_conversion(model, input_df):
    """
    Predict whether an ad will convert (0 or 1)
    """
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    return prediction, probability


# ==============================
# ✅ DONE: Project Complete
# ==============================
