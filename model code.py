import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# STEP 1: Load Dataset
df = pd.read_csv("GoogleAds.csv")

# STEP 2: Data Cleaning
df.drop_duplicates(inplace=True)

df['Cost'] = df['Cost'].replace('[\₹$,]', '', regex=True).astype(float)
df['Sale_Amount'] = df['Sale_Amount'].replace('[\₹$,]', '', regex=True).astype(float)

df['Cost'].fillna(df['Cost'].median(), inplace=True)
df['Sale_Amount'].fillna(df['Sale_Amount'].median(), inplace=True)
df['Clicks'].fillna(0, inplace=True)
df['Impressions'].fillna(0, inplace=True)
df['Conversions'].fillna(0, inplace=True)

df['Ad_Date'] = pd.to_datetime(df['Ad_Date'], errors='coerce')

df['Device'] = df['Device'].str.strip().str.title()
df['Location'] = df['Location'].str.strip().str.title()
df['Campaign_Name'] = df['Campaign_Name'].str.strip().str.title()

# STEP 3: Feature Engineering
df['Converted'] = df['Conversions'].apply(lambda x: 1 if x > 0 else 0)

df['CPC'] = df.apply(lambda row: row['Cost'] / row['Clicks'] if row['Clicks'] > 0 else 0, axis=1)
df['CTR'] = df.apply(lambda row: row['Clicks'] / row['Impressions'] if row['Impressions'] > 0 else 0, axis=1)

# Drop columns that cause leakage
df.drop(['Conversion Rate', 'Conversions'], axis=1, errors='ignore', inplace=True)

# STEP 4: Encode categorical variables
cat_cols = ['Campaign_Name', 'Device', 'Location']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# STEP 5: Train-Test Split
X = df.drop(['Ad_ID', 'Ad_Date', 'Keyword', 'Converted'], axis=1)
y = df['Converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 6: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# STEP 7: Predictions & Evaluation
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nRandom Forest Classifier Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# STEP 8: Feature Importance
importances = rf_model.feature_importances_
features = X.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance (Random Forest)")
plt.show()
