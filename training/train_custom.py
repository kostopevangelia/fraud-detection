# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:01:03 2025

@author: kostopevangelia

Card Fraud Custom
"""

import logging

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

# Step 1: Load and inspect the data

import pandas as pd

df = pd.read_csv("custom_transactions.csv")

print(df.head())
print(df['fraud'].value_counts())


# Step 2: Feature Engineering
## we extract features like bin, hour, day of week, etc.

### extract bin
df['bin'] = df['cardNumber'].astype(str).str[:6]

### convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

### extract hour of day
df['hour'] = df['timestamp'].dt.hour

### extract day of the week
df['day_of_week'] = df['timestamp'].dt.dayofweek 

# Step 3: Drop unhelpful columns
## Some columns don't need to be fed into the model

df = df.drop(columns=['transactionId', 'cardNumber', 'timestamp'])


# Step 4: Encoding of categorical columns
df_encoded = pd.get_dummies(df, columns=['currency', 'paymentType', 'transactionType', 'userId', 'bin'])


# Step 5: We split X and Y
X = df_encoded.drop(columns=['fraud'])
y = df_encoded['fraud']


# Step 6: Split Train/Test sets
## we use an 80/20 split and stratify on the fraud columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 7: We are handling class imbalance with SMOTE

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

### check new class balance
print(y_resampled.value_counts())


# Step 8: We train the model
## We choose Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

## we predict on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 9: Model evaluation
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")


# Step 10: Export the model
import joblib

joblib.dump(model, "fraud_model_custom.pkl")
print("Model saved as fraud_model_custom.pkl")


import pickle
with open("custom_model_features.pkl", "wb") as f:
    pickle.dump(X.columns, f)
