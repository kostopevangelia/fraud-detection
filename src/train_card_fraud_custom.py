# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:28:10 2025

@author: user
"""

import logging

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

import joblib
import pandas as pd

# Load model and training features
model = joblib.load("fraud_model_custom.pkl")

# You must reload the same feature set (X.columns from training)
# If you saved X.columns to a file, load it here
import pickle
with open("custom_model_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

sample = {
    'amount': 3500.0,
    'currency': 'USD',
    'paymentType': 'card',
    'transactionType': 'PAYMENT',
    'userId': 'USER123',
    'bin': '400005',
    'hour': 2,
    'day_of_week': 6
}

df_sample = pd.DataFrame([sample])
df_sample_encoded = pd.get_dummies(df_sample)
df_sample_encoded = df_sample_encoded.reindex(columns=feature_columns, fill_value=0)

fraud_score = model.predict_proba(df_sample_encoded)[:, 1][0]
fraud_label = model.predict(df_sample_encoded)[0]

print(f" Fraud Score: {fraud_score:.4f}")
print(f" Fraud Detected? {'YES' if fraud_label == 1 else 'NO'}")
