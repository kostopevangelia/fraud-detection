# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:43:22 2025

@author: kostopevangelia

Fraud Api
"""
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

import os

model_path = os.path.join("src", "models")

from fastapi import FastAPI
import pandas as pd

import os, joblib, pickle

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

model = joblib.load(os.path.join(MODEL_DIR, "fraud_model_custom.pkl"))
with open(os.path.join(MODEL_DIR, "custom_model_features.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

# Create the api

### this object handles incoming HTTP requests
app = FastAPI()

### We define a class that validates the input data that the API will receive (this class matches the transaciton format)

from pydantic import BaseModel, Field
from typing import Literal


class Transaction(BaseModel):
    amount: float = Field(..., gt=0)
    currency: Literal["USD", "EUR", "GBP"]
    paymentType: Literal["card", "bank_transfer"]
    transactionType: Literal["PAYMENT", "TRANSFER", "WITHDRAWAL"]
    userId: str
    bin: str
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)


## We create the prediction endpoint

@app.post("/predict-fraud")
def predict(transaction: Transaction):
    logger.info("Received transaction: %s", transaction.dict())

    # Step 1: Convert to DataFrame
    data = pd.DataFrame([transaction.dict()])

    # Step 2: One-hot encode and align columns
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)

    # Step 3: Predict fraud probability and label
    fraud_score = model.predict_proba(data_encoded)[:, 1][0]
    fraud_label = model.predict(data_encoded)[0]

    logger.info(f"Predicted fraud score: {fraud_score:.4f}, label: {fraud_label}")

    return {
        "fraudScore": round(fraud_score, 4),
        "fraud": bool(fraud_label)
    }
