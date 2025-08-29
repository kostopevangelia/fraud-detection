# -*- coding: utf-8 -*-
"""
Train Custom Fraud Model
"""

import argparse
import os
import pandas as pd
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


def main(data_path: str, outdir: str):
    # 1. Load dataset
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Feature Engineering
    df['bin'] = df['cardNumber'].astype(str).str[:6]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Drop unused columns
    df = df.drop(columns=['transactionId', 'cardNumber', 'timestamp'])

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['currency', 'paymentType', 'transactionType', 'userId', 'bin'])

    # 3. Split X and y
    X = df_encoded.drop(columns=['fraud'])
    y = df_encoded['fraud']

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 5. Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 6. Train model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_res, y_res)

    # 7. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print(f"🔎 AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # 8. Save artifacts
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, "fraud_model_custom.pkl")
    feats_path = os.path.join(outdir, "custom_model_features.pkl")

    joblib.dump(model, model_path)
    with open(feats_path, "wb") as f:
        pickle.dump(X.columns, f)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Features saved to: {feats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom fraud detection model")
    parser.add_argument(
        "--data",
        default=os.path.join("resources", "custom_transactions.csv"),
        help="Path to the training dataset CSV"
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("models"),
        help="Directory to save model artifacts"
    )

    args = parser.parse_args()
    main(args.data, args.outdir)
