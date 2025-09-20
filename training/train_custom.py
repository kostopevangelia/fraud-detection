# -*- coding: utf-8 -*-
"""
Train Custom Fraud Model
"""

import argparse
import os
import pandas as pd
import joblib
import pickle
import numpy as np
import time

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    precision_recall_curve
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")   # no Tkinter
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


def main(data_path: str, outdir: str, fast: bool = False):
    t0_total = time.perf_counter()

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
    df_encoded = pd.get_dummies(
        df,
        columns=['currency', 'paymentType', 'transactionType', 'userId', 'bin']
    )

    # 3. Split X and y
    X = df_encoded.drop(columns=['fraud'])
    y = df_encoded['fraud']

    # 4. Train/test split (stratified to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # class weights for the classifier
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    w_map = {0: float(weights[0]), 1: float(weights[1])}

    # 5. Build pipeline: SMOTE (only on train folds) + RandomForest
    pipe = Pipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight=w_map,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 6. Fit model
    pipe.fit(X_train, y_train)

    # 7. Evaluation
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # --- pick threshold by maximizing F1 ---
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1_vals = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = np.nanargmax(f1_vals)
    best_thr = float(thr[best_idx])
    y_pred = (y_prob >= best_thr).astype(int)

    print(f" Chosen threshold: {best_thr:.3f}")
    print("\nClassification Report (custom threshold):\n",
          classification_report(y_test, y_pred, zero_division=0))

    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, zero_division=0))
    print(f" AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # 8. Save artifacts
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, "fraud_model_custom.pkl")
    feats_path = os.path.join(outdir, "custom_model_features.pkl")

    joblib.dump(pipe, model_path)
    with open(feats_path, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Features saved to: {feats_path}")

    # 9. Visualization reports
    reports_dir = os.path.join(outdir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    def _savefig(path):
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    # Progress prints + reports
    print("[reports] feature importance…")
    try:
        clf = pipe.named_steps["clf"]
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(X.columns)[idx][::-1], importances[idx][::-1])
        plt.xlabel("Importance")
        plt.title("Top-20 Feature Importances (RandomForest)")
        _savefig(os.path.join(reports_dir, "feature_importance.png"))
    except Exception as e:
        print(f"[feature_importance skipped] {e}")

    print("[reports] confusion matrix…")
    try:
        plt.figure(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False)
        plt.title("Confusion Matrix (Test)")
        _savefig(os.path.join(reports_dir, "confusion_matrix.png"))
    except Exception as e:
        print(f"[confusion_matrix skipped] {e}")

    print("[reports] ROC curve…")
    try:
        plt.figure(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC Curve (Test)")
        _savefig(os.path.join(reports_dir, "roc_curve.png"))
    except Exception as e:
        print(f"[roc_curve skipped] {e}")

    print("[reports] Precision–Recall curve…")
    try:
        plt.figure(figsize=(5, 5))
        PrecisionRecallDisplay.from_predictions(y_test, y_prob)
        plt.title("Precision–Recall Curve (Test)")
        _savefig(os.path.join(reports_dir, "pr_curve.png"))
    except Exception as e:
        print(f"[pr_curve skipped] {e}")

    print("[reports] threshold trade-off plot…")
    try:
        plt.figure(figsize=(7, 5))
        plt.plot(thr, prec[:-1], label="Precision")
        plt.plot(thr, rec[:-1], label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision/Recall vs Threshold")
        plt.legend()
        _savefig(os.path.join(reports_dir, "threshold_tradeoff.png"))
    except Exception as e:
        print(f"[threshold_tradeoff skipped] {e}")

    # Heavy bits (controlled by --fast)
    if fast:
        print("[reports] fast mode enabled → skipping learning curve & SHAP.")
        return

    print("[reports] learning curve… (this may take a while)")
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        t0 = time.perf_counter()
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X_train, y_train,
            cv=cv, scoring="f1",
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        elapsed = time.perf_counter() - t0
        print(f"[reports] learning curve finished in {elapsed:.2f}s")

        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="CV F1")
        plt.xlabel("Training samples")
        plt.ylabel("F1 score")
        plt.title("Learning Curve (SMOTE + RandomForest Pipeline)")
        plt.legend()
        _savefig(os.path.join(reports_dir, "learning_curve.png"))
    except Exception as e:
        print(f"[learning_curve skipped] {e}")


    print("[reports] SHAP summary…")
    try:
        import shap

        t0 = time.perf_counter()

        # --- sample background and test data (must have same columns as training) ---
        bg_n = min(200, len(X_train))
        background_df = X_train.sample(n=bg_n, random_state=42)

        samp_n = min(200, len(X_test))
        X_test_sample_df = X_test.sample(n=samp_n, random_state=42)

        # --- convert to float64 and sanitize data ---
        background_df = background_df.astype(np.float64)
        X_test_sample_df = X_test_sample_df.astype(np.float64)

        # replace NaN/Inf with 0.0 to avoid SHAP crashes
        background_df = background_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_test_sample_df = X_test_sample_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # convert to contiguous numpy arrays for TreeExplainer
        background = np.ascontiguousarray(background_df.values, dtype=np.float64)
        X_test_sample = np.ascontiguousarray(X_test_sample_df.values, dtype=np.float64)
        feature_names = list(X_train.columns)

        # --- create explainer for the RandomForest classifier (not the whole pipeline) ---
        clf = pipe.named_steps["clf"]
        explainer = shap.TreeExplainer(
            clf,
            data=background,                        # background data for interventional mode
            feature_perturbation="interventional",
            model_output="probability",
        )

        shap_vals = explainer.shap_values(X_test_sample, check_additivity=False)

        # handle SHAP output type: list per class or single ndarray
        if isinstance(shap_vals, list):
            shap_to_plot = shap_vals[1]  # positive class
        else:
            shap_to_plot = shap_vals

        plt.figure()
        shap.summary_plot(
            shap_to_plot,
            features=X_test_sample,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        _savefig(os.path.join(reports_dir, "shap_summary.png"))

        print(f"[reports] SHAP finished in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        print(f"[SHAP skipped] {e}")




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
    # (4) CLI flag για γρήγορη εκτέλεση
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip heavy reports (learning curve & SHAP) and finish faster"
    )

    args = parser.parse_args()
    # (3) Progress prints υπάρχουν στα blocks παραπάνω
    main(args.data, args.outdir, fast=args.fast)
