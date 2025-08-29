# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:11:48 2025

@author: kostopevangelia

Card Fraud Baseline
"""

import logging

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


import pandas as pd

# Load data
logger.info("Loading Data")
df = pd.read_csv("creditcard.csv")

logger.info("Dataset Info")
print(df.info())
print(df['Class'].value_counts(normalize=True))  # Imbalance check

logger.info("Visualization")
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal")
plt.show()

sns.histplot(df['Amount'], bins=50)

logger.info("Feature Scaling")
from sklearn.preprocessing import StandardScaler

df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df['Time'] = StandardScaler().fit_transform(df[['Time']])

logger.info("Train Test Split")
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

logger.info("Oversampling with SMOTE")
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

print(pd.Series(y_resampled).value_counts())

logger.info("Model Definitions")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=50, verbose=1),
    #"SVM": SVC(probability=True)
}


logger.info("Training and evaluation loop")
for name, model in models.items():
    logger.info(f" Starting training for model: {name}")
    
    model.fit(X_resampled, y_resampled)
    logger.debug(f"{name} training completed ✅")
    
    logger.debug(f"{name} predicting on test data...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"\n {name} results:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"AUC: {auc:.4f}")

