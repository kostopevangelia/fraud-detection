import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("jainilcoder/online-payment-fraud-detection")

print("Path to dataset files:", path)

df = pd.read_csv(os.path.join(path, 'onlinefraud.csv'))

# Encode 'merchant_category' and 'transaction_type'
merchant_encoder = LabelEncoder()
transaction_encoder = LabelEncoder()


# Reshape and fit
X = df[['oldbalanceOrg']]
y = df[['newbalanceOrig']]

line_fitter = LinearRegression()
line_fitter.fit(X, y)

# Predict and plot
y_pred = line_fitter.predict(X)

plt.scatter(X, y, alpha=0.4, label='Original')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title("Linear Regression: oldbalanceOrg vs newbalanceOrig")
plt.xlabel("oldbalanceOrg")
plt.ylabel("newbalanceOrig")
plt.legend()
plt.show()
