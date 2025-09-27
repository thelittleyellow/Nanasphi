import tensorflow as tf
import pandas as pd
from src.model_builder import Nanasphi
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# --- Load test data ---
X_test = pd.read_csv("data/test_X.csv").values
y_test = pd.read_csv("data/test_label.csv").values.squeeze().astype(int)

input_dim = X_test.shape[1]
num_classes = len(set(y_test))

# --- Build model ---
model = Nanasphi(input_dim=input_dim, hidden_units=[128,64], output_units=num_classes)
model.load_weights("models/my_model.weights.h5")

# --- Prediksi ---
y_pred_prob = model.predict(X_test)
y_pred_class = np.argmax(y_pred_prob, axis=1)

# --- Hitung metrik ---
acc = accuracy_score(y_test, y_pred_class)
try:
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
except:
    auc = None

print(f"Test Accuracy: {acc:.4f}")
if auc:
    print(f"Test AUC: {auc:.4f}")
