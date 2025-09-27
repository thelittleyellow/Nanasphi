import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

num_features = 20   # jumlah fitur
num_classes = 10    # 10 kelas
num_train = 1000
num_test = 200

# --- Data Training ---
X_train = np.random.rand(num_train, num_features)
y_train = np.random.randint(0, num_classes, size=(num_train,))

pd.DataFrame(X_train).to_csv("data/train_X.csv", index=False)
pd.DataFrame(y_train).to_csv("data/train_label.csv", index=False)

# --- Data Testing ---
X_test = np.random.rand(num_test, num_features)
y_test = np.random.randint(0, num_classes, size=(num_test,))

pd.DataFrame(X_test).to_csv("data/test_X.csv", index=False)
pd.DataFrame(y_test).to_csv("data/test_label.csv", index=False)

print("Dummy CSV untuk train/test berhasil dibuat di folder 'data/'")
