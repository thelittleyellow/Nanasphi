import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model_builder import Nanasphi

# --- Fungsi untuk menampilkan satu prediksi ---
# Fungsi ini sudah benar, tidak perlu diubah.
def show_one_prediction(images, true_labels, pred_labels, index=0):
    plt.figure(figsize=(3,3))
    img = images[index]
    # dukung input vektor, (28,28,1), atau (28,28)
    if img.ndim == 1:
        img = img.reshape(28,28)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    # Pastikan scalar
    true_label = int(true_labels[index])
    pred_label = int(pred_labels[index])

    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"Pred: {pred_label}, Asli: {true_label}", color=color)
    plt.show()

# --- Load test data ---
X_test = pd.read_csv("data/test_X.csv")
y_test = pd.read_csv("data/test_label.csv").squeeze().to_numpy().astype(int)

# --- Tambahkan dummy columns jika kolom < 784 ---
if X_test.shape[1] < 784:
    n_missing = 784 - X_test.shape[1]
    dummy_cols = pd.DataFrame(np.zeros((X_test.shape[0], n_missing)),
                                  columns=[f"dummy_{i}" for i in range(n_missing)])
    X_test = pd.concat([X_test, dummy_cols], axis=1)

# --- Convert ke numpy array dan reshape ke gambar 28x28 ---
X_test_img = X_test.values.reshape(-1,28,28,1)

# --- Build model dan load weights ---
input_dim = X_test.shape[1]
num_classes = len(np.unique(y_test))
model = Nanasphi(input_dim=input_dim, hidden_units=[128,64], output_units=num_classes)
model.build(input_shape=(None, input_dim))
model.load_weights("models/my_model.weights.h5")

# --- Prediksi ---
y_pred_prob = model.predict(X_test.values)
y_pred_class = np.argmax(y_pred_prob, axis=1)

# --- KODE LAMA YANG MENAMPILKAN BANYAK GAMBAR ---
"""
num_show = 12  # jumlah sample
plt.figure(figsize=(12,6))
for i in range(num_show):
    plt.subplot(3,4,i+1)
    img = X_test_img[i]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    plt.imshow(img, cmap='gray')
    
    true_label = int(y_test[i])
    pred_label = int(y_pred_class[i])
    color = 'green' if pred_label == true_label else 'red'
    
    plt.title(f"P:{pred_label}\nT:{true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
"""

# 2. PANGGIL FUNGSI show_one_prediction YANG SUDAH ANDA BUAT
# Anda bisa mengubah nilai 'index' untuk melihat gambar yang berbeda.
# Misalnya, index=0 untuk gambar pertama, index=1 untuk gambar kedua, dst.
gambar_yang_ingin_dilihat = 2  # <-- Ubah angka ini untuk melihat gambar lain
show_one_prediction(X_test_img, y_test, y_pred_class, index=gambar_yang_ingin_dilihat)