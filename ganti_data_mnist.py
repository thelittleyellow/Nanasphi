import tensorflow as tf
import pandas as pd
import numpy as np

print("🚀 Memulai proses pengunduhan dataset MNIST...")

# 1. Muat dataset MNIST standar dari TensorFlow
#    Ini akan mengunduh data jika belum ada di cache Anda.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("✅ Dataset MNIST berhasil diunduh.")
print(f"   - Data Latih (Train): {x_train.shape[0]} gambar")
print(f"   - Data Uji (Test): {x_test.shape[0]} gambar")

# 2. Siapkan data untuk format CSV
#    Setiap gambar 28x28 piksel akan diratakan (flatten) menjadi satu baris berisi 784 kolom.
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train_flat = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test_flat = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

print("🔄 Mengubah format data gambar ke format CSV...")

# 3. Buat DataFrame menggunakan Pandas
df_train_x = pd.DataFrame(x_train_flat)
df_train_label = pd.DataFrame(y_train, columns=['label'])
df_test_x = pd.DataFrame(x_test_flat)
df_test_label = pd.DataFrame(y_test, columns=['label'])

# 4. Simpan ke file CSV dan timpa file yang sudah ada
#    Parameter index=False mencegah penambahan kolom indeks yang tidak perlu.
print("💾 Menyimpan dan menimpa file CSV yang ada...")

df_train_x.to_csv('train_X.csv', index=False)
df_train_label.to_csv('train_label.csv', index=False)
df_test_x.to_csv('test_X.csv', index=False)
df_test_label.to_csv('test_label.csv', index=False)

print("\n🎉 Berhasil! Semua file telah diganti dengan data MNIST yang asli.")