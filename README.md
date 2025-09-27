# NP-PROJECT: Image Classification with TensorFlow 

Proyek ini adalah sistem **Deep Learning** yang dirancang untuk mengenali angka tulisan tangan dari gambar, mirip dengan tugas klasifikasi **MNIST**.  
Kami menggunakan **TensorFlow** dan **Keras API** untuk membangun, melatih, dan mengevaluasi model **Neural Network** (biasanya *Convolutional Neural Network* atau CNN) untuk mengklasifikasikan 10 kelas angka (0–9).

---

##  Referensi & Teknologi

### Dataset
Proyek ini mengasumsikan penggunaan **MNIST Data Set** atau dataset gambar angka tulisan tangan serupa.  
Data harus diolah menjadi format **CSV**.

### Teknologi Inti
- **Deep Learning Stack**: TensorFlow & Keras  
- **Bahasa Pemrograman**: Python  

---

## Struktur Proyek
Struktur ini memisahkan logika data, model, dan eksekusi utama, mengikuti kerangka kerja **MLOps dasar**:

```
TF-PROJECT/
├── .venv/                     # Python virtual environment
├── data/                      # Data gambar angka yang sudah di-preprocess dan di-split
│   ├── test_label.csv         # Label angka (Y) untuk test set
│   ├── test_X.csv             # Data piksel gambar (fitur X) untuk test set
│   ├── train_label.csv        # Label angka (Y) untuk training set
│   └── train_X.csv            # Data piksel gambar (fitur X) untuk training set
├── models/                    # Model TensorFlow/Keras yang telah dilatih (.h5 atau SavedModel)
├── src/                       # Modul Python inti
│   ├── data_loader.py         # Skrip untuk loading, normalisasi, dan persiapan data
│   └── model_builder.py       # Definisi arsitektur Neural Network (misalnya, CNN)
├── config.yaml                # File konfigurasi utama untuk hyperparameters (epochs, batch size, dll.)
├── create_dummy_data.py       # Utilitas untuk membuat subset data kecil untuk pengujian
├── evaluate.py                # Skrip utama untuk menguji akurasi model pada test set
├── plot_predictions.py        # Skrip untuk visualisasi hasil prediksi pada sampel data
└── train.py                   # Skrip utama untuk menjalankan proses pelatihan model
```

---

## Panduan Penggunaan

### 1. Penyiapan Lingkungan

Clone repositori:

```bash
git clone [URL_REPOSITORI_ANDA]
cd TF-PROJECT
```

Siapkan data:  
Pastikan Anda telah mengunduh data gambar angka (misalnya MNIST) dan menjalankannya melalui preprocessing untuk menghasilkan empat file CSV di direktori `data/`.

Install libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Aktifkan virtual environment
pip install tensorflow keras
pip install -r requirements.txt  # Install dependencies lainnya
```

---

### 2. Menjalankan Pelatihan (Training)

Pelatihan model dilakukan dengan menjalankan skrip `train.py`.  
Skrip ini akan membaca semua parameter eksperimen penting dari file `config.yaml`.

```bash
python train.py
```

Model terbaik akan disimpan secara otomatis di direktori `models/`.

---

### 3. Menjalankan Evaluasi & Visualisasi

Setelah model dilatih, gunakan skrip berikut untuk menguji performanya dan melihat hasilnya:

| Skrip                   | Fungsi                                                                 |
|--------------------------|------------------------------------------------------------------------|
| `python evaluate.py`     | Menghitung metrik performa akhir (akurasi, loss) pada test set.        |
| `python plot_predictions.py` | Memuat model yang telah dilatih dan menampilkan beberapa gambar dengan hasil prediksi model (cocok/tidak cocok). |

---
