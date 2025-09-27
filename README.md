{
  "project_title": "NP-PROJECT: Image Classification with TensorFlow",
  "short_description": "Proyek ini adalah implementasi sistem Deep Learning menggunakan TensorFlow/Keras untuk **mengenali angka tulisan tangan** dari gambar. Tujuannya adalah melatih model untuk mengklasifikasikan 10 kelas angka (0-9).",
 "project_structure": {
    "root_directory": "NP-PROJECT/",
    "directories": [
      { "name": ".venv/", "description": "Python virtual environment." },
      { "name": "data/", "description": "Berisi data gambar angka yang sudah di-preprocess dan di-split (e.g., fitur piksel dan label)." },
      { "name": "models/", "description": "Model TensorFlow/Keras yang telah dilatih (misalnya, arsitektur CNN atau Dense Net)." },
      { "name": "src/", "description": "Modul Python inti untuk pengelolaan data dan model." }
    ],
    "files": [
      { "name": "data/train_X.csv", "description": "Data piksel gambar (fitur X) untuk training set." },
      { "name": "data/train_label.csv", "description": "Label angka (Y) untuk training set." },
      { "name": "data/test_X.csv", "description": "Data piksel gambar (fitur X) untuk test set." },
      { "name": "data/test_label.csv", "description": "Label angka (Y) untuk test set." },
      { "name": "src/data_loader.py", "description": "Skrip untuk loading dan normalisasi data gambar (misalnya, data MNIST)." },
      { "name": "src/model_builder.py", "description": "Definisi arsitektur Neural Network (misalnya, model Convolutional Neural Network/CNN)." },
      { "name": "config.yaml", "description": "File konfigurasi utama untuk parameter pelatihan (epochs, batch size, learning rate)." },
      { "name": "train.py", "description": "Skrip utama untuk menjalankan proses pelatihan model pada data gambar angka." },
      { "name": "evaluate.py", "description": "Skrip utama untuk menguji akurasi model pada test set." },
      { "name": "plot_predictions.py", "description": "Skrip untuk menampilkan contoh gambar beserta prediksi model (misalnya, prediksi benar/salah)." },
      { "name": "create_dummy_data.py", "description": "Utilitas untuk membuat data sintetis atau subset kecil untuk pengujian." }
    ]
  },
  "usage_guide": {
    "step_1_setup": {
      "title": "1. Penyiapan Lingkungan",
      "actions": [
        "Clone Repositori.",
        "Pastikan data gambar angka (misalnya, MNIST) telah diolah dan disimpan dalam format CSV di `data/`.",
        "Buat Virtual Environment dan aktifkan.",
        "Install Libraries (`pip install tensorflow keras` dan `pip install -r requirements.txt`)."
      ]
    },
    "step_2_training": {
      "title": "2. Menjalankan Pelatihan Model",
      "command": "python train.py",
      "notes": "Model akan dilatih sesuai parameter di `config.yaml` dan disimpan di `models/`."
    },
    "step_3_evaluation": {
      "title": "3. Menjalankan Evaluasi dan Visualisasi",
      "commands": [
        { "cmd": "python evaluate.py", "desc": "Menghitung metrik (akurasi, loss) pada test set." },
        { "cmd": "python plot_predictions.py", "desc": "Memvisualisasikan hasil prediksi pada sampel data." }
      ]
    }
  }
}
