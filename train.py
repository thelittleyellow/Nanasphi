from src.model_builder import Nanasphi
from src.data_loader import get_datasets
import tensorflow as tf
import os

# --- Buat folder models jika belum ada ---
os.makedirs("models", exist_ok=True)

# --- Load data ---
train_ds, test_ds, input_dim, num_classes = get_datasets(batch_size=32)

# --- Build model ---
model = Nanasphi(input_dim=input_dim, hidden_units=[128,64], output_units=num_classes)

# --- Compile ---
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train ---
history = model.fit(train_ds, epochs=10, validation_data=test_ds)

# --- Simpan weights ---
model.save_weights("models/my_model.weights.h5")
print("Model weights saved to models/my_model.weights.h5")
