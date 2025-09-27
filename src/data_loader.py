import pandas as pd
import tensorflow as tf

def get_datasets(batch_size=32):
    X_train = pd.read_csv("data/train_X.csv").values
    y_train = pd.read_csv("data/train_label.csv").values.squeeze().astype(int)

    X_test  = pd.read_csv("data/test_X.csv").values
    y_test  = pd.read_csv("data/test_label.csv").values.squeeze().astype(int)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    input_dim = X_train.shape[1]
    num_classes = len(set(y_train))

    return train_ds, test_ds, input_dim, num_classes
