import tensorflow as tf

class Nanasphi(tf.keras.Model):
    def __init__(self, input_dim, hidden_units=[128,64], output_units=2):
        super(Nanasphi, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]
        self.output_layer = tf.keras.layers.Dense(output_units, activation="softmax")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
