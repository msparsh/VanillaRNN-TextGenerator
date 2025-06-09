import numpy as np
import tensorflow as tf


class LayerRNN(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LayerRNN, self).__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.W_x = self.add_weight(
            shape=(input_dim, self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.W_h = self.add_weight(
            shape=(self.units, self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs, hidden_state):
        x = tf.matmul(inputs, self.W_x)  
        h = tf.matmul(hidden_state, self.W_h)
        hidden_state = tf.tanh(x + h + self.b)

        return hidden_state
    



class VanillaRNN(tf.keras.Model):
    def __init__(self, n_hidden_layers: int, units_per_layer: list, vocab_size : int):
        super(VanillaRNN, self).__init__()

        self.layers_list = [LayerRNN(units) for units in units_per_layer]
        self.output_layer = tf.keras.layers.Dense(units = vocab_size, activation='softmax')
        
    def call(self, inputs, epochs=1):
        batch_size = tf.shape(inputs)[0]
        time_steps = inputs.shape[1]

        hidden_states = [tf.zeros((batch_size, layer.units)) for layer in self.layers_list]
        
        for epoch in range(epochs):
            outputs = []
            for t in range(time_steps):
                x = inputs[:, t, :]  
                for i, layer in enumerate(self.layers_list):
                    hidden_states[i] = layer(x, hidden_states[i])
                    x = hidden_states[i]
                outputs.append(self.output_layer(x))

        # Stack outputs to create a tensor of shape (batch_size, time_steps - 1, vocab_size)
        return tf.stack(outputs, axis=1)

