from typing import Optional

import tensorflow as tf


# Remove SampleModel and replace
class SampleModel(tf.keras.Model):
    """
    This is sample model.

    Arguments:
        hidden_dim: Integer, the hidden dimension size of SampleModel.

    Call arguments:
        inputs: A 2D tensor, with shape of `[BatchSize, FeatureSize]`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, HiddenDim]
    """

    def __init__(self, hidden_dim: int):
        super(SampleModel, self).__init__()

        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        output = self.dense2(self.dense1(inputs))
        return output
