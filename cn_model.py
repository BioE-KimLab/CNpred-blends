from typing import Tuple

import tensorflow as tf


class CNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def build(self, input_shape: Tuple[int]):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.L1(1E-6),
            trainable=True,
        )

    def call(self, x):
        return tf.einsum('ij,ij->i', x, tf.matmul(a=x, b=self.W))


class LowRankCNModel(CNModel):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def build(self, input_shape: Tuple[int]):
        self.U = self.add_weight(
            shape=(input_shape[-1], self.rank),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.L2(1E-6),
            trainable=True,
        )

        self.V = self.add_weight(
            shape=(self.rank, input_shape[-1]),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.L2(1E-6),
            trainable=True,
        )

    @property
    def W(self):
        return tf.matmul(a=self.U, b=self.V)
