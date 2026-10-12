import unittest

import complexnn as conn
import numpy as np
import tensorflow as tf


class TestLayersInSequential(unittest.TestCase):
    """Each layer builds, predicts and trains inside a Sequential model (issue #23)"""

    def check(self, layers):
        model = tf.keras.Sequential(layers)
        model.compile(optimizer="adam", loss="mse")
        shape = (4, *model.input_shape[1:])
        x = np.random.default_rng(0).standard_normal(shape).astype("float32")
        y = model.predict(x, verbose=0)
        model.fit(x, y, epochs=1, verbose=0)

    def test_complex_dense(self):
        self.check([tf.keras.Input((16,)), conn.ComplexDense(8)])

    def test_complex_conv1d(self):
        self.check([tf.keras.Input((32, 4)), conn.ComplexConv1D(4, 3, padding="same")])

    def test_complex_conv2d(self):
        self.check(
            [tf.keras.Input((16, 16, 4)), conn.ComplexConv2D(4, 3, padding="same")]
        )

    def test_complex_batch_normalization(self):
        self.check([tf.keras.Input((16,)), conn.ComplexBN()])

    def test_complex_layer_normalization(self):
        self.check([tf.keras.Input((16,)), conn.ComplexLayerNorm()])
