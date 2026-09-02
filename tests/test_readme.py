import unittest

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D

import complexnn as conn


class TestDNCMethods(unittest.TestCase):
    """Unit test class"""

    def test_github_example(self):
        # example from repository https://github.com/JesperDramsch/keras-complex/blob/master/README.md page
        model = tf.keras.models.Sequential()
        model.add(conn.conv.ComplexConv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 2)))
        model.add(conn.bn.ComplexBatchNormalization())
        model.add(MaxPooling2D((2, 2), padding="same"))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        model.summary()
