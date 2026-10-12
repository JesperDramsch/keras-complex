import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

import complexnn as conn


class TestTrainingRuns(unittest.TestCase):
    """Unit test class"""

    def test_train_transpose(self):
        """Train using Conv2DTranspose"""
        x = np.random.randn(64 * 64).reshape((64, 64))
        y = np.random.randn(64 * 64).reshape((64, 64))
        X = np.stack((x, y), -1)
        X = np.expand_dims(X, 0)
        Y = X
        inputs = Input(shape=(64, 64, 2))
        conv1 = conn.ComplexConv2D(
            filters=2,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=False,  # = 4 Keras filters
        )(inputs)
        outputs = conn.ComplexConv2D(
            filters=1,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True,  # = 2 Keras filters => 1 complex layer
        )(conv1)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
        model.fit(X, Y, batch_size=1, epochs=10)

    def test_train_dense(self):
        inputs = 28
        outputs = 128
        # build a sequential complex dense model
        model = Sequential(name="complex")
        model.add(conn.ComplexDense(32, activation="relu", input_shape=(inputs * 2,)))
        model.add(conn.ComplexBN())
        model.add(conn.ComplexDense(64, activation="relu"))
        model.add(conn.ComplexBN())
        model.add(Dense(128, activation="sigmoid"))
        model.compile(optimizer="adam", loss="mse")
        model.summary()
        # create some random data
        re = np.random.randn(inputs)
        im = np.random.randn(inputs)
        X = np.expand_dims(np.concatenate((re, im), -1), 0)
        Y = np.expand_dims(np.random.randn(outputs), 0)
        model.fit(X, Y, batch_size=1, epochs=10)
