import unittest

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import complexnn as conn


def fit_one_step(layer, shape):
    """Build a model around `layer`, train it for one step and return its output"""
    inputs = Input(shape=shape)
    model = Model(inputs, layer(inputs))
    model.compile("sgd", "mse")
    data = np.ones((2, *shape), dtype="float32")
    model.fit(data, data, epochs=1, verbose=0)
    return model.predict(data, verbose=0)


class TestSpectralPooling1D(unittest.TestCase):
    """SpectralPooling1D builds and trains with topf= or gamma="""

    def test_topf(self):
        out = fit_one_step(conn.SpectralPooling1D(topf=(4,)), (16, 2))
        self.assertEqual(out.shape, (2, 16, 2))
        np.testing.assert_array_equal(out[0, :, 0], [1] * 2 + [0] * 12 + [1] * 2)

    def test_gamma(self):
        out = fit_one_step(conn.SpectralPooling1D(gamma=(0.5,)), (16, 2))
        self.assertEqual(out.shape, (2, 16, 2))
        np.testing.assert_array_equal(out[0, :, 0], [1] * 4 + [0] * 8 + [1] * 4)

    def test_forwards_layer_kwargs(self):
        layer = conn.SpectralPooling1D(topf=(4,), name="pool1d")
        self.assertEqual(layer.name, "pool1d")

    def test_needs_topf_or_gamma(self):
        with self.assertRaises(RuntimeError):
            conn.SpectralPooling1D()


class TestSpectralPooling2D(unittest.TestCase):
    """SpectralPooling2D builds and trains with topf= or gamma="""

    def test_topf(self):
        out = fit_one_step(conn.SpectralPooling2D(topf=(4, 4)), (8, 8, 2))
        self.assertEqual(out.shape, (2, 8, 8, 2))
        keep = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(out[0, :, :, 0], np.outer(keep, keep))

    def test_gamma(self):
        out = fit_one_step(conn.SpectralPooling2D(gamma=(0.5, 0.5)), (8, 8, 2))
        self.assertEqual(out.shape, (2, 8, 8, 2))
        keep = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(out[0, :, :, 0], np.outer(keep, keep))

    def test_forwards_layer_kwargs(self):
        layer = conn.SpectralPooling2D(gamma=(0.5, 0.5), name="pool2d")
        self.assertEqual(layer.name, "pool2d")

    def test_unknown_spatial_dim(self):
        inputs = Input(shape=(None, 8, 2))
        with self.assertRaises(ValueError):
            conn.SpectralPooling2D(topf=(4, 4))(inputs)


if __name__ == "__main__":
    unittest.main()
