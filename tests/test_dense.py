import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

import complexnn as conn


class TestDenseMethods(unittest.TestCase):
    """Test Dense layer"""

    def test_outputs_dense(self):
        """Test computed shape of dense layer output"""
        layer = conn.ComplexDense(units=16, activation="relu")
        input_shape = (None, 8)
        true = (None, 16 * 2)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_call_multiplies_by_conjugate_kernel(self):
        """The output is the complex product of the input and the conjugate kernel"""
        # ComplexConv multiplies by its kernel, ComplexDense by the conjugate of
        # its kernel. This test records the current behaviour.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 6)).astype("float32")  # 3 real, then 3 imaginary
        layer = conn.ComplexDense(units=2, use_bias=False)
        y = layer(x).numpy()
        kernel = layer.real_kernel.numpy() + 1j * layer.imag_kernel.numpy()
        expected = (x[:, :3] + 1j * x[:, 3:]) @ np.conj(kernel)
        np.testing.assert_allclose(y[:, :2] + 1j * y[:, 2:], expected, atol=1e-5)


class TestDenseConfig(unittest.TestCase):
    """ComplexDense configs rebuild the same kernel initialiser"""

    def test_kernel_initializer_round_trip(self):
        """A kernel initialiser stored as a dict is read back, not hashed"""
        cases = [
            ("complex", str),
            ("glorot_uniform", tf.keras.initializers.GlorotUniform),
            (
                conn.ComplexInit(kernel_size=(4, 4), input_dim=4, weight_dim=2),
                conn.ComplexInit,
            ),
        ]
        for kernel_initializer, expected_type in cases:
            with self.subTest(expected_type.__name__):
                layer = conn.ComplexDense(4, kernel_initializer=kernel_initializer)
                config = layer.get_config()
                clone = conn.ComplexDense.from_config(config)
                self.assertIsInstance(clone.kernel_initializer, expected_type)
                self.assertEqual(clone.get_config(), config)
