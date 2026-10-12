import unittest

import complexnn as conn
import tensorflow as tf
from complexnn.bn import sqrt_init


class TestComplexLayerNormConfig(unittest.TestCase):
    """ComplexLayerNorm configs rebuild the same layer"""

    def test_from_config(self):
        layer = conn.ComplexLayerNorm()
        config = layer.get_config()
        clone = conn.ComplexLayerNorm.from_config(config)
        self.assertEqual(clone.get_config(), config)

    def test_sqrt_init_serialises_by_name(self):
        """The default gamma initialiser is stored by name and restored as sqrt_init"""
        config = conn.ComplexLayerNorm().get_config()
        self.assertEqual(config["gamma_diag_initializer"], "sqrt_init")
        clone = conn.ComplexLayerNorm.from_config(config)
        self.assertIs(clone.gamma_diag_initializer, sqrt_init)

    def test_keras_initializer_round_trip(self):
        """Keras initialisers still go through the Keras registry"""
        layer = conn.ComplexLayerNorm(gamma_diag_initializer="ones")
        clone = conn.ComplexLayerNorm.from_config(layer.get_config())
        self.assertIsInstance(clone.gamma_diag_initializer, tf.keras.initializers.Ones)


class TestLayerNormalizationConfig(unittest.TestCase):
    """LayerNormalization configs keep their initialisers and regularisers"""

    def test_from_config(self):
        layer = conn.LayerNormalization(
            beta_init="ones",
            gamma_init="zeros",
            gamma_regularizer=tf.keras.regularizers.L2(0.01),
        )
        config = layer.get_config()
        clone = conn.LayerNormalization.from_config(config)
        self.assertEqual(clone.get_config(), config)
        self.assertIsInstance(clone.beta_init, tf.keras.initializers.Ones)
        self.assertIsInstance(clone.gamma_init, tf.keras.initializers.Zeros)
        self.assertAlmostEqual(clone.gamma_regularizer.get_config()["l2"], 0.01)


class TestComplexLayerNormShape(unittest.TestCase):
    """ComplexLayerNorm keeps the input shape"""

    def test_compute_output_shape(self):
        layer = conn.ComplexLayerNorm()
        for shape in [(None, 8), (None, 4, 4, 8)]:
            with self.subTest(shape=shape):
                self.assertEqual(tuple(layer.compute_output_shape(shape)), shape)
