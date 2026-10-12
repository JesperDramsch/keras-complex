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
