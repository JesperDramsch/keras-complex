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

    def test_outputs_dense(self):
        """Test computed shape of dense layer output"""
        layer = conn.ComplexDense(units=16, activation="relu")
        input_shape = (None, 8)
        true = (None, 16 * 2)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)
