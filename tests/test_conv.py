import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

import complexnn as conn


class TestConvMethods(unittest.TestCase):
    """Test Conv methods class"""

    def test_conv_outputs_forward(self):
        """Test computed shape of forward convolution output"""
        layer = conn.ComplexConv2D(
            filters=4, kernel_size=3, strides=2, padding="same", transposed=False
        )
        input_shape = (None, 128, 128, 2)
        true = (None, 64, 64, 8)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_outputs_transpose(self):
        """Test computed shape of transposed convolution output"""
        layer = conn.ComplexConv2D(
            filters=2, kernel_size=3, strides=2, padding="same", transposed=True
        )
        input_shape = (None, 64, 64, 4)
        true = (None, 128, 128, 4)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_conv2D_forward(self):
        """Test shape of model output, forward"""
        inputs = Input(shape=(128, 128, 2))
        outputs = conn.ComplexConv2D(
            filters=4, kernel_size=3, strides=2, padding="same", transposed=False
        )(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 64, 64, 8)
        calc = model.output_shape
        self.assertEqual(true, calc)

    def test_conv2Dtranspose(self):
        """Test shape of model output, transposed"""
        inputs = Input(shape=(64, 64, 20))  # = 10 CDN filters
        outputs = conn.ComplexConv2D(
            filters=2,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True,  # = 4 Keras filters
        )(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 128, 128, 4)
        calc = model.output_shape
        self.assertEqual(true, calc)
