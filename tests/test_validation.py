import unittest

import numpy as np
from tensorflow.keras.layers import Input

import complexnn as conn

# Part of every error message about the real and imaginary layout.
LAYOUT_ERROR = "must be even and at least 2"


class TestComplexAxisValidation(unittest.TestCase):
    """Layers reject inputs whose complex axis cannot hold both parts (issue #15)"""

    def assertLayoutError(self, layer, shape, *fragments):
        """Calling `layer` on an input of `shape` raises the layout error."""
        with self.assertRaisesRegex(ValueError, LAYOUT_ERROR) as context:
            layer(Input(shape=shape))
        message = str(context.exception)
        self.assertIn(type(layer).__name__, message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_issue_15_channels_first(self):
        """The exact layer and input from issue #15"""
        layer = conn.ComplexConv3D(
            filters=2,
            kernel_size=(3, 3, 2),
            padding="same",
            data_format="channels_first",
        )
        self.assertLayoutError(
            layer,
            (1, 32, 32, 2),
            "axis 1 has size 1",
            "data_format is 'channels_first'",
        )

    def test_issue_15_valid_layout(self):
        """The same layer builds once the parts are on the channel axis"""
        layer = conn.ComplexConv3D(
            filters=2,
            kernel_size=(3, 3, 2),
            padding="same",
            data_format="channels_first",
        )
        outputs = layer(Input(shape=(2, 32, 32, 1)))
        self.assertEqual((None, 4, 32, 32, 1), tuple(outputs.shape))

    def test_conv_odd_channels(self):
        self.assertLayoutError(conn.ComplexConv1D(2, 3), (16, 3), "axis -1 has size 3")
        self.assertLayoutError(
            conn.ComplexConv2D(2, 3), (8, 8, 5), "axis -1 has size 5"
        )
        self.assertLayoutError(
            conn.ComplexConv3D(2, 3), (8, 8, 8, 3), "axis -1 has size 3"
        )

    def test_conv_too_few_channels(self):
        self.assertLayoutError(conn.ComplexConv1D(2, 3), (16, 1), "axis -1 has size 1")
        self.assertLayoutError(
            conn.ComplexConv2D(2, 3, data_format="channels_first"),
            (1, 8, 8),
            "axis 1 has size 1",
        )

    def test_conv_zero_channels(self):
        """A size of 0 is even, so only the "at least 2" condition rejects it"""
        self.assertLayoutError(conn.ComplexConv1D(2, 3), (16, 0), "axis -1 has size 0")
        self.assertLayoutError(
            conn.ComplexConv2D(2, 3), (8, 8, 0), "axis -1 has size 0"
        )
        self.assertLayoutError(
            conn.ComplexConv3D(2, 3), (8, 8, 8, 0), "axis -1 has size 0"
        )
        self.assertLayoutError(
            conn.ComplexConv2D(2, 3, data_format="channels_first"),
            (0, 8, 8),
            "axis 1 has size 0",
        )

    def test_conv_transposed(self):
        for size in (3, 1, 0):
            with self.subTest(size=size):
                layer = conn.ComplexConv2D(
                    2, 3, strides=2, padding="same", transposed=True
                )
                self.assertLayoutError(layer, (8, 8, size), f"axis -1 has size {size}")

    def test_dense(self):
        self.assertLayoutError(conn.ComplexDense(4), (5,), "axis -1 has size 5")
        self.assertLayoutError(conn.ComplexDense(4), (1,), "axis -1 has size 1")
        self.assertLayoutError(conn.ComplexDense(4), (0,), "axis -1 has size 0")

    def test_dense_undefined_size(self):
        self.assertLayoutError(
            conn.ComplexDense(4), (None,), "the size of axis -1 is not defined"
        )

    def test_batch_normalization(self):
        self.assertLayoutError(conn.ComplexBN(), (5,), "axis -1 has size 5")
        self.assertLayoutError(conn.ComplexBN(), (8, 8, 1), "axis -1 has size 1")
        self.assertLayoutError(conn.ComplexBN(), (8, 0), "axis -1 has size 0")
        self.assertLayoutError(
            conn.ComplexBN(axis=1),
            (3, 8, 8),
            "axis 1 has size 3",
            "`axis=1` when they are on axis 1 of an input of rank 2, 4 or 5",
        )

    def test_layer_normalization(self):
        self.assertLayoutError(conn.ComplexLayerNorm(), (3,), "axis -1 has size 3")
        self.assertLayoutError(conn.ComplexLayerNorm(), (8, 1), "axis -1 has size 1")
        self.assertLayoutError(conn.ComplexLayerNorm(), (8, 0), "axis -1 has size 0")
        self.assertLayoutError(
            conn.ComplexLayerNorm(axis=1),
            (3, 8, 8),
            "axis 1 has size 3",
            "`axis=1` when they are on axis 1 of an input of rank 2, 4 or 5",
        )

    def test_get_parts(self):
        for layer_class in (conn.GetReal, conn.GetImag, conn.GetAbs):
            with self.subTest(layer=layer_class.__name__):
                self.assertLayoutError(layer_class(), (3,), "axis 1 has size 3")
                self.assertLayoutError(layer_class(), (8, 5), "axis -1 has size 5")
                self.assertLayoutError(layer_class(), (1,), "axis 1 has size 1")
                self.assertLayoutError(layer_class(), (8, 0), "axis -1 has size 0")

    def test_get_parts_reused_instance(self):
        """A part layer checks every input, not only the one it was built on

        These layers have no weights, so one instance is often reused. Keras
        builds a layer once, so the check also runs in call and in
        compute_output_shape.
        """
        for layer_class in (conn.GetReal, conn.GetImag, conn.GetAbs):
            with self.subTest(layer=layer_class.__name__):
                layer = layer_class()
                layer(Input(shape=(8, 2)))
                with self.assertRaisesRegex(ValueError, LAYOUT_ERROR):
                    layer(Input(shape=(8, 3)))
                with self.assertRaisesRegex(ValueError, LAYOUT_ERROR):
                    layer(np.zeros((2, 8, 5), dtype="float32"))
                # A different valid width still works with the same instance.
                output = layer(Input(shape=(8, 4)))
                self.assertEqual(tuple(output.shape), (None, 8, 2))

    def test_get_parts_undefined_size(self):
        """An undefined size is left to the dynamic split in call"""
        for layer_class in (conn.GetReal, conn.GetImag, conn.GetAbs):
            with self.subTest(layer=layer_class.__name__):
                layer = layer_class()
                layer.build((None, None, None))
                self.assertTrue(layer.built)


class TestValidComplexAxis(unittest.TestCase):
    """The smallest valid layout, one complex channel, still builds"""

    def test_layers_build_with_two_channels(self):
        cases = [
            (conn.ComplexConv1D(2, 3), (16, 2), (None, 14, 4)),
            (conn.ComplexConv2D(2, 3), (8, 8, 2), (None, 6, 6, 4)),
            (conn.ComplexDense(3), (2,), (None, 6)),
            (conn.ComplexBN(), (8, 2), (None, 8, 2)),
            (conn.ComplexLayerNorm(), (8, 2), (None, 8, 2)),
            (conn.GetReal(), (8, 2), (None, 8, 1)),
            (conn.GetImag(), (8, 2), (None, 8, 1)),
            (conn.GetAbs(), (8, 2), (None, 8, 1)),
        ]
        for layer, shape, expected in cases:
            with self.subTest(layer=type(layer).__name__):
                outputs = layer(Input(shape=shape))
                self.assertEqual(expected, tuple(outputs.shape))

    def test_normalisation_supported_axes(self):
        """The `axis` settings that the docs list for the normalisation layers run"""
        rng = np.random.default_rng(15)
        cases = [
            (-1, (4, 6)),
            (-1, (4, 5, 6)),
            (-1, (4, 3, 3, 6)),
            (-1, (4, 3, 3, 3, 6)),
            (1, (4, 6)),
            (1, (4, 6, 3, 3)),
            (1, (4, 6, 3, 3, 3)),
        ]
        for layer_class in (conn.ComplexBN, conn.ComplexLayerNorm):
            for axis, shape in cases:
                with self.subTest(layer=layer_class.__name__, axis=axis, shape=shape):
                    x = rng.normal(size=shape).astype("float32")
                    outputs = np.asarray(layer_class(axis=axis)(x, training=True))
                    self.assertEqual(shape, outputs.shape)
                    self.assertTrue(np.all(np.isfinite(outputs)))


if __name__ == "__main__":
    unittest.main()
