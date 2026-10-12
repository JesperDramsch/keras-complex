import unittest

import complexnn as conn
from tensorflow.keras.layers import Input

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

    def test_conv_transposed_odd_channels(self):
        layer = conn.ComplexConv2D(2, 3, strides=2, padding="same", transposed=True)
        self.assertLayoutError(layer, (8, 8, 3), "axis -1 has size 3")

    def test_dense(self):
        self.assertLayoutError(conn.ComplexDense(4), (5,), "axis -1 has size 5")
        self.assertLayoutError(conn.ComplexDense(4), (1,), "axis -1 has size 1")

    def test_dense_undefined_size(self):
        self.assertLayoutError(
            conn.ComplexDense(4), (None,), "the size of axis -1 is not defined"
        )

    def test_batch_normalization(self):
        self.assertLayoutError(conn.ComplexBN(), (5,), "axis -1 has size 5")
        self.assertLayoutError(conn.ComplexBN(), (8, 8, 1), "axis -1 has size 1")
        self.assertLayoutError(
            conn.ComplexBN(axis=1), (3, 8, 8), "axis 1 has size 3", "`axis` argument"
        )

    def test_layer_normalization(self):
        self.assertLayoutError(conn.ComplexLayerNorm(), (3,), "axis -1 has size 3")
        self.assertLayoutError(conn.ComplexLayerNorm(), (8, 1), "axis -1 has size 1")

    def test_get_parts(self):
        for layer_class in (conn.GetReal, conn.GetImag, conn.GetAbs):
            with self.subTest(layer=layer_class.__name__):
                self.assertLayoutError(layer_class(), (3,), "axis 1 has size 3")
                self.assertLayoutError(layer_class(), (8, 5), "axis -1 has size 5")

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


if __name__ == "__main__":
    unittest.main()
