import unittest
import warnings

import tensorflow as tf

import complexnn as conn
from complexnn.utils import _ignore_input_shape_warning

# Keras 3 warns with this message when a layer receives input_shape or input_dim.
INPUT_SHAPE_WARNING = "Do not pass an `input_shape`/`input_dim` argument"


def input_shape_warnings(make_layer):
    """Return the input_shape warnings that make_layer() raises"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_layer()
    return [str(w.message) for w in caught if INPUT_SHAPE_WARNING in str(w.message)]


class TestInputShapeArgument(unittest.TestCase):
    """Layers accept input_shape and input_dim without Keras 3's warning"""

    cases = {
        "ComplexDense input_dim": (lambda: conn.ComplexDense(8, input_dim=16), (16,)),
        "ComplexDense input_shape": (
            lambda: conn.ComplexDense(8, input_shape=(16,)),
            (16,),
        ),
        "ComplexConv1D": (
            lambda: conn.ComplexConv1D(2, 3, input_shape=(16, 4)),
            (16, 4),
        ),
        "ComplexConv2D": (
            lambda: conn.ComplexConv2D(2, 3, input_shape=(8, 8, 4)),
            (8, 8, 4),
        ),
        "ComplexConv3D": (
            lambda: conn.ComplexConv3D(2, 3, input_shape=(4, 4, 4, 2)),
            (4, 4, 4, 2),
        ),
        "ComplexBN": (lambda: conn.ComplexBN(input_shape=(16,)), (16,)),
        "ComplexLayerNorm": (lambda: conn.ComplexLayerNorm(input_shape=(16,)), (16,)),
        "LayerNormalization": (
            lambda: conn.LayerNormalization(input_shape=(16,)),
            (16,),
        ),
    }

    def test_no_warning(self):
        for name, (make_layer, _) in self.cases.items():
            with self.subTest(name):
                self.assertEqual(input_shape_warnings(make_layer), [])

    def test_sequential_uses_the_shape(self):
        for name, (make_layer, shape) in self.cases.items():
            with self.subTest(name):
                model = tf.keras.Sequential([make_layer()])
                self.assertEqual(model.input_shape, (None, *shape))


class TestIgnoreInputShapeWarning(unittest.TestCase):
    """The helper hides only the input_shape warning, on any Keras version"""

    def test_hides_only_the_input_shape_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _ignore_input_shape_warning():
                warnings.warn(f"{INPUT_SHAPE_WARNING} to a layer.", UserWarning)
                warnings.warn("Another warning.", UserWarning)
            warnings.warn(f"{INPUT_SHAPE_WARNING} outside the block.", UserWarning)
        self.assertEqual(
            [str(w.message) for w in caught],
            ["Another warning.", f"{INPUT_SHAPE_WARNING} outside the block."],
        )
