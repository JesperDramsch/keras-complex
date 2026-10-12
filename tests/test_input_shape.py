import unittest
import warnings

import tensorflow as tf

import complexnn as conn
from complexnn.utils import _KERAS_3, _ignore_input_shape_warning

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
    """The helper filters only the input_shape warning, and only when it can fire"""

    def warnings_shown(self, kwargs):
        """Warn inside and after the helper's block, and return what was shown"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _ignore_input_shape_warning(kwargs):
                warnings.warn(f"{INPUT_SHAPE_WARNING} to a layer.", UserWarning)
                warnings.warn("Another warning.", UserWarning)
            warnings.warn(f"{INPUT_SHAPE_WARNING} outside the block.", UserWarning)
        return [str(w.message) for w in caught]

    @unittest.skipUnless(_KERAS_3, "only Keras 3 emits the input_shape warning")
    def test_hides_only_the_input_shape_warning(self):
        for kwargs in ({"input_shape": (16,)}, {"input_dim": 16}):
            with self.subTest(kwargs):
                self.assertEqual(
                    self.warnings_shown(kwargs),
                    ["Another warning.", f"{INPUT_SHAPE_WARNING} outside the block."],
                )

    def test_filters_nothing_when_the_warning_cannot_fire(self):
        everything = [
            f"{INPUT_SHAPE_WARNING} to a layer.",
            "Another warning.",
            f"{INPUT_SHAPE_WARNING} outside the block.",
        ]
        cases = {"no input_shape": {"name": "layer"}}
        if not _KERAS_3:
            cases["Keras 2 input_shape"] = {"input_shape": (16,)}
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertEqual(self.warnings_shown(kwargs), everything)


REPEATED_WARNING = "A warning from one place in the code."


def warn_from_one_place():
    warnings.warn(REPEATED_WARNING, UserWarning)


class TestWarningsShownOnce(unittest.TestCase):
    """Building a layer does not make a warning that was shown once show again

    ``warnings.catch_warnings`` resets the record Python keeps of warnings it
    has shown, so the layers must not enter it when the input_shape warning
    cannot fire.
    """

    def count_shown(self, make_layer):
        """Warn from one place three times, building a layer after each warning"""
        with warnings.catch_warnings(record=True) as caught:
            # The default action shows a warning once per place in the code.
            warnings.simplefilter("default")
            for _ in range(3):
                warn_from_one_place()
                make_layer()
        return [str(w.message) for w in caught].count(REPEATED_WARNING)

    def test_warning_shown_once(self):
        cases = {
            "ComplexDense": lambda: conn.ComplexDense(2),
            "ComplexConv2D": lambda: conn.ComplexConv2D(2, 3),
            "ComplexBN": lambda: conn.ComplexBN(),
            "ComplexLayerNorm": lambda: conn.ComplexLayerNorm(),
            "LayerNormalization": lambda: conn.LayerNormalization(),
        }
        if not _KERAS_3:
            # Keras 2 never warns about input_dim, so the filters stay untouched.
            cases["ComplexDense input_dim"] = lambda: conn.ComplexDense(2, input_dim=4)
        for name, make_layer in cases.items():
            with self.subTest(name):
                self.assertEqual(self.count_shown(make_layer), 1)
