import unittest

import complexnn as conn
import numpy as np
import tensorflow as tf

# Every working layer, with an input shape it accepts. Each entry builds a new
# layer, so tests never share an instance.
LAYERS = {
    "ComplexDense": ((8,), lambda: conn.ComplexDense(4)),
    "ComplexConv1D": ((10, 4), lambda: conn.ComplexConv1D(2, 3, padding="same")),
    "ComplexConv2D": ((6, 6, 4), lambda: conn.ComplexConv2D(2, 3, padding="same")),
    "ComplexConv2D normalize_weight": (
        (6, 6, 4),
        lambda: conn.ComplexConv2D(2, 3, padding="same", normalize_weight=True),
    ),
    "ComplexConv2D transposed": (
        (6, 6, 4),
        lambda: conn.ComplexConv2D(1, 3, strides=2, padding="same", transposed=True),
    ),
    "ComplexConv3D": ((4, 4, 4, 4), lambda: conn.ComplexConv3D(2, 3, padding="same")),
    "ComplexBN": ((4, 4, 8), lambda: conn.ComplexBN()),
    "ComplexLayerNorm": ((8,), lambda: conn.ComplexLayerNorm()),
    "LayerNormalization": ((8,), lambda: conn.LayerNormalization()),
    "GetReal": ((8,), lambda: conn.GetReal()),
    "GetImag": ((4, 4, 8), lambda: conn.GetImag()),
    "GetAbs": ((8,), lambda: conn.GetAbs()),
}


def build_model(shape, make_layer, rng):
    """A Sequential model around one layer, with every weight moved off its initial value"""
    model = tf.keras.Sequential([tf.keras.Input(shape), make_layer()])
    # With moved weights, matching outputs show that the weights were restored
    # and not just initialised again. The noise is small enough to keep the
    # normalisation variances positive.
    model.set_weights(
        [w + rng.uniform(-0.1, 0.1, w.shape) for w in model.get_weights()]
    )
    return model


class TestLayerConfig(unittest.TestCase):
    """Each working layer rebuilds from its config and computes the same output"""

    def test_from_config(self):
        rng = np.random.default_rng(0)
        for name, (shape, make_layer) in LAYERS.items():
            with self.subTest(name):
                layer = build_model(shape, make_layer, rng).layers[-1]
                config = layer.get_config()
                clone = type(layer).from_config(config)
                self.assertEqual(clone.get_config(), config)

                x = rng.standard_normal((3, *shape)).astype("float32")
                clone(x)  # builds the clone, so it can take the weights
                clone.set_weights(layer.get_weights())
                np.testing.assert_allclose(
                    np.asarray(clone(x)), np.asarray(layer(x)), rtol=1e-6, atol=1e-6
                )
