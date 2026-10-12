import inspect
import os
import tempfile
import unittest
import warnings

import complexnn as conn
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_registered_name, get_registered_object

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


class TestRegistration(unittest.TestCase):
    """Every class that complexnn exports is registered for serialisation"""

    def test_public_classes_are_registered(self):
        classes = [
            obj
            for obj in vars(conn).values()
            if inspect.isclass(obj) and obj.__module__.startswith("complexnn.")
        ]
        self.assertGreater(len(classes), 0)
        for cls in classes:
            with self.subTest(cls.__name__):
                registered_name = "complexnn>" + cls.__name__
                self.assertEqual(get_registered_name(cls), registered_name)
                self.assertIs(get_registered_object(registered_name), cls)


class TestKerasFileRoundTrip(unittest.TestCase):
    """Models with complex layers save to .keras and load without custom_objects"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def save_and_load(self, model, name):
        path = os.path.join(self.tmp, name + ".keras")
        with warnings.catch_warnings():
            # TensorFlow variables implement __array__ without the copy keyword
            # that numpy 2 passes, so Keras 3 warns while saving any model,
            # built-in layers included. The warning is not about complexnn.
            warnings.filterwarnings(
                "ignore",
                message="__array__ implementation doesn't accept a copy keyword",
                category=DeprecationWarning,
            )
            model.save(path)
        return tf.keras.models.load_model(path)

    def test_layers(self):
        rng = np.random.default_rng(1)
        for i, (name, (shape, make_layer)) in enumerate(LAYERS.items()):
            with self.subTest(name):
                model = build_model(shape, make_layer, rng)
                loaded = self.save_and_load(model, f"layer_{i}")
                self.assertIs(type(loaded.layers[-1]), type(model.layers[-1]))

                x = rng.standard_normal((3, *shape)).astype("float32")
                np.testing.assert_allclose(
                    loaded.predict(x, verbose=0),
                    model.predict(x, verbose=0),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_initializers(self):
        """A built-in layer keeps a complex initialiser through saving and loading"""
        initializers = [
            conn.ComplexInit(kernel_size=(4, 6), input_dim=4, weight_dim=2, seed=3),
            conn.IndependentFilters(
                kernel_size=(4, 6), input_dim=4, weight_dim=2, seed=3
            ),
            conn.ComplexIndependentFilters(
                kernel_size=(4, 3), input_dim=4, weight_dim=2, seed=3
            ),
        ]
        for i, init in enumerate(initializers):
            with self.subTest(type(init).__name__):
                dense = tf.keras.layers.Dense(6, kernel_initializer=init)
                model = tf.keras.Sequential([tf.keras.Input((4,)), dense])
                loaded = self.save_and_load(model, f"initializer_{i}")
                restored = loaded.layers[-1].kernel_initializer
                self.assertIs(type(restored), type(init))
                np.testing.assert_array_equal(restored((4, 6)), init((4, 6)))
