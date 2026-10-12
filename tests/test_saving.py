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
# layer, so tests never share an instance. Layers that take arguments set some
# of them away from their defaults: a value that from_config does not restore
# then changes the config, and a lost argument that affects the output
# (activation, strides, dilation_rate, epsilon) changes the output. Arguments
# missing from the config altogether are caught by TestConfigCompleteness.
LAYERS = {
    "ComplexDense": (
        (8,),
        lambda: conn.ComplexDense(4, activation="tanh", init_criterion="glorot"),
    ),
    # A seeded Keras initialiser: Keras 2 warns when an unseeded one is called
    # twice, and ComplexDense calls it for the real and the imaginary kernel.
    "ComplexDense GlorotUniform": (
        (8,),
        lambda: conn.ComplexDense(
            4, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1)
        ),
    ),
    "ComplexDense ComplexInit": (
        (8,),
        lambda: conn.ComplexDense(
            4,
            kernel_initializer=conn.ComplexInit(
                kernel_size=(4, 4), input_dim=4, weight_dim=2
            ),
        ),
    ),
    "ComplexConv1D": (
        (10, 4),
        lambda: conn.ComplexConv1D(
            2, 3, padding="same", dilation_rate=2, activation="tanh"
        ),
    ),
    "ComplexConv2D": (
        (6, 6, 4),
        lambda: conn.ComplexConv2D(
            2, 3, strides=2, padding="same", init_criterion="glorot"
        ),
    ),
    "ComplexConv2D normalize_weight": (
        (6, 6, 4),
        lambda: conn.ComplexConv2D(
            2, 3, padding="same", normalize_weight=True, epsilon=1e-2
        ),
    ),
    "ComplexConv2D transposed": (
        (6, 6, 4),
        lambda: conn.ComplexConv2D(1, 3, strides=2, padding="same", transposed=True),
    ),
    "ComplexConv3D": (
        (4, 4, 4, 4),
        lambda: conn.ComplexConv3D(
            2, 3, padding="same", dilation_rate=2, use_bias=False
        ),
    ),
    "ComplexBN": ((4, 4, 8), lambda: conn.ComplexBN(epsilon=1e-2, momentum=0.5)),
    # On a 2D input both layer norms reduce over no axes and return beta, so
    # they get inputs with a feature axis on which the output depends on gamma.
    "ComplexLayerNorm": ((4, 4, 8), lambda: conn.ComplexLayerNorm(epsilon=1e-2)),
    "LayerNormalization": ((4, 8), lambda: conn.LayerNormalization(epsilon=1e-2)),
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


def init_parameters(cls):
    """Named __init__ arguments of cls and of the complexnn classes it passes them to"""
    names = set()
    for klass in cls.__mro__:
        if "__init__" in vars(klass) and klass.__module__.startswith("complexnn."):
            for param in inspect.signature(klass.__init__).parameters.values():
                if param.name != "self" and param.kind not in (
                    param.VAR_POSITIONAL,
                    param.VAR_KEYWORD,
                ):
                    names.add(param.name)
    return names


# Arguments that get_config leaves out. ComplexConv1D/2D/3D pass rank to
# ComplexConv themselves, and ComplexConv1D also data_format, so the config
# must not repeat them. They accept seed but never use it, so there is
# nothing to store yet.
NOT_IN_CONFIG = {
    conn.ComplexConv1D: {"rank", "data_format", "seed"},
    conn.ComplexConv2D: {"rank", "seed"},
    conn.ComplexConv3D: {"rank", "seed"},
}


class TestConfigCompleteness(unittest.TestCase):
    """get_config stores every argument that __init__ takes"""

    def test_every_argument_in_config(self):
        checked = set()
        for _, make_layer in LAYERS.values():
            layer = make_layer()
            cls = type(layer)
            if cls in checked:
                continue
            checked.add(cls)
            with self.subTest(cls.__name__):
                missing = init_parameters(cls) - set(layer.get_config())
                # assertEqual rather than a subset check, so the allow-list has
                # to shrink once a gap is closed.
                self.assertEqual(missing, NOT_IN_CONFIG.get(cls, set()))


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
                restored = loaded.get_weights()
                self.assertEqual(len(restored), len(model.get_weights()))
                for w_loaded, w_saved in zip(restored, model.get_weights()):
                    np.testing.assert_array_equal(w_loaded, w_saved)

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
