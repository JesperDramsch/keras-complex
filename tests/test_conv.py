import unittest

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

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

    def test_spectral_parametrization_not_implemented(self):
        """spectral_parametrization=True raises NotImplementedError when called"""
        for layer, shape in (
            (conn.ComplexConv1D(2, 3, spectral_parametrization=True), (2, 8, 4)),
            (conn.ComplexConv2D(2, 3, spectral_parametrization=True), (2, 8, 8, 4)),
        ):
            with self.subTest(layer=type(layer).__name__):
                with self.assertRaises(NotImplementedError):
                    layer(np.zeros(shape, "float32"))


class TestConvConfig(unittest.TestCase):
    """Conv layer configs rebuild the same layer"""

    def test_conv1d_from_config(self):
        """ComplexConv1D sets data_format itself, so its config must not repeat it"""
        layer = conn.ComplexConv1D(filters=2, kernel_size=3, padding="same")
        config = layer.get_config()
        clone = conn.ComplexConv1D.from_config(config)
        self.assertEqual(clone.get_config(), config)
        self.assertEqual(clone.data_format, "channels_last")

    def test_epsilon_in_config(self):
        """epsilon changes the normalised kernel, so from_config must keep it"""
        layer = conn.ComplexConv2D(
            filters=2, kernel_size=3, normalize_weight=True, epsilon=1e-3
        )
        clone = conn.ComplexConv2D.from_config(layer.get_config())
        self.assertEqual(clone.epsilon, 1e-3)

    def test_complex_initializer_instance_in_config(self):
        """A complexnn initialiser instance keeps its own config, seed included"""
        initializers = [
            conn.ComplexInit(
                kernel_size=(3, 3),
                input_dim=2,
                weight_dim=2,
                nb_filters=2,
                criterion="glorot",
                seed=5,
            ),
            conn.ComplexIndependentFilters(
                kernel_size=(3, 3),
                input_dim=2,
                weight_dim=2,
                nb_filters=2,
                criterion="he",
                seed=5,
            ),
        ]
        for init in initializers:
            with self.subTest(type(init).__name__):
                layer = conn.ComplexConv2D(
                    filters=2, kernel_size=3, kernel_initializer=init
                )
                config = layer.get_config()
                clone = conn.ComplexConv2D.from_config(config)
                self.assertIs(type(clone.kernel_initializer), type(init))
                self.assertEqual(
                    clone.kernel_initializer.get_config(), init.get_config()
                )
                self.assertEqual(clone.get_config(), config)

                layer.build((None, 6, 6, 4))
                clone.build((None, 6, 6, 4))
                np.testing.assert_array_equal(
                    np.asarray(clone.kernel), np.asarray(layer.kernel)
                )


class TestTransposedConvTraining(unittest.TestCase):
    """Transposed convolutions train with unknown batch and spatial sizes"""

    def _fit(self, layer, input_shape, batches):
        inputs = Input(shape=input_shape)
        model = Model(inputs=inputs, outputs=layer(inputs))
        model.compile(optimizer="sgd", loss="mse")
        for x in batches:
            y = model.predict(x, verbose=0)
            model.fit(x, np.zeros_like(y), batch_size=2, epochs=1, verbose=0)
        return model

    def test_conv1d_transposed_fit(self):
        rng = np.random.default_rng(0)
        model = self._fit(
            conn.ComplexConv1D(2, 3, strides=2, padding="same", transposed=True),
            (16, 4),
            [rng.standard_normal((5, 16, 4)).astype("float32")],
        )
        self.assertEqual(model.output_shape, (None, 32, 4))

    def test_conv1d_transposed_variable_length(self):
        rng = np.random.default_rng(0)
        model = self._fit(
            conn.ComplexConv1D(2, 3, strides=2, padding="same", transposed=True),
            (None, 4),
            [rng.standard_normal((3, n, 4)).astype("float32") for n in (8, 13)],
        )
        x = rng.standard_normal((1, 11, 4)).astype("float32")
        self.assertEqual(model.predict(x, verbose=0).shape, (1, 22, 4))

    def test_conv2d_transposed_variable_size(self):
        rng = np.random.default_rng(0)
        model = self._fit(
            conn.ComplexConv2D(2, 3, strides=2, padding="valid", transposed=True),
            (None, None, 4),
            [rng.standard_normal((3, h, w, 4)).astype("float32")
             for h, w in ((6, 6), (5, 9))],
        )
        x = rng.standard_normal((1, 5, 9, 4)).astype("float32")
        self.assertEqual(model.predict(x, verbose=0).shape, (1, 11, 19, 4))

    def test_conv3d_transposed_fit(self):
        rng = np.random.default_rng(0)
        model = self._fit(
            conn.ComplexConv3D(2, 3, strides=2, padding="same", transposed=True),
            (4, 4, 4, 4),
            [rng.standard_normal((3, 4, 4, 4, 4)).astype("float32")],
        )
        self.assertEqual(model.output_shape, (None, 8, 8, 8, 4))
