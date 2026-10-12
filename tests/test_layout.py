import unittest

import numpy as np

import complexnn as conn


def to_layout(z):
    """Stack the real parts, then the imaginary parts, as the README describes."""
    return np.concatenate([z.real, z.imag], axis=-1).astype("float32")


def from_layout(x):
    """Turn a tensor in that layout back into complex numbers."""
    real, imag = np.split(np.asarray(x), 2, axis=-1)
    return real + 1j * imag


class TestDataLayout(unittest.TestCase):
    """The layers read and write the layout that the README and the docs describe"""

    def setUp(self):
        self.rng = np.random.default_rng(15)

    def random_complex(self, shape):
        return self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)

    def test_get_parts(self):
        z = self.random_complex((4, 6, 6, 3))
        x = to_layout(z)
        np.testing.assert_allclose(np.asarray(conn.GetReal()(x)), z.real, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(conn.GetImag()(x)), z.imag, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(conn.GetAbs()(x)), np.abs(z), rtol=1e-5)

    def test_conv_multiplies_by_its_complex_kernel(self):
        """A 1x1 complex convolution multiplies each pixel by a complex matrix"""
        z = self.random_complex((4, 6, 6, 3))
        layer = conn.ComplexConv2D(filters=2, kernel_size=1, use_bias=False)
        outputs = from_layout(layer(to_layout(z)))
        # The kernel holds the real parts of the complex kernel in its first
        # `filters` output columns and the imaginary parts in the rest.
        kernel = np.asarray(layer.kernel)[0, 0]
        weights = kernel[:, :2] + 1j * kernel[:, 2:]
        np.testing.assert_allclose(outputs, z @ weights, rtol=1e-4, atol=1e-5)

    def test_layers_are_complex_linear(self):
        """Multiplying the inputs by i multiplies the outputs by i"""
        cases = [
            ("Conv1D", conn.ComplexConv1D(2, 3, use_bias=False), (4, 8, 3)),
            ("Conv2D", conn.ComplexConv2D(2, 3, use_bias=False), (4, 6, 6, 3)),
            ("Conv3D", conn.ComplexConv3D(2, 3, use_bias=False), (2, 5, 5, 5, 1)),
            (
                "Conv2D transposed",
                conn.ComplexConv2D(
                    2, 3, strides=2, padding="same", use_bias=False, transposed=True
                ),
                (4, 4, 4, 3),
            ),
            ("Dense", conn.ComplexDense(5, use_bias=False), (4, 3)),
        ]
        for name, layer, shape in cases:
            with self.subTest(layer=name):
                z = self.random_complex(shape)
                outputs = from_layout(layer(to_layout(z)))
                rotated = from_layout(layer(to_layout(1j * z)))
                np.testing.assert_allclose(rotated, 1j * outputs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
