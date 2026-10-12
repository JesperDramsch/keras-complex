import unittest

import tensorflow as tf

import complexnn as conn
from complexnn import conv_utils


class TestNormalizeArguments(unittest.TestCase):
    """Argument helpers that replace TensorFlow's private conv_utils"""

    def test_normalize_tuple(self):
        self.assertEqual(conv_utils.normalize_tuple(3, 2, "kernel_size"), (3, 3))
        self.assertEqual(conv_utils.normalize_tuple([1, 2], 2, "strides"), (1, 2))

    def test_normalize_tuple_rejects_bad_values(self):
        for value in ((1, 2, 3), "ab", None, (1, "a")):
            with self.assertRaisesRegex(ValueError, "`kernel_size` argument"):
                conv_utils.normalize_tuple(value, 2, "kernel_size")

    def test_normalize_padding(self):
        self.assertEqual(conv_utils.normalize_padding("SAME"), "same")
        self.assertEqual(conv_utils.normalize_padding("causal"), "causal")
        self.assertEqual(conv_utils.normalize_padding((1, 1)), (1, 1))
        with self.assertRaisesRegex(ValueError, "`padding` argument"):
            conv_utils.normalize_padding("full")

    def test_normalize_data_format(self):
        self.assertEqual(
            conv_utils.normalize_data_format("Channels_First"), "channels_first"
        )
        with self.assertRaisesRegex(ValueError, "`data_format` argument"):
            conv_utils.normalize_data_format("nhwc")


class TestImageDataFormat(unittest.TestCase):
    """data_format=None follows the public Keras image data format setting"""

    def setUp(self):
        previous = tf.keras.backend.image_data_format()
        self.addCleanup(tf.keras.backend.set_image_data_format, previous)
        tf.keras.backend.set_image_data_format("channels_first")

    def test_normalize_data_format_reads_keras_config(self):
        self.assertEqual(conv_utils.normalize_data_format(None), "channels_first")

    def test_complex_conv2d_default_data_format(self):
        self.assertEqual(conn.ComplexConv2D(2, 3).data_format, "channels_first")

    def test_complex_conv1d_stays_channels_last(self):
        self.assertEqual(conn.ComplexConv1D(2, 3).data_format, "channels_last")


class TestOutputLengths(unittest.TestCase):
    """Output lengths match the public Keras convolution layers"""

    def test_conv_output_length_matches_keras(self):
        cases = [
            (length, kernel, padding, stride, 1)
            for length in (5, 6, 7, 8)
            for kernel in (1, 2, 3, 4)
            for padding in ("valid", "same", "causal")
            for stride in (1, 2, 3)
        ] + [
            (length, kernel, padding, 1, dilation)
            for length in (7, 8)
            for kernel in (2, 3)
            for padding in ("valid", "same", "causal")
            for dilation in (2, 3)
        ]
        for length, kernel, padding, stride, dilation in cases:
            with self.subTest(
                length=length,
                kernel=kernel,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ):
                layer = tf.keras.layers.Conv1D(
                    1, kernel, strides=stride, padding=padding, dilation_rate=dilation
                )
                expected = layer.compute_output_shape((None, length, 1))[1]
                calc = conv_utils.conv_output_length(
                    length, kernel, padding, stride, dilation
                )
                self.assertEqual(calc, expected)

    def test_deconv_output_length_matches_keras(self):
        cases = [
            (length, kernel, padding, stride, output_padding)
            for length in (5, 6)
            for kernel in (1, 2, 3, 4)
            for padding in ("valid", "same")
            for stride in (1, 2, 3)
            # Keras 3.3 rejects output_padding=0 in Conv1DTranspose, so the
            # Keras reference only sees values every supported release accepts.
            # test_zero_output_padding covers 0 with hand-computed lengths.
            for output_padding in (None, *range(1, stride))
        ]
        for length, kernel, padding, stride, output_padding in cases:
            with self.subTest(
                length=length,
                kernel=kernel,
                padding=padding,
                stride=stride,
                output_padding=output_padding,
            ):
                layer = tf.keras.layers.Conv1DTranspose(
                    1,
                    kernel,
                    strides=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                expected = layer.compute_output_shape((None, length, 1))[1]
                calc = conv_utils.deconv_output_length(
                    length,
                    kernel,
                    padding,
                    output_padding=output_padding,
                    stride=stride,
                )
                self.assertEqual(calc, expected)

    def test_zero_output_padding(self):
        # (length - 1) * stride + kernel - 2 * pad + output_padding, where pad is
        # 0 for "valid" and kernel // 2 for "same".
        self.assertEqual(
            conv_utils.deconv_output_length(5, 3, "valid", output_padding=0, stride=2),
            11,
        )
        self.assertEqual(
            conv_utils.deconv_output_length(5, 3, "same", output_padding=0, stride=2),
            9,
        )

    def test_full_padding(self):
        self.assertEqual(conv_utils.conv_output_length(5, 3, "full", 1), 7)
        self.assertEqual(conv_utils.conv_output_length(5, 3, "full", 2, 2), 5)
        self.assertEqual(conv_utils.deconv_output_length(5, 3, "full", stride=2), 7)
        self.assertEqual(
            conv_utils.deconv_output_length(5, 3, "full", output_padding=1, stride=2),
            8,
        )

    def test_unknown_length(self):
        self.assertIsNone(conv_utils.conv_output_length(None, 3, "same", 1))
        self.assertIsNone(conv_utils.deconv_output_length(None, 3, "same", stride=2))

    def test_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "`padding` argument"):
            conv_utils.conv_output_length(5, 3, "reflect", 1)
        with self.assertRaisesRegex(ValueError, "`padding` argument"):
            conv_utils.deconv_output_length(5, 3, "causal", stride=1)
