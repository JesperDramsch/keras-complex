import unittest

import complexnn as conn
import numpy as np
import tensorflow as tf


class TestComplexBatchNormalization(unittest.TestCase):
    """Moving statistics and train/inference behaviour of ComplexBatchNormalization"""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = (rng.standard_normal((32, 8)) * 2.0 + 3.0).astype("float32")
        self.bn = conn.ComplexBN(momentum=0.9)
        inputs = tf.keras.Input((8,))
        self.model = tf.keras.Model(inputs, self.bn(inputs))

    def test_training_call_updates_moving_mean(self):
        """A training call moves the moving mean towards the batch mean"""
        before = self.bn.moving_mean.numpy()
        self.model(self.x, training=True)
        expected = 0.9 * before + 0.1 * self.x.mean(axis=0)
        np.testing.assert_allclose(self.bn.moving_mean.numpy(), expected, atol=1e-5)

    def test_inference_call_leaves_moving_mean(self):
        """An inference call does not touch the moving statistics"""
        before = self.bn.moving_mean.numpy()
        self.model(self.x, training=False)
        np.testing.assert_array_equal(self.bn.moving_mean.numpy(), before)

    def test_training_output_uses_batch_statistics(self):
        """Training output is centred with the batch mean"""
        out = self.model(self.x, training=True).numpy()
        self.assertLess(float(np.abs(out.mean(axis=0)).max()), 1e-4)

    def test_inference_output_uses_moving_statistics(self):
        """Inference output changes once the moving statistics change"""
        before = self.model(self.x, training=False).numpy()
        self.model(self.x, training=True)
        after = self.model(self.x, training=False).numpy()
        self.assertFalse(np.allclose(before, after))

    def test_fit_updates_moving_mean(self):
        """The compiled training step updates the moving statistics"""
        self.model.compile(optimizer=tf.keras.optimizers.SGD(0.0), loss="mse")
        before = self.bn.moving_mean.numpy()
        self.model.fit(
            self.x, np.zeros_like(self.x), batch_size=32, epochs=1, verbose=0
        )
        self.assertFalse(np.allclose(self.bn.moving_mean.numpy(), before))


class TestComplexBatchNormalizationShape(unittest.TestCase):
    """ComplexBatchNormalization keeps the input shape"""

    def test_compute_output_shape(self):
        layer = conn.ComplexBN()
        for shape in [(None, 8), (None, 4, 4, 8)]:
            with self.subTest(shape=shape):
                self.assertEqual(tuple(layer.compute_output_shape(shape)), shape)
