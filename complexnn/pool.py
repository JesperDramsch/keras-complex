#!/usr/bin/env python

import numpy as np
import tensorflow.keras as KE
import tensorflow.keras.backend as KB
import tensorflow.keras.layers as KL
from tensorflow.keras.utils import register_keras_serializable


def _static_shape(x):
    """Return the static shape of `x`, requiring every spatial dimension to be known"""
    xshape = tuple(x.shape)
    if any(dim is None for dim in xshape[1:]):
        raise ValueError(
            "Spectral pooling needs fully defined spatial dimensions, got shape "
            f"{xshape}"
        )
    return xshape


#
# Spectral Pooling Layer
#


@register_keras_serializable(package="complexnn")
class SpectralPooling1D(KL.Layer):
    def __init__(self, topf=None, gamma=None, **kwargs):
        super().__init__(**kwargs)
        if topf is not None:
            self.topf = (int(topf[0]) // 2,)
        elif gamma is not None:
            self.gamma = (float(gamma[0]) / 2,)
        else:
            raise RuntimeError("Must provide either topf= or gamma= !")

    def call(self, x, mask=None):
        xshape = _static_shape(x)
        if hasattr(self, "topf"):
            topf = self.topf
        else:
            if KB.image_data_format() == "channels_first":
                topf = (int(self.gamma[0] * xshape[2]),)
            else:
                topf = (int(self.gamma[0] * xshape[1]),)

        if KB.image_data_format() == "channels_first":
            if topf[0] > 0 and xshape[2] >= 2 * topf[0]:
                mask = (
                    [1] * (topf[0]) + [0] * (xshape[2] - 2 * topf[0]) + [1] * (topf[0])
                )
                mask = [[mask]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0, 1, 2))
                mask = KB.constant(mask)
                x *= mask
        else:
            if topf[0] > 0 and xshape[1] >= 2 * topf[0]:
                mask = (
                    [1] * (topf[0]) + [0] * (xshape[1] - 2 * topf[0]) + [1] * (topf[0])
                )
                mask = [[mask]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0, 2, 1))
                mask = KB.constant(mask)
                x *= mask

        return x


@register_keras_serializable(package="complexnn")
class SpectralPooling2D(KL.Layer):
    def __init__(self, topf=None, gamma=None, **kwargs):
        super().__init__(**kwargs)
        if topf is not None:
            self.topf = (int(topf[0]) // 2, int(topf[1]) // 2)
        elif gamma is not None:
            self.gamma = (float(gamma[0]) / 2, float(gamma[1]) / 2)
        else:
            raise RuntimeError("Must provide either topf= or gamma= !")

    def call(self, x, mask=None):
        xshape = _static_shape(x)
        if hasattr(self, "topf"):
            topf = self.topf
        else:
            if KB.image_data_format() == "channels_first":
                topf = (int(self.gamma[0] * xshape[2]), int(self.gamma[1] * xshape[3]))
            else:
                topf = (int(self.gamma[0] * xshape[1]), int(self.gamma[1] * xshape[2]))

        if KB.image_data_format() == "channels_first":
            if topf[0] > 0 and xshape[2] >= 2 * topf[0]:
                mask = (
                    [1] * (topf[0]) + [0] * (xshape[2] - 2 * topf[0]) + [1] * (topf[0])
                )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0, 1, 3, 2))
                mask = KB.constant(mask)
                x *= mask
            if topf[1] > 0 and xshape[3] >= 2 * topf[1]:
                mask = (
                    [1] * (topf[1]) + [0] * (xshape[3] - 2 * topf[1]) + [1] * (topf[1])
                )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0, 1, 2, 3))
                mask = KB.constant(mask)
                x *= mask
        else:
            if topf[0] > 0 and xshape[1] >= 2 * topf[0]:
                mask = (
                    [1] * (topf[0]) + [0] * (xshape[1] - 2 * topf[0]) + [1] * (topf[0])
                )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0, 3, 1, 2))
                mask = KB.constant(mask)
                x *= mask
            if topf[1] > 0 and xshape[2] >= 2 * topf[1]:
                mask = (
                    [1] * (topf[1]) + [0] * (xshape[2] - 2 * topf[1]) + [1] * (topf[1])
                )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0, 1, 3, 2))
                mask = KB.constant(mask)
                x *= mask

        return x


if __name__ == "__main__":
    import sys

    import cv2
    import fft as CF

    import __main__ as SP

    # Build Model
    x = i = KL.Input(shape=(6, 512, 512))
    f = CF.FFT2()(x)
    p = SP.SpectralPooling2D(gamma=[0.15, 0.15])(f)
    o = CF.IFFT2()(p)

    model = KE.Model([i], [f, p, o])
    model.compile("sgd", "mse")

    # Use it
    img = cv2.imread(sys.argv[1])
    imgBatch = img[np.newaxis, ...].transpose((0, 3, 1, 2))
    imgBatch = np.concatenate([imgBatch, np.zeros_like(imgBatch)], axis=1)
    f, p, o = model.predict(imgBatch)
    ffted = np.sqrt(np.sum(f[:, :3] ** 2 + f[:, 3:] ** 2, axis=1))
    ffted = ffted.transpose((1, 2, 0)) / 255
    pooled = np.sqrt(np.sum(p[:, :3] ** 2 + p[:, 3:] ** 2, axis=1))
    pooled = pooled.transpose((1, 2, 0)) / 255
    filtered = np.clip(o, 0, 255).transpose((0, 2, 3, 1))[0, :, :, :3].astype("uint8")

    # Display it
    cv2.imshow("Original", img)
    cv2.imshow("FFT", ffted)
    cv2.imshow("Pooled", pooled)
    cv2.imshow("Filtered", filtered)
    cv2.waitKey(0)
