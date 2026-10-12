#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import re
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# Keras 3 warns whenever a layer receives `input_shape` or `input_dim` and
# recommends `keras.Input(shape)` instead. The complex layers keep accepting
# both arguments, as Keras 2 documented them, and Keras 3's Sequential still
# turns them into an input layer, so for these layers the warning is noise.
_INPUT_SHAPE_WARNING = re.escape("Do not pass an `input_shape`/`input_dim` argument")

# Only Keras 3 emits that warning. Keras 3 has `keras.version()`; Keras 2 and
# tf-keras do not.
_KERAS_3 = hasattr(tf.keras, "version") and int(tf.keras.version().split(".")[0]) >= 3


@contextlib.contextmanager
def _ignore_input_shape_warning(kwargs):
    """Ignore Keras 3's input_shape/input_dim warning inside the block.

    The layers wrap their call to ``Layer.__init__(**kwargs)`` with it. The
    warning filters change only when the warning can fire: on Keras 3, and only
    when kwargs holds ``input_shape`` or ``input_dim``. Otherwise the block
    does nothing.

    Trade-off: ``warnings.catch_warnings`` makes Python forget which warnings
    it has already shown. So on Keras 3, every layer built with
    ``input_shape`` or ``input_dim`` lets warnings that normally show once per
    place in the code show again. A filter installed once at import would avoid
    that, but any ``warnings.simplefilter("always")`` set later would override
    it and bring the input_shape warning back. While the filter applies it also
    changes the process-wide filters, like any use of ``catch_warnings``, so it
    is not thread-safe.
    """
    if not _KERAS_3 or not ("input_shape" in kwargs or "input_dim" in kwargs):
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_INPUT_SHAPE_WARNING, category=UserWarning
        )
        yield


#
# GetReal/GetImag Lambda layer Implementation
#


def get_realpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == "channels_first" and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, :input_dim]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, :input_dim]
    elif ndim == 4:
        return x[:, :, :, :input_dim]
    elif ndim == 5:
        return x[:, :, :, :, :input_dim]


def get_imagpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == "channels_first" and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, input_dim:]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, input_dim:]
    elif ndim == 4:
        return x[:, :, :, input_dim:]
    elif ndim == 5:
        return x[:, :, :, :, input_dim:]


def get_abs(x):
    real = get_realpart(x)
    imag = get_imagpart(x)

    return K.sqrt(real * real + imag * imag)


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    image_format = K.image_data_format()
    ndim = len(returned_shape)

    if (image_format == "channels_first" and ndim != 3) or ndim == 2:
        axis = 1
    else:
        axis = -1

    returned_shape[axis] = returned_shape[axis] // 2

    return tuple(returned_shape)


# _compute_fans is different in keras-2 keras.initializers and tensorflow.python.ops.init_ops
# this is the implementation copied from keras-2:
def _compute_fans(shape, data_format="channels_last"):
    """Computes the number of input and output units for a weight shape.
    # Arguments

        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).
    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.
    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == "channels_first":
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == "channels_last":
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError("Invalid data_format: " + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


class GetReal(Layer):
    def call(self, inputs):
        return get_realpart(inputs)

    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)


class GetImag(Layer):
    def call(self, inputs):
        return get_imagpart(inputs)

    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)


class GetAbs(Layer):
    def call(self, inputs):
        return get_abs(inputs)

    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
