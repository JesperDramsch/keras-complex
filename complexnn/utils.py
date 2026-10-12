#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Layer

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


def _complex_part_axis(ndim):
    """Axis that `get_realpart` and `get_imagpart` split for a rank `ndim` input."""
    if (K.image_data_format() == "channels_first" and ndim != 3) or ndim == 2:
        return 1
    return -1


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    axis = _complex_part_axis(len(returned_shape))

    returned_shape[axis] = returned_shape[axis] // 2

    return tuple(returned_shape)


def _check_complex_axis(layer, input_shape, axis, hint=""):
    """Check that an axis of the input can hold real and imaginary parts.

    The layers store complex values as real tensors: along one axis, usually
    the channel axis, the first half holds the real parts and the second half
    the imaginary parts. That axis must therefore have an even size of at
    least 2.

    # Arguments
        layer: The layer doing the check, named in the error message.
        input_shape: Shape of the layer input, including the batch axis.
        axis: Integer, the axis that holds the real and imaginary parts.
        hint: Optional sentence added to the end of the error message.

    # Raises
        ValueError: If the size of `axis` is undefined, odd or smaller than 2.
    """
    shape = tuple(input_shape)
    size = shape[axis]
    if size is not None and size >= 2 and size % 2 == 0:
        return
    if size is None:
        received = f"the size of axis {axis} is not defined"
    else:
        received = f"axis {axis} has size {size}"
    message = (
        f"{type(layer).__name__} expects the real and imaginary parts of its "
        f"input stacked along axis {axis}: the first half of that axis holds "
        "the real parts and the second half the imaginary parts, so its size "
        f"must be even and at least 2. Received an input with shape {shape}, "
        f"where {received}."
    )
    if hint:
        message = f"{message} {hint}"
    raise ValueError(message)


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


class _ComplexPart(Layer):
    """Base class for the layers that split their input into its two parts."""

    def build(self, input_shape):
        axis = _complex_part_axis(len(input_shape))
        # An undefined size is split from the dynamic shape in `call`, which
        # works on Keras 2, so only check sizes that are known here.
        if tuple(input_shape)[axis] is not None:
            _check_complex_axis(self, input_shape, axis)
        super(_ComplexPart, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)


class GetReal(_ComplexPart):
    def call(self, inputs):
        return get_realpart(inputs)


class GetImag(_ComplexPart):
    def call(self, inputs):
        return get_imagpart(inputs)


class GetAbs(_ComplexPart):
    def call(self, inputs):
        return get_abs(inputs)
