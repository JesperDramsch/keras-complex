#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import re
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_registered_object, register_keras_serializable

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


def get_initializer(identifier):
    """Resolve an initialiser like ``initializers.get``, complexnn ones included.

    The legacy .h5 loader looks a serialised initialiser up by its bare class
    name, while complexnn registers its classes as ``complexnn>ClassName``, so
    a ``ComplexInit`` saved with its config would not load. When Keras cannot
    resolve such a config, look it up by its registered name instead.
    """
    try:
        return initializers.get(identifier)
    except (ValueError, TypeError):
        if isinstance(identifier, dict):
            registered = str(identifier.get("registered_name") or "")
            cls = get_registered_object(registered) if registered else None
            if cls is not None and registered.startswith("complexnn>"):
                return cls.from_config(identifier.get("config") or {})
        raise


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

    Arguments:
        layer: The layer doing the check, named in the error message.
        input_shape: Shape of the layer input, including the batch axis.
        axis: Integer, the axis that holds the real and imaginary parts.
        hint: Optional sentence added to the end of the error message.

    Raises:
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

    Arguments:
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).

    Returns:
        A tuple of scalars, `(fan_in, fan_out)`.

    Raises:
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


def _check_part_shape(layer, input_shape):
    """Check the axis that `get_realpart` and `get_imagpart` split, if known."""
    shape = tuple(input_shape)
    axis = _complex_part_axis(len(shape))
    # An undefined size is split from the dynamic shape in `call`, which
    # works on Keras 2, so only check sizes that are known here.
    if shape[axis] is not None:
        _check_complex_axis(layer, shape, axis)


class _ComplexPart(Layer):
    """Base class for the layers that split their input into its two parts.

    These layers have no weights and accept inputs of any width and rank, so
    one instance can be called on several tensors. Keras only builds a layer
    on its first call, so the layout is also checked in `call`, which covers
    eager tensors and symbolic tensors on Keras 2, and in
    `compute_output_shape`, which Keras 3 uses for symbolic tensors instead
    of tracing `call`.
    """

    def build(self, input_shape):
        _check_part_shape(self, input_shape)
        super(_ComplexPart, self).build(input_shape)

    def call(self, inputs):
        _check_part_shape(self, inputs.shape)
        return self._split(inputs)

    def compute_output_shape(self, input_shape):
        _check_part_shape(self, input_shape)
        return getpart_output_shape(input_shape)


@register_keras_serializable(package="complexnn")
class GetReal(_ComplexPart):
    def _split(self, inputs):
        return get_realpart(inputs)


@register_keras_serializable(package="complexnn")
class GetImag(_ComplexPart):
    def _split(self, inputs):
        return get_imagpart(inputs)


@register_keras_serializable(package="complexnn")
class GetAbs(_ComplexPart):
    def _split(self, inputs):
        return get_abs(inputs)
