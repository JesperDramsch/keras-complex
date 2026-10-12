#!/usr/bin/env python

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for keras-complex; the module docstring lists the changes.

"""Argument and shape helpers for the convolution layers.

keras-complex used to import these from ``tensorflow.python.keras.utils``, a
private copy of Keras 2 inside TensorFlow. Neither Keras 2 nor Keras 3 makes
them public, so the package keeps its own copy. They are derived from
TensorFlow's functions and stay under the Apache License 2.0 (see the header
above, LICENSE.md and LICENSE-APACHE), with two changes: invalid padding
raises ValueError instead of failing an assert, and
``normalize_data_format`` reads the image data format from the public Keras
backend, so it follows ``tf.keras.backend.set_image_data_format``.
"""

from tensorflow.keras import backend as K


def normalize_tuple(value, n, name):
    """Turn an integer or an iterable of integers into a tuple of length n.

    Arguments:
        value: An integer, or an iterable of n integers.
        n: The length of the returned tuple.
        name: The argument name used in error messages, e.g. "kernel_size".

    Returns:
        A tuple of n values.

    Raises:
        ValueError: If value is neither an integer nor an iterable of n
            integers.
    """
    if isinstance(value, int):
        return (value,) * n
    message = (
        f"The `{name}` argument must be a tuple of {n} integers. Received: {value}"
    )
    try:
        value_tuple = tuple(value)
    except TypeError:
        raise ValueError(message) from None
    if len(value_tuple) != n:
        raise ValueError(message)
    for single_value in value_tuple:
        try:
            int(single_value)
        except (ValueError, TypeError):
            raise ValueError(
                f"{message} including element {single_value} of type "
                f"{type(single_value)}"
            ) from None
    return value_tuple


def normalize_padding(value):
    """Return the padding in lower case; lists and tuples pass through.

    Raises:
        ValueError: If value is not "valid", "same" or "causal".
    """
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {"valid", "same", "causal"}:
        raise ValueError(
            'The `padding` argument must be a list/tuple or one of "valid", '
            f'"same" (or "causal", only for `Conv1D`). Received: {value}'
        )
    return padding


def normalize_data_format(value):
    """Return "channels_first" or "channels_last".

    None selects the image data format from the Keras configuration.

    Raises:
        ValueError: If value is not one of the two formats.
    """
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            'The `data_format` argument must be one of "channels_first", '
            f'"channels_last". Received: {value}'
        )
    return data_format


def _check_padding(padding, allowed):
    if padding not in allowed:
        raise ValueError(
            f"The `padding` argument must be one of {sorted(allowed)}. "
            f"Received: {padding}"
        )


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Return the output length of a convolution along one axis.

    Arguments:
        input_length: Integer, or None if the length is unknown.
        filter_size: Integer.
        padding: One of "same", "valid", "full" or "causal".
        stride: Integer.
        dilation: Integer dilation rate.

    Returns:
        The output length, or None if input_length is None.
    """
    if input_length is None:
        return None
    _check_padding(padding, {"same", "valid", "full", "causal"})
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ("same", "causal"):
        output_length = input_length
    elif padding == "valid":
        output_length = input_length - dilated_filter_size + 1
    else:  # full
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def deconv_output_length(
    input_length, filter_size, padding, output_padding=None, stride=0, dilation=1
):
    """Return the output length of a transposed convolution along one axis.

    Arguments:
        input_length: Integer, or None if the length is unknown.
        filter_size: Integer.
        padding: One of "same", "valid" or "full".
        output_padding: Integer padding added to the output, or None to infer
            the output length.
        stride: Integer.
        dilation: Integer dilation rate.

    Returns:
        The output length, or None if input_length is None.
    """
    _check_padding(padding, {"same", "valid", "full"})
    if input_length is None:
        return None

    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    if output_padding is None:
        if padding == "valid":
            return input_length * stride + max(filter_size - stride, 0)
        if padding == "full":
            return input_length * stride - (stride + filter_size - 2)
        return input_length * stride  # same

    if padding == "same":
        pad = filter_size // 2
    elif padding == "valid":
        pad = 0
    else:  # full
        pad = filter_size - 1
    return (input_length - 1) * stride + filter_size - 2 * pad + output_padding
